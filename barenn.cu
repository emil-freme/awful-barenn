#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace Barenn {

// Cada imagem tem 64 atributos e pertence a uma das dez classes.
constexpr int entradas = 64;
constexpr int classes = 10;
// Rede densa com dimensões 64 -> 32 -> 16 -> 10.
constexpr int oculta1 = 32;
constexpr int oculta2 = 16;

void verificar(cudaError_t erro) {
    if (erro != cudaSuccess) {
        throw runtime_error(cudaGetErrorString(erro));
    }
}

template <typename T>
class Buffer {
    T* ponteiro = nullptr;
    size_t bytes;

public:
    explicit Buffer(size_t elementos) : bytes(elementos * sizeof(T)) {
        if (bytes != 0) {
            verificar(cudaMalloc(&ponteiro, bytes));
        }
    }
    ~Buffer() {
        if (ponteiro) {
            cudaFree(ponteiro);
        }
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    T* data() const { return ponteiro; }

    void enviar(const T* origem) const {
        verificar(cudaMemcpy(ponteiro, origem, bytes, cudaMemcpyHostToDevice));
    }

    void receber(T* destino) const {
        verificar(cudaMemcpy(destino, ponteiro, bytes, cudaMemcpyDeviceToHost));
    }
};

using BufferGpu = Buffer<float>;

constexpr int bloco = 128;
int grade(int n) { return (n + bloco - 1) / bloco; }

// pesos[j * n_saidas + i], igual à CPU

// uma thread por neurônio
__global__ void forward_gpu(const float* entrada, const float* pesos,
                            const float* vies, int n_entradas, int n_saidas,
                            bool final, float* saida) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_saidas) return;
    float soma = vies[i];
    for (int j = 0; j < n_entradas; ++j) {
        soma += entrada[j] * pesos[j * n_saidas + i];
    }
    saida[i] = final ? soma : (0.0f < soma ? soma : 0.0f);
}

// Treino inteiro em um kernel de 1 bloco (o SGD é sequencial). O modelo fica
// na memória compartilhada, com uma coluna extra por linha (passo O + 1) para
// evitar conflito de banco no backward.
// threads do bloco de treino (>= 64); 256 foi o melhor entre 64, 128, 256 e 512 (ver issue #11)
// ex.: make barenn_gpu DEFS=-DBLOCO_TREINO=64
#ifndef BLOCO_TREINO
#define BLOCO_TREINO 256
#endif
constexpr int bloco_treino = BLOCO_TREINO;
static_assert(bloco_treino >= entradas, "o bloco carrega uma entrada por thread");

template <int I, int O>
__device__ void carregar(const float* global, float* compartilhada) {
    for (int k = threadIdx.x; k < I * O; k += blockDim.x) {
        compartilhada[(k / O) * (O + 1) + k % O] = global[k];
    }
}

template <int I, int O>
__device__ void devolver(const float* compartilhada, float* global) {
    for (int k = threadIdx.x; k < I * O; k += blockDim.x) {
        global[k] = compartilhada[(k / O) * (O + 1) + k % O];
    }
}

template <int I, int O>
__device__ float neuronio(const float* x, const float* w, const float* b, int i) {
    float soma = b[i];
    for (int j = 0; j < I; ++j) {
        soma += x[j] * w[j * (O + 1) + i];
    }
    return soma;
}

template <int I, int O>
__device__ float retropropagar(const float* delta, const float* w, int i) {
    float soma = 0.0f;
    for (int j = 0; j < O; ++j) {
        soma += delta[j] * w[i * (O + 1) + j];
    }
    return soma;
}

template <int I, int O>
__device__ void atualizar(const float* x, const float* delta, float taxa,
                          float* w, float* b) {
    for (int k = threadIdx.x; k < I * O; k += blockDim.x) {
        const int j = k / O;
        const int i = k % O;
        w[j * (O + 1) + i] -= taxa * (delta[i] * x[j]);
    }
    if (threadIdx.x < O) {
        b[threadIdx.x] -= taxa * delta[threadIdx.x];
    }
}

__global__ void treinar_gpu(const float* imagens, const int* rotulos, int n,
                            int epocas, float taxa,
                            float* pesos1, float* vies1, float* pesos2, float* vies2,
                            float* pesos3, float* vies3, double* perdas) {
    __shared__ float w1[entradas * (oculta1 + 1)], b1[oculta1];
    __shared__ float w2[oculta1 * (oculta2 + 1)], b2[oculta2];
    __shared__ float w3[oculta2 * (classes + 1)], b3[classes];
    __shared__ float x[entradas];
    __shared__ int rotulo;
    __shared__ float z1[oculta1], a1[oculta1], z2[oculta2], a2[oculta2];
    __shared__ float d1[oculta1], d2[oculta2], d3[classes];

    const int t = threadIdx.x;
    carregar<entradas, oculta1>(pesos1, w1);
    carregar<oculta1, oculta2>(pesos2, w2);
    carregar<oculta2, classes>(pesos3, w3);
    if (t < oculta1) b1[t] = vies1[t];
    if (t < oculta2) b2[t] = vies2[t];
    if (t < classes) b3[t] = vies3[t];

    for (int epoca = 0; epoca < epocas; ++epoca) {
        double total = 0.0;  // só a thread 0 usa
        for (int amostra = 0; amostra < n; ++amostra) {
            if (t < entradas) x[t] = imagens[static_cast<size_t>(amostra) * entradas + t];
            if (t == 0) rotulo = rotulos[amostra];
            __syncthreads();

            // forward (ReLU nas ocultas)
            if (t < oculta1) {
                const float z = neuronio<entradas, oculta1>(x, w1, b1, t);
                z1[t] = z;
                a1[t] = 0.0f < z ? z : 0.0f;
            }
            __syncthreads();
            if (t < oculta2) {
                const float z = neuronio<oculta1, oculta2>(a1, w2, b2, t);
                z2[t] = z;
                a2[t] = 0.0f < z ? z : 0.0f;
            }
            __syncthreads();
            // última camada: o erro já é o delta
            if (t < classes) {
                d3[t] = neuronio<oculta2, classes>(a2, w3, b3, t)
                      - (t == rotulo ? 1.0f : 0.0f);
            }
            __syncthreads();

            if (t == 0) {
                float perda = 0.0f;
                for (int i = 0; i < classes; ++i) perda += 0.5f * d3[i] * d3[i];
                total += perda;
            }
            // backward com os pesos antigos
            if (t < oculta2) {
                d2[t] = retropropagar<oculta2, classes>(d3, w3, t)
                      * (z2[t] > 0.0f ? 1.0f : 0.0f);
            }
            __syncthreads();
            if (t < oculta1) {
                d1[t] = retropropagar<oculta1, oculta2>(d2, w2, t)
                      * (z1[t] > 0.0f ? 1.0f : 0.0f);
            }
            __syncthreads();

            atualizar<oculta2, classes>(a2, d3, taxa, w3, b3);
            atualizar<oculta1, oculta2>(a1, d2, taxa, w2, b2);
            atualizar<entradas, oculta1>(x, d1, taxa, w1, b1);
            __syncthreads();
        }
        if (t == 0) perdas[epoca] = total;
    }

    devolver<entradas, oculta1>(w1, pesos1);
    devolver<oculta1, oculta2>(w2, pesos2);
    devolver<oculta2, classes>(w3, pesos3);
    if (t < oculta1) vies1[t] = b1[t];
    if (t < oculta2) vies2[t] = b2[t];
    if (t < classes) vies3[t] = b3[t];
}

// pesos da camada na GPU; aqui ela só faz o forward da avaliação
class DenseLayer {
    int in_features;
    int out_features;
    bool final;
    BufferGpu weights;
    BufferGpu bias;
    BufferGpu out;

public:
    DenseLayer(int input_size, int output_size, bool is_final = false)
        : in_features(input_size), out_features(output_size), final(is_final),
          weights(static_cast<size_t>(input_size) * output_size), bias(output_size),
          out(output_size) {
        vector<float> pesos(static_cast<size_t>(in_features) * out_features);
        vector<float> vies(out_features, 0.0f);
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        for (int i = 0; i < out_features; ++i) {
            for (int j = 0; j < in_features; ++j) {
                pesos[j * out_features + i] = distribution(gen);
            }
        }
        weights.enviar(pesos.data());
        bias.enviar(vies.data());
    }

    float* pesos_gpu() const { return weights.data(); }
    float* vies_gpu() const { return bias.data(); }

    const float* Forward(const float* input_gpu) {
        forward_gpu<<<grade(out_features), bloco>>>(
            input_gpu, weights.data(), bias.data(), in_features, out_features,
            final, out.data());
        verificar(cudaGetLastError());
        return out.data();
    }
};

// imagens em um vetor só: a imagem i começa em i * entradas
struct Dados {
    vector<float> imagens;
    vector<int> rotulos;

    size_t size() const { return rotulos.size(); }
    const float* imagem(size_t i) const { return &imagens[i * entradas]; }
};

Dados ler_dados(const string& caminho) {
    ifstream arquivo(caminho);
    if (!arquivo) {
        throw runtime_error("Não foi possível abrir " + caminho);
    }

    Dados dados;
    string linha;
    size_t numero_linha = 0;
    while (getline(arquivo, linha)) {
        ++numero_linha;
        istringstream campos(linha);
        array<float, entradas> imagem;
        int lidos = 0;
        int rotulo = -1;
        // Cada atributo deve ser seguido por vírgula, inclusive o último.
        for (; lidos < entradas; ++lidos) {
            float valor;
            char separador;
            if (!(campos >> valor >> separador) || separador != ',' ||
                !isfinite(valor) || valor < 0 || valor > 16) {
                break;
            }
            // Converte os valores de [0, 16] para [0, 1].
            imagem[lidos] = valor / 16.0f;
        }
        // Depois dos atributos, deve restar apenas o rótulo de 0 a 9.
        if (lidos != entradas ||
            !(campos >> rotulo) ||
            rotulo < 0 || rotulo >= classes ||
            !(campos >> ws).eof()) {
            throw runtime_error(caminho + ":" + to_string(numero_linha) +
                " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
        }
        dados.imagens.insert(dados.imagens.end(), imagem.begin(), imagem.end());
        dados.rotulos.push_back(rotulo);
    }
    if (arquivo.bad() || dados.size() == 0) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }
    return dados;
}

// Calcula metade da soma dos erros quadráticos. O alvo vale 1 para a
// classe correta e 0 para as demais. O erro também é a derivada por escore.
float perda(const float* escores, int rotulo, float* gradiente = nullptr) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        if (gradiente) {
            gradiente[i] = erro;
        }
    }
    return total;
}

void relatar(const vector<float>& escores_todos, const Dados& teste) {
    // Linhas representam a classe esperada e colunas, a classe prevista.
    array<array<size_t, classes>, classes> confusoes{};
    struct Exemplo {
        int real;
        int previsto;
    };
    vector<Exemplo> exemplos;
    size_t acertos = 0;
    size_t exemplos_acerto = 0;
    size_t exemplos_erro = 0;
    double perda_total = 0;

    for (size_t i = 0; i < teste.size(); ++i) {
        const int rotulo = teste.rotulos[i];
        const float* escores = &escores_todos[i * classes];
        // O índice do maior escore indica a classe prevista.
        int previsto = static_cast<int>(max_element(escores, escores + classes) - escores);
        bool acertou = previsto == rotulo;
        acertos += acertou;
        ++confusoes[rotulo][previsto];
        perda_total += perda(escores, rotulo);

        // Guarda até três acertos e três erros para ilustrar a saída.
        size_t& exemplos_tipo = acertou ? exemplos_acerto : exemplos_erro;
        if (exemplos_tipo < 3) {
            exemplos.push_back({rotulo, previsto});
            ++exemplos_tipo;
        }
    }

    cout << "\nAVALIACAO FINAL\n"
         << "Acuracia " << setprecision(1) << 100.0 * acertos / teste.size()
         << "% | " << acertos << " acertos em " << teste.size() << " imagens\n"
         << "Perda media " << setprecision(4) << perda_total / teste.size() << "\n";

    cout << "\nALGUMAS INFERENCIAS\n";
    cout << "Esperado  Previsto\n";

    for (const auto& exemplo : exemplos) {
        cout << right << setw(8) << exemplo.real
             << "  " << setw(8) << exemplo.previsto << '\n';
    }

    struct Confusao {
        int real;
        int previsto;
        size_t ocorrencias;
    };
    vector<Confusao> erros;
    for (int real = 0; real < classes; ++real) {
        for (int previsto = 0; previsto < classes; ++previsto) {
            if (real != previsto && confusoes[real][previsto] > 0) {
                erros.push_back({real, previsto, confusoes[real][previsto]});
            }
        }
    }
    // Ordena os pares incorretos do mais frequente para o menos frequente.
    stable_sort(erros.begin(), erros.end(), [](const auto& a, const auto& b) {
        return a.ocorrencias > b.ocorrencias;
    });
    cout << "\nCONFUSOES MAIS FREQUENTES\n";

    if (erros.empty()) {
        cout << "Nenhuma neste conjunto.\n";
    } else {
        cout << "Esperado  Previsto  Ocorrencias\n";

        for (size_t i = 0; i < min<size_t>(3, erros.size()); ++i) {
            const auto& erro = erros[i];
            cout << right << setw(8) << erro.real
                << "  " << setw(8) << erro.previsto
                << "  " << setw(11) << erro.ocorrencias << '\n';
        }
    }
}

void avaliar(DenseLayer& l1, DenseLayer& l2, DenseLayer& l3, const Dados& teste) {
    vector<float> escores(teste.size() * classes);
    const auto inicio = chrono::steady_clock::now();
    BufferGpu teste_gpu(teste.imagens.size());
    teste_gpu.enviar(teste.imagens.data());
    for (size_t i = 0; i < teste.size(); ++i) {
        const float* s = l3.Forward(l2.Forward(l1.Forward(teste_gpu.data() + i * entradas)));
        verificar(cudaMemcpy(&escores[i * classes], s, classes * sizeof(float),
                             cudaMemcpyDeviceToHost));
    }
    const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;
    relatar(escores, teste);
    cout << "Tempo de avaliacao " << setprecision(6) << tempo.count() << " s\n";
}

} // namespace Barenn

int main(int argc, char** argv) {
    // épocas opcionais, para perfis curtos no nsys
    if (argc != 1 && argc != 3 && argc != 4) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste [epocas]]\n";
        return 1;
    }
    const string arquivo_treino = argc >= 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc >= 3 ? argv[2] : "optdigits.tes";

    try {
        int epocas = 51;
        if (argc == 4) {
            size_t lidos = 0;
            epocas = stoi(argv[3], &lidos);
            if (epocas <= 0 || argv[3][lidos] != '\0') {
                throw runtime_error("Informe um número inteiro positivo de épocas.");
            }
        }
        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }

        constexpr float taxa = 0.02f;
        // cria o contexto CUDA fora da medição
        Barenn::verificar(cudaFree(nullptr));
        Barenn::DenseLayer l1(Barenn::entradas, Barenn::oculta1);
        Barenn::DenseLayer l2(Barenn::oculta1, Barenn::oculta2);
        Barenn::DenseLayer l3(Barenn::oculta2, Barenn::classes, true);
        vector<double> perdas(epocas);
        const int n = static_cast<int>(treino.size());

        const auto inicio = chrono::steady_clock::now();
        Barenn::BufferGpu treino_gpu(treino.imagens.size());
        Barenn::Buffer<int> rotulos_gpu(treino.size());
        Barenn::Buffer<double> perdas_gpu(epocas);
        treino_gpu.enviar(treino.imagens.data());
        rotulos_gpu.enviar(treino.rotulos.data());
        // um lançamento só para todas as épocas
        Barenn::treinar_gpu<<<1, Barenn::bloco_treino>>>(
            treino_gpu.data(), rotulos_gpu.data(), n, epocas, taxa,
            l1.pesos_gpu(), l1.vies_gpu(), l2.pesos_gpu(), l2.vies_gpu(),
            l3.pesos_gpu(), l3.vies_gpu(), perdas_gpu.data());
        Barenn::verificar(cudaGetLastError());
        perdas_gpu.receber(perdas.data());
        for (double& perda : perdas) {
            perda /= treino.size();
        }
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        // Imprime as métricas e roda a avaliação após encerrar o cronômetro.
        cout << fixed << setprecision(4)
             << "TREINO | " << treino.size() << " imagens\n";
        for (int epoca = 0; epoca < epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << tempo.count() << " s\n";
        Barenn::avaliar(l1, l2, l3, teste);
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
