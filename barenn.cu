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

// forward de um lote: uma thread por (amostra, neurônio)
__global__ void forward_lote_gpu(const float* entrada, const float* pesos,
                                 const float* vies, int n_entradas, int n_saidas,
                                 bool final, int n_amostras, float* saida) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_amostras * n_saidas) return;
    const int amostra = k / n_saidas;
    const int i = k % n_saidas;
    const float* x = entrada + static_cast<size_t>(amostra) * n_entradas;
    float soma = vies[i];
    for (int j = 0; j < n_entradas; ++j) {
        soma += x[j] * pesos[j * n_saidas + i];
    }
    saida[k] = final ? soma : (0.0f < soma ? soma : 0.0f);
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

public:
    DenseLayer(int input_size, int output_size, bool is_final = false)
        : in_features(input_size), out_features(output_size), final(is_final),
          weights(static_cast<size_t>(input_size) * output_size), bias(output_size) {
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

    void ForwardLote(const float* entrada_gpu, float* saida_gpu, int n,
                     cudaStream_t stream) const {
        forward_lote_gpu<<<grade(n * out_features), bloco, 0, stream>>>(
            entrada_gpu, weights.data(), bias.data(), in_features, out_features,
            final, n, saida_gpu);
        verificar(cudaGetLastError());
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

void relatar(const float* escores_todos, const Dados& teste) {
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

// avaliação em pedaços alternados entre as streams; cada stream tem seus buffers
#ifndef NSTREAMS
#define NSTREAMS 2
#endif
#ifndef TAMANHO_PEDACO
#define TAMANHO_PEDACO 1024
#endif

struct BuffersStream {
    BufferGpu entrada{static_cast<size_t>(TAMANHO_PEDACO) * entradas};
    BufferGpu a1{static_cast<size_t>(TAMANHO_PEDACO) * oculta1};
    BufferGpu a2{static_cast<size_t>(TAMANHO_PEDACO) * oculta2};
    BufferGpu escores{static_cast<size_t>(TAMANHO_PEDACO) * classes};
};

void avaliar(const DenseLayer& l1, const DenseLayer& l2, const DenseLayer& l3,
             const Dados& teste) {
    const int n = static_cast<int>(teste.size());
    const size_t bytes_imagens = teste.imagens.size() * sizeof(float);
    const size_t bytes_escores = teste.size() * classes * sizeof(float);

    // mede tudo: memória pinned, cópias e kernels
    const auto inicio = chrono::steady_clock::now();
    float* h_imagens = nullptr;
    float* h_escores = nullptr;
    verificar(cudaMallocHost(&h_imagens, bytes_imagens));
    verificar(cudaMallocHost(&h_escores, bytes_escores));
    copy(teste.imagens.begin(), teste.imagens.end(), h_imagens);

    cudaStream_t streams[NSTREAMS];
    for (int s = 0; s < NSTREAMS; ++s) {
        verificar(cudaStreamCreate(&streams[s]));
    }
    BuffersStream buffers[NSTREAMS];

    for (int base = 0; base < n; base += NSTREAMS * TAMANHO_PEDACO) {
        const int ativos =
            min(NSTREAMS, (n - base + TAMANHO_PEDACO - 1) / TAMANHO_PEDACO);
        // como na aula: primeiro as cópias do grupo, depois kernels, depois retornos
        for (int s = 0; s < ativos; ++s) {
            const int ini = base + s * TAMANHO_PEDACO;
            const int qtd = min(TAMANHO_PEDACO, n - ini);
            verificar(cudaMemcpyAsync(buffers[s].entrada.data(),
                                      h_imagens + static_cast<size_t>(ini) * entradas,
                                      static_cast<size_t>(qtd) * entradas * sizeof(float),
                                      cudaMemcpyHostToDevice, streams[s]));
        }
        for (int s = 0; s < ativos; ++s) {
            const int ini = base + s * TAMANHO_PEDACO;
            const int qtd = min(TAMANHO_PEDACO, n - ini);
            BuffersStream& b = buffers[s];
            l1.ForwardLote(b.entrada.data(), b.a1.data(), qtd, streams[s]);
            l2.ForwardLote(b.a1.data(), b.a2.data(), qtd, streams[s]);
            l3.ForwardLote(b.a2.data(), b.escores.data(), qtd, streams[s]);
        }
        for (int s = 0; s < ativos; ++s) {
            const int ini = base + s * TAMANHO_PEDACO;
            const int qtd = min(TAMANHO_PEDACO, n - ini);
            verificar(cudaMemcpyAsync(h_escores + static_cast<size_t>(ini) * classes,
                                      buffers[s].escores.data(),
                                      static_cast<size_t>(qtd) * classes * sizeof(float),
                                      cudaMemcpyDeviceToHost, streams[s]));
        }
    }
    for (int s = 0; s < NSTREAMS; ++s) {
        verificar(cudaStreamSynchronize(streams[s]));
    }
    const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

    relatar(h_escores, teste);
    cout << "Tempo de avaliacao " << setprecision(6) << tempo.count() << " s | "
         << NSTREAMS << " stream(s), pedacos de " << TAMANHO_PEDACO << " imagens\n";

    for (int s = 0; s < NSTREAMS; ++s) {
        cudaStreamDestroy(streams[s]);
    }
    cudaFreeHost(h_imagens);
    cudaFreeHost(h_escores);
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
