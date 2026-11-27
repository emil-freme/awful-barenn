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

void verificar(cudaError_t erro) {
    if (erro != cudaSuccess) {
        throw runtime_error(cudaGetErrorString(erro));
    }
}

class BufferGpu {
    float* ponteiro = nullptr;
    size_t bytes;

public:
    explicit BufferGpu(size_t elementos) : bytes(elementos * sizeof(float)) {
        if (bytes != 0) {
            verificar(cudaMalloc(&ponteiro, bytes));
        }
    }
    ~BufferGpu() {
        if (ponteiro) {
            cudaFree(ponteiro);
        }
    }
    BufferGpu(const BufferGpu&) = delete;
    BufferGpu& operator=(const BufferGpu&) = delete;

    float* data() const { return ponteiro; }

    void enviar(const float* origem) const {
        verificar(cudaMemcpy(ponteiro, origem, bytes, cudaMemcpyHostToDevice));
    }

    void receber(float* destino) const {
        verificar(cudaMemcpy(destino, ponteiro, bytes, cudaMemcpyDeviceToHost));
    }
};

constexpr int bloco = 128;
int grade(int n) { return (n + bloco - 1) / bloco; }

// pesos[j * n_saidas + i], igual à CPU

// uma thread por neurônio
__global__ void forward_gpu(const float* entrada, const float* pesos,
                            const float* vies, int n_entradas, int n_saidas,
                            bool final, float* z, float* saida) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_saidas) return;
    float soma = vies[i];
    for (int j = 0; j < n_entradas; ++j) {
        soma += entrada[j] * pesos[j * n_saidas + i];
    }
    z[i] = soma;
    saida[i] = final ? soma : (0.0f < soma ? soma : 0.0f);
}

__global__ void delta_gpu(const float* grad_saida, const float* z, bool final,
                          int n_saidas, float* delta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_saidas) return;
    delta[i] = final ? grad_saida[i]
                     : grad_saida[i] * (z[i] > 0.0f ? 1.0f : 0.0f);
}

// uma thread por entrada
__global__ void gradiente_entrada_gpu(const float* pesos, const float* delta,
                                      int n_entradas, int n_saidas,
                                      float* grad_entrada) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_entradas) return;
    float soma = 0.0f;
    for (int j = 0; j < n_saidas; ++j) {
        soma += delta[j] * pesos[i * n_saidas + j];
    }
    grad_entrada[i] = soma;
}

// uma thread por peso
__global__ void atualizar_gpu(const float* entrada, const float* delta,
                              int n_entradas, int n_saidas, float taxa,
                              float* pesos, float* vies) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_entradas * n_saidas) return;
    int j = k / n_saidas;
    int i = k % n_saidas;
    pesos[k] -= taxa * (delta[i] * entrada[j]);
    if (j == 0) vies[i] -= taxa * delta[i];
}

// porte direto: a cada chamada aloca na GPU, copia, executa e copia de volta
class DenseLayer {
    int in_features;
    int out_features;
    bool final;
    vector<float> weights;
    vector<float> bias;
    const float* last_input = nullptr;
    vector<float> last_z;
    vector<float> out;
    vector<float> grad_in;

public:
    DenseLayer(int input_size, int output_size, bool is_final = false)
        : in_features(input_size), out_features(output_size), final(is_final),
          weights(static_cast<size_t>(input_size) * output_size), bias(output_size, 0),
          last_z(output_size), out(output_size), grad_in(input_size) {
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        for (int i = 0; i < out_features; ++i) {
            for (int j = 0; j < in_features; ++j) {
                weights[j * out_features + i] = distribution(gen);
            }
        }
    }

    const float* Forward(const float* input) {
        last_input = input;
        BufferGpu d_entrada(in_features), d_pesos(weights.size()), d_vies(out_features);
        BufferGpu d_z(out_features), d_saida(out_features);
        d_entrada.enviar(input);
        d_pesos.enviar(weights.data());
        d_vies.enviar(bias.data());

        forward_gpu<<<grade(out_features), bloco>>>(
            d_entrada.data(), d_pesos.data(), d_vies.data(), in_features,
            out_features, final, d_z.data(), d_saida.data());
        verificar(cudaGetLastError());

        d_z.receber(last_z.data());
        d_saida.receber(out.data());
        return out.data();
    }

    const float* Backward(const float* grad_out, float learning_rate,
                          bool propagate = true) {
        BufferGpu d_entrada(in_features), d_pesos(weights.size()), d_vies(out_features);
        BufferGpu d_grad_saida(out_features), d_z(out_features), d_delta(out_features);
        BufferGpu d_grad_entrada(propagate ? in_features : 0);
        d_entrada.enviar(last_input);
        d_pesos.enviar(weights.data());
        d_vies.enviar(bias.data());
        d_grad_saida.enviar(grad_out);
        d_z.enviar(last_z.data());

        delta_gpu<<<grade(out_features), bloco>>>(
            d_grad_saida.data(), d_z.data(), final, out_features, d_delta.data());
        verificar(cudaGetLastError());

        // mesma stream: o gradiente usa os pesos antes da atualização
        if (propagate) {
            gradiente_entrada_gpu<<<grade(in_features), bloco>>>(
                d_pesos.data(), d_delta.data(), in_features, out_features,
                d_grad_entrada.data());
            verificar(cudaGetLastError());
        }
        atualizar_gpu<<<grade(in_features * out_features), bloco>>>(
            d_entrada.data(), d_delta.data(), in_features, out_features,
            learning_rate, d_pesos.data(), d_vies.data());
        verificar(cudaGetLastError());

        d_pesos.receber(weights.data());
        d_vies.receber(bias.data());
        if (propagate) {
            d_grad_entrada.receber(grad_in.data());
        }
        return propagate ? grad_in.data() : nullptr;
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
    for (size_t i = 0; i < teste.size(); ++i) {
        const float* s = l3.Forward(l2.Forward(l1.Forward(teste.imagem(i))));
        copy(s, s + classes, &escores[i * classes]);
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
        // Rede densa com dimensões 64 -> 32 -> 16 -> 10.
        Barenn::DenseLayer l1(Barenn::entradas, 32);
        Barenn::DenseLayer l2(32, 16);
        Barenn::DenseLayer l3(16, Barenn::classes, true);
        array<float, Barenn::classes> gradiente{};
        vector<double> perdas(epocas);

        // A medição inclui forward, perda e backward de todas as épocas.
        const auto inicio = chrono::steady_clock::now();
        for (int epoca = 0; epoca < epocas; ++epoca) {
            double perda_total = 0;
            // Uma época percorre todo o conjunto na ordem do arquivo.
            for (size_t i = 0; i < treino.size(); ++i) {
                const float* escores = l3.Forward(l2.Forward(l1.Forward(treino.imagem(i))));
                perda_total += Barenn::perda(escores, treino.rotulos[i], gradiente.data());

                // Propaga o gradiente da última camada até a primeira.
                // Cada Backward também atualiza os parâmetros da sua camada.
                const float* grad3 = l3.Backward(gradiente.data(), taxa);
                const float* grad2 = l2.Backward(grad3, taxa);
                l1.Backward(grad2, taxa, false);
            }
            perdas[epoca] = perda_total / treino.size();
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
