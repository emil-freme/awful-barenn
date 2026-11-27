#include <omp.h>

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
constexpr int oculta1 = 32;
constexpr int oculta2 = 16;

class DenseLayer {
    int in_features;
    int out_features;
    bool final;
    // pesos contíguos e transpostos: weights[j * out_features + i] (entrada j, neurônio i)
    vector<float> weights;
    vector<float> bias;
    // Dados do último forward usados para calcular os gradientes.
    const float* last_input = nullptr;
    vector<float> last_z;
    // buffers reaproveitados entre amostras
    vector<float> out;
    vector<float> delta;
    vector<float> grad_in;

public:
    DenseLayer(int input_size, int output_size, bool is_final = false)
        : in_features(input_size), out_features(output_size), final(is_final),
          weights(static_cast<size_t>(input_size) * output_size), bias(output_size, 0),
          last_z(output_size), out(output_size), delta(output_size),
          grad_in(input_size) {
        // A semente fixa torna a inicialização de cada camada reproduzível.
        // mesma ordem de sorteio do legado
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
        float* z = last_z.data();
        for (int i = 0; i < out_features; ++i) {
            z[i] = bias[i];
        }
        // mesma ordem de soma do legado (j = 0, 1, ...)
        for (int j = 0; j < in_features; ++j) {
            const float x = input[j];
            const float* w = &weights[j * out_features];
            for (int i = 0; i < out_features; ++i) {
                z[i] += x * w[i];
            }
        }
        // As camadas ocultas usam ReLU. A última devolve escores lineares.
        for (int i = 0; i < out_features; ++i) {
            out[i] = final ? z[i] : max(0.0f, z[i]);
        }
        return out.data();
    }

    // Forward sem efeito colateral: não escreve nos buffers da camada, então
    // várias threads podem chamar ao mesmo tempo. Mesma conta do Forward.
    void Inferir(const float* input, float* saida) const {
        for (int i = 0; i < out_features; ++i) {
            saida[i] = bias[i];
        }
        for (int j = 0; j < in_features; ++j) {
            const float x = input[j];
            const float* w = &weights[j * out_features];
            for (int i = 0; i < out_features; ++i) {
                saida[i] += x * w[i];
            }
        }
        if (!final) {
            for (int i = 0; i < out_features; ++i) {
                saida[i] = max(0.0f, saida[i]);
            }
        }
    }

    // a primeira camada não precisa do gradiente da entrada
    const float* Backward(const float* grad_out, float learning_rate,
                          bool propagate = true) {
        // Delta é o gradiente da perda em relação à soma antes da ativação.
        // Na ReLU, o gradiente passa apenas quando essa soma é positiva.
        for (int i = 0; i < out_features; ++i) {
            delta[i] = final ? grad_out[i]
                             : grad_out[i] * (last_z[i] > 0.0f ? 1.0f : 0.0f);
        }

        // O gradiente da entrada usa os pesos anteriores à atualização.
        if (propagate) {
            for (int i = 0; i < in_features; ++i) {
                const float* w = &weights[i * out_features];
                float sum = 0;
                for (int j = 0; j < out_features; ++j) {
                    sum += delta[j] * w[j];
                }
                grad_in[i] = sum;
            }
        }

        // SGD por amostra. O gradiente do peso é delta vezes a entrada.
        // O gradiente do viés é o próprio delta.
        for (int j = 0; j < in_features; ++j) {
            const float x = last_input[j];
            float* w = &weights[j * out_features];
            for (int i = 0; i < out_features; ++i) {
                w[i] -= learning_rate * (delta[i] * x);
            }
        }
        for (int i = 0; i < out_features; ++i) {
            bias[i] -= learning_rate * delta[i];
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

    // A leitura do disco é sequencial; o parsing de cada linha é independente.
    vector<string> linhas;
    string linha;
    while (getline(arquivo, linha)) {
        linhas.push_back(move(linha));
    }
    const long n = static_cast<long>(linhas.size());

    Dados dados;
    dados.imagens.resize(static_cast<size_t>(n) * entradas);
    dados.rotulos.resize(n);
    vector<char> invalida(n, 0);

    #pragma omp parallel for schedule(runtime)
    for (long i = 0; i < n; ++i) {
        istringstream campos(linhas[i]);
        float* imagem = &dados.imagens[i * entradas];
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
            invalida[i] = 1;  // cada thread só escreve na sua posição
        }
        dados.rotulos[i] = rotulo;
    }

    // O erro reportado é o da primeira linha inválida, como no legado.
    for (long i = 0; i < n; ++i) {
        if (invalida[i]) {
            throw runtime_error(caminho + ":" + to_string(i + 1) +
                " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
        }
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

void avaliar(const DenseLayer& l1, const DenseLayer& l2, const DenseLayer& l3,
             const Dados& teste) {
    // Parte paralela: o forward de cada imagem é independente. Cada thread usa
    // ativações locais e escreve só nos escores da própria imagem.
    const long n = static_cast<long>(teste.size());
    vector<float> escores_todos(static_cast<size_t>(n) * classes);
    const double inicio = omp_get_wtime();
    #pragma omp parallel for schedule(runtime)
    for (long i = 0; i < n; ++i) {
        array<float, oculta1> a1;
        array<float, oculta2> a2;
        l1.Inferir(teste.imagem(i), a1.data());
        l2.Inferir(a1.data(), a2.data());
        l3.Inferir(a2.data(), &escores_todos[i * classes]);
    }
    const double tempo = omp_get_wtime() - inicio;

    struct Exemplo {
        size_t linha;
        int real;
        int previsto;
        int segundo;
        float maior_escore;
        float segundo_escore;
    };

    // Linhas representam a classe esperada e colunas, a classe prevista.
    array<array<size_t, classes>, classes> confusoes{};
    vector<Exemplo> exemplos;
    size_t acertos = 0;
    size_t exemplos_acerto = 0;
    size_t exemplos_erro = 0;
    double perda_total = 0;

    // Só fazemos forward aqui. Os pesos ficam fixos.
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
        // A acurácia usa todas as imagens, inclusive as não exibidas.
        size_t& exemplos_tipo = acertou ? exemplos_acerto : exemplos_erro;
        if (exemplos_tipo < 3) {
            int segundo = previsto == 0 ? 1 : 0;
            for (int classe = 0; classe < classes; ++classe) {
                if (classe != previsto && escores[classe] > escores[segundo]) {
                    segundo = classe;
                }
            }
            exemplos.push_back({i + 1, rotulo, previsto, segundo,
                                escores[previsto], escores[segundo]});
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
    cout << "Tempo de avaliacao " << setprecision(6) << tempo << " s | "
         << omp_get_max_threads() << " thread(s)\n";
}

} // namespace Barenn

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return 1;
    }
    const string arquivo_treino = argc == 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc == 3 ? argv[2] : "optdigits.tes";

    try {
        const double inicio_programa = omp_get_wtime();
        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        const double tempo_leitura = omp_get_wtime() - inicio_programa;
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }

        constexpr int epocas = 51;
        constexpr float taxa = 0.02f;
        // Rede densa com dimensões 64 -> 32 -> 16 -> 10.
        Barenn::DenseLayer l1(Barenn::entradas, Barenn::oculta1);
        Barenn::DenseLayer l2(Barenn::oculta1, Barenn::oculta2);
        Barenn::DenseLayer l3(Barenn::oculta2, Barenn::classes, true);
        array<float, Barenn::classes> gradiente{};
        array<double, epocas> perdas{};

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
        cout << "Tempo de leitura " << setprecision(6) << tempo_leitura << " s\n"
             << setprecision(4);
        Barenn::avaliar(l1, l2, l3, teste);
        cout << "Tempo de execucao " << setprecision(6)
             << omp_get_wtime() - inicio_programa << " s\n";
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
