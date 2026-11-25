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

class DenseLayer {
    int in_features;
    int out_features;
    bool final;
    // Matriz de pesos contígua e transposta (column-major em relação ao
    // legado): weights[j * out_features + i] liga a entrada j ao neurônio de
    // saída i. Assim os laços internos do Forward e da atualização percorrem
    // neurônios de saída consecutivos, sem dependência entre iterações, e
    // podem ser vetorizados (SIMD) sem reordenar somas de ponto flutuante.
    vector<float> weights;
    vector<float> bias;
    // Dados do último forward usados para calcular os gradientes.
    vector<float> last_input;
    vector<float> last_z;
    // Buffers pré-alocados e reaproveitados a cada amostra, evitando
    // malloc/free e cópias de vector no laço de treino.
    vector<float> out;
    vector<float> delta;
    vector<float> grad_in;

public:
    DenseLayer(int input_size, int output_size, bool is_final = false)
        : in_features(input_size), out_features(output_size), final(is_final),
          weights(static_cast<size_t>(output_size) * input_size),
          bias(output_size, 0), last_input(input_size), last_z(output_size),
          out(output_size), delta(output_size), grad_in(input_size) {
        // A semente fixa torna a inicialização de cada camada reproduzível.
        // A ordem de preenchimento (linha a linha) é a mesma do código legado.
        // Os pesos são sorteados na mesma ordem do legado (neurônio i, entrada
        // j) e gravados na posição transposta.
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        for (int i = 0; i < out_features; ++i) {
            for (int j = 0; j < in_features; ++j) {
                weights[static_cast<size_t>(j) * out_features + i] = distribution(gen);
            }
        }
    }

    // Devolve uma referência ao buffer interno de saída, válida até a
    // próxima chamada de Forward desta camada.
    const vector<float>& Forward(const vector<float>& input) {
        copy(input.begin(), input.end(), last_input.begin());
        const float* x = input.data();
        float* z = last_z.data();
        // Cada neurônio soma o viés aos produtos entre entradas e pesos.
        // A soma de cada neurônio i segue a mesma ordem do legado (j = 0, 1,
        // ...), mas os neurônios são acumulados juntos: o laço interno sobre i
        // é independente e vetorizável.
        for (int i = 0; i < out_features; ++i) {
            z[i] = bias[i];
        }
        for (int j = 0; j < in_features; ++j) {
            const float xj = x[j];
            const float* col = &weights[static_cast<size_t>(j) * out_features];
            for (int i = 0; i < out_features; ++i) {
                z[i] += xj * col[i];
            }
        }
        // As camadas ocultas usam ReLU. A última devolve escores lineares.
        for (int i = 0; i < out_features; ++i) {
            out[i] = final ? z[i] : max(0.0f, z[i]);
        }
        return out;
    }

    // need_grad_in = false pula o cálculo do gradiente da entrada. Isso vale
    // para a primeira camada, cujo gradiente de entrada nunca é usado.
    // Devolve uma referência ao buffer interno grad_in.
    const vector<float>& Backward(const vector<float>& grad_out,
                                  float learning_rate,
                                  bool need_grad_in = true) {
        // Delta é o gradiente da perda em relação à soma antes da ativação.
        // Na ReLU, o gradiente passa apenas quando essa soma é positiva.
        for (int i = 0; i < out_features; ++i) {
            delta[i] = final ? grad_out[i]
                             : grad_out[i] * (last_z[i] > 0.0f ? 1.0f : 0.0f);
        }

        // O gradiente da entrada usa os pesos anteriores à atualização.
        if (need_grad_in) {
            for (int i = 0; i < in_features; ++i) {
                float sum = 0;
                const float* col = &weights[static_cast<size_t>(i) * out_features];
                for (int j = 0; j < out_features; ++j) {
                    sum += delta[j] * col[j];
                }
                grad_in[i] = sum;
            }
        }

        // SGD por amostra. O gradiente do peso é delta vezes a entrada.
        // O gradiente do viés é o próprio delta.
        const float* x = last_input.data();
        const float* d = delta.data();
        for (int j = 0; j < in_features; ++j) {
            float* col = &weights[static_cast<size_t>(j) * out_features];
            const float xj = x[j];
            for (int i = 0; i < out_features; ++i) {
                col[i] -= learning_rate * (d[i] * xj);
            }
        }
        for (int i = 0; i < out_features; ++i) {
            bias[i] -= learning_rate * d[i];
        }
        return grad_in;
    }
};

struct Amostra {
    vector<float> imagem;
    int rotulo;
};

// Valida e converte uma linha "a0,a1,...,a63,rotulo". Devolve false se a
// linha estiver fora do formato esperado.
bool ler_linha(const string& linha, Amostra& amostra) {
    istringstream campos(linha);
    amostra.imagem.clear();
    amostra.imagem.reserve(entradas);
    // Cada atributo deve ser seguido por vírgula, inclusive o último.
    for (int i = 0; i < entradas; ++i) {
        float valor;
        char separador;
        if (!(campos >> valor >> separador) || separador != ',' ||
            !isfinite(valor) || valor < 0 || valor > 16) {
            break;
        }
        // Converte os valores de [0, 16] para [0, 1].
        amostra.imagem.push_back(valor / 16.0f);
    }
    // Depois dos atributos, deve restar apenas o rótulo de 0 a 9.
    return amostra.imagem.size() == entradas &&
           (campos >> amostra.rotulo) &&
           amostra.rotulo >= 0 && amostra.rotulo < classes &&
           (campos >> ws).eof();
}

vector<Amostra> ler_dados(const string& caminho) {
    ifstream arquivo(caminho);
    if (!arquivo) {
        throw runtime_error("Não foi possível abrir " + caminho);
    }

    // Etapa sequencial (barata): separa o arquivo em linhas.
    vector<string> linhas;
    string linha;
    while (getline(arquivo, linha)) {
        linhas.push_back(move(linha));
    }
    if (arquivo.bad() || linhas.empty()) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }

    // Etapa paralela: cada linha é convertida de forma independente e
    // gravada na sua própria posição, preservando a ordem do arquivo.
    // Exceções não podem sair de uma região OpenMP, então cada thread só
    // registra a menor linha inválida encontrada.
    const long n = static_cast<long>(linhas.size());
    vector<Amostra> dados(n);
    long primeira_invalida = n;
    #pragma omp parallel for schedule(static) reduction(min : primeira_invalida)
    for (long i = 0; i < n; ++i) {
        if (!ler_linha(linhas[i], dados[i]) && i < primeira_invalida) {
            primeira_invalida = i;
        }
    }
    if (primeira_invalida < n) {
        throw runtime_error(caminho + ":" + to_string(primeira_invalida + 1) +
            " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
    }
    return dados;
}

// Calcula metade da soma dos erros quadráticos. O alvo vale 1 para a
// classe correta e 0 para as demais. O erro também é a derivada por escore.
float perda(const vector<float>& escores, int rotulo,
            vector<float>* gradiente = nullptr) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        if (gradiente) {
            (*gradiente)[i] = erro;
        }
    }
    return total;
}

void avaliar(DenseLayer& l1, DenseLayer& l2, DenseLayer& l3,
             const vector<Amostra>& teste) {
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
        const auto& amostra = teste[i];
        const auto& escores = l3.Forward(l2.Forward(l1.Forward(amostra.imagem)));
        // O índice do maior escore indica a classe prevista.
        int previsto = static_cast<int>(
            max_element(escores.begin(), escores.end()) - escores.begin());
        bool acertou = previsto == amostra.rotulo;
        acertos += acertou;
        ++confusoes[amostra.rotulo][previsto];
        perda_total += perda(escores, amostra.rotulo);

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
            exemplos.push_back({i + 1, amostra.rotulo, previsto, segundo,
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
        // A leitura e a conversão dos arquivos também são medidas, pois
        // são a parte do programa que pode ser paralelizada sem alterar o
        // algoritmo de treino.
        const auto inicio_total = chrono::steady_clock::now();
        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        const chrono::duration<double> tempo_leitura =
            chrono::steady_clock::now() - inicio_total;
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }

        constexpr int epocas = 51;
        constexpr float taxa = 0.02f;
        // Rede densa com dimensões 64 -> 32 -> 16 -> 10.
        Barenn::DenseLayer l1(Barenn::entradas, 32);
        Barenn::DenseLayer l2(32, 16);
        Barenn::DenseLayer l3(16, Barenn::classes, true);
        vector<float> gradiente(Barenn::classes);
        array<double, epocas> perdas{};

        // A medição inclui forward, perda e backward de todas as épocas.
        const auto inicio = chrono::steady_clock::now();
        for (int epoca = 0; epoca < epocas; ++epoca) {
            double perda_total = 0;
            // Uma época percorre todo o conjunto na ordem do arquivo.
            for (const auto& amostra : treino) {
                const auto& escores =
                    l3.Forward(l2.Forward(l1.Forward(amostra.imagem)));
                perda_total += Barenn::perda(escores, amostra.rotulo, &gradiente);

                // Propaga o gradiente da última camada até a primeira.
                // Cada Backward também atualiza os parâmetros da sua camada.
                const auto& grad3 = l3.Backward(gradiente, taxa);
                const auto& grad2 = l2.Backward(grad3, taxa);
                // A entrada da rede não tem parâmetros: dispensa o grad_in de l1.
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
        cout << "Tempo de leitura " << tempo_leitura.count() << " s\n";
        Barenn::avaliar(l1, l2, l3, teste);
        const chrono::duration<double> tempo_total =
            chrono::steady_clock::now() - inicio_total;
        cout << "Tempo total " << tempo_total.count() << " s\n";
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
