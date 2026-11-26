// Versão GPU (CUDA) do treino do Barenn.
//
// Porting inicial: o algoritmo é o mesmo SGD por amostra da versão CPU, com
// os mesmos dados, a mesma inicialização e a mesma ordem de operações de
// ponto flutuante em cada neurônio. Cada etapa de cada camada é um kernel,
// com uma thread por neurônio (ou por peso, na atualização), e as amostras
// são processadas em sequência, como na CPU.
//
// A leitura dos dados, a inicialização das camadas e a avaliação reaproveitam
// o código da CPU (barenn.hpp). Os pesos treinados são copiados de volta para
// as camadas da CPU, que fazem a avaliação.
//
// Compilar com --fmad=false: sem isso o nvcc funde a*b+c em FMA, o que muda o
// arredondamento e a saída deixa de ser idêntica à referência.

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "barenn.hpp"

using namespace std;

// Aborta com mensagem clara se uma chamada da API CUDA falhar.
#define CUDA_CHECK(chamada)                                                   \
    do {                                                                      \
        cudaError_t erro_cuda = (chamada);                                    \
        if (erro_cuda != cudaSuccess) {                                       \
            fprintf(stderr, "Erro CUDA em %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(erro_cuda));                           \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// Verifica erros de lançamento de kernel (configuração inválida etc.).
#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())

namespace {

constexpr int threads_por_bloco = 128;

int blocos_para(int n) { return (n + threads_por_bloco - 1) / threads_por_bloco; }

// ---------------------------------------------------------------------------
// Kernels. Layout dos pesos igual ao da CPU: w[j * saidas + i] liga a
// entrada j ao neurônio i.
// ---------------------------------------------------------------------------

// z = b + x·W e a = ReLU(z) (ou a = z na última camada). Uma thread por
// neurônio; a soma de cada neurônio segue a mesma ordem da CPU (viés
// primeiro, depois j = 0, 1, ...).
__global__ void k_forward(const float* x, const float* w, const float* b,
                          float* z, float* a, int entradas_c, int saidas_c,
                          bool final) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= saidas_c) return;
    float soma = b[i];
    for (int j = 0; j < entradas_c; ++j) {
        soma += x[j] * w[j * saidas_c + i];
    }
    z[i] = soma;
    a[i] = final ? soma : fmaxf(0.0f, soma);
}

// Perda (metade da soma dos erros quadráticos) e gradiente nos escores.
// Uma única thread: são só 10 classes, e a perda da época é acumulada em
// double na mesma ordem da CPU (amostra por amostra).
__global__ void k_perda(const float* escores, const int* rotulos, int amostra,
                        float* gradiente, double* perda_epoca) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int rotulo = rotulos[amostra];
    float total = 0;
    for (int i = 0; i < Barenn::classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        gradiente[i] = erro;
    }
    *perda_epoca += total;
}

// delta = gradiente da perda antes da ativação (derivada da ReLU).
__global__ void k_delta(const float* grad_saida, const float* z, float* delta,
                        int saidas_c, bool final) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= saidas_c) return;
    delta[i] = final ? grad_saida[i]
                     : grad_saida[i] * (z[i] > 0.0f ? 1.0f : 0.0f);
}

// Gradiente da entrada, com os pesos anteriores à atualização. Uma thread
// por entrada, somando os neurônios na mesma ordem da CPU.
__global__ void k_grad_entrada(const float* delta, const float* w,
                               float* grad_entrada, int entradas_c,
                               int saidas_c) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= entradas_c) return;
    float soma = 0;
    for (int i = 0; i < saidas_c; ++i) {
        soma += delta[i] * w[j * saidas_c + i];
    }
    grad_entrada[j] = soma;
}

// SGD: w -= taxa * delta_i * x_j e b -= taxa * delta_i. Uma thread por peso.
__global__ void k_atualiza(float* w, float* b, const float* x,
                           const float* delta, float taxa, int entradas_c,
                           int saidas_c) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= entradas_c * saidas_c) return;
    const int j = k / saidas_c;
    const int i = k % saidas_c;
    w[k] -= taxa * (delta[i] * x[j]);
    if (j == 0) {
        b[i] -= taxa * delta[i];
    }
}

// ---------------------------------------------------------------------------
// Camada na GPU: parâmetros e buffers alocados no device.
// ---------------------------------------------------------------------------
struct CamadaGPU {
    int entradas_c = 0;
    int saidas_c = 0;
    bool final = false;
    float* w = nullptr;      // pesos (entradas x saídas, transpostos)
    float* b = nullptr;      // vieses
    float* z = nullptr;      // soma antes da ativação (último forward)
    float* a = nullptr;      // saída após a ativação
    float* delta = nullptr;  // gradiente antes da ativação
    float* grad = nullptr;   // gradiente da entrada

    // Aloca no device e copia os parâmetros iniciais da camada da CPU.
    void carregar(Barenn::DenseLayer& cpu, bool ultima) {
        entradas_c = cpu.entradas_camada();
        saidas_c = cpu.saidas_camada();
        final = ultima;
        const size_t n_pesos = static_cast<size_t>(entradas_c) * saidas_c;
        CUDA_CHECK(cudaMalloc(&w, n_pesos * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&b, saidas_c * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&z, saidas_c * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&a, saidas_c * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&delta, saidas_c * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&grad, entradas_c * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(w, cpu.pesos().data(), n_pesos * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b, cpu.vieses().data(), saidas_c * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // Copia os parâmetros treinados de volta para a camada da CPU.
    void devolver(Barenn::DenseLayer& cpu) const {
        CUDA_CHECK(cudaMemcpy(cpu.pesos().data(), w,
                              cpu.pesos().size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(cpu.vieses().data(), b,
                              cpu.vieses().size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    void liberar() {
        for (float* p : {w, b, z, a, delta, grad}) {
            CUDA_CHECK(cudaFree(p));
        }
        w = b = z = a = delta = grad = nullptr;
    }

    void forward(const float* x) const {
        k_forward<<<blocos_para(saidas_c), threads_por_bloco>>>(
            x, w, b, z, a, entradas_c, saidas_c, final);
        CUDA_CHECK_KERNEL();
    }

    // Backward completo da camada. grad_saida é o gradiente vindo da camada
    // seguinte (ou da perda); x é a entrada usada no forward.
    void backward(const float* grad_saida, const float* x, float taxa,
                  bool precisa_grad_entrada) const {
        k_delta<<<blocos_para(saidas_c), threads_por_bloco>>>(
            grad_saida, z, delta, saidas_c, final);
        CUDA_CHECK_KERNEL();
        if (precisa_grad_entrada) {
            k_grad_entrada<<<blocos_para(entradas_c), threads_por_bloco>>>(
                delta, w, grad, entradas_c, saidas_c);
            CUDA_CHECK_KERNEL();
        }
        k_atualiza<<<blocos_para(entradas_c * saidas_c), threads_por_bloco>>>(
            w, b, x, delta, taxa, entradas_c, saidas_c);
        CUDA_CHECK_KERNEL();
    }
};

} // namespace

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return 1;
    }
    const string arquivo_treino = argc == 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc == 3 ? argv[2] : "optdigits.tes";

    try {
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
        // Mesma rede e mesma inicialização da versão CPU: 64 -> 32 -> 16 -> 10.
        Barenn::DenseLayer l1(Barenn::entradas, 32);
        Barenn::DenseLayer l2(32, 16);
        Barenn::DenseLayer l3(16, Barenn::classes, true);
        array<double, epocas> perdas{};

        // Inicializa o contexto CUDA fora do cronômetro (custo fixo, ~centenas
        // de ms, que não depende do treino).
        CUDA_CHECK(cudaFree(nullptr));

        // A medição inclui as cópias Host->Device dos dados e dos pesos, todos
        // os kernels do treino e a cópia Device->Host dos pesos treinados.
        const auto inicio = chrono::steady_clock::now();

        // Conjunto de treino achatado: imagens contíguas (n x 64) e rótulos.
        const int n = static_cast<int>(treino.size());
        vector<float> imagens(static_cast<size_t>(n) * Barenn::entradas);
        vector<int> rotulos(n);
        for (int s = 0; s < n; ++s) {
            copy(treino[s].imagem.begin(), treino[s].imagem.end(),
                 imagens.begin() + static_cast<size_t>(s) * Barenn::entradas);
            rotulos[s] = treino[s].rotulo;
        }
        float* d_imagens = nullptr;
        int* d_rotulos = nullptr;
        float* d_gradiente = nullptr;
        double* d_perdas = nullptr;
        CUDA_CHECK(cudaMalloc(&d_imagens, imagens.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rotulos, rotulos.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_gradiente, Barenn::classes * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_perdas, epocas * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_imagens, imagens.data(),
                              imagens.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rotulos, rotulos.data(),
                              rotulos.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_perdas, 0, epocas * sizeof(double)));

        CamadaGPU g1, g2, g3;
        g1.carregar(l1, false);
        g2.carregar(l2, false);
        g3.carregar(l3, true);

        for (int epoca = 0; epoca < epocas; ++epoca) {
            for (int s = 0; s < n; ++s) {
                const float* x = d_imagens + static_cast<size_t>(s) * Barenn::entradas;
                g1.forward(x);
                g2.forward(g1.a);
                g3.forward(g2.a);
                k_perda<<<1, 1>>>(g3.a, d_rotulos, s, d_gradiente,
                                  d_perdas + epoca);
                CUDA_CHECK_KERNEL();
                g3.backward(d_gradiente, g2.a, taxa, true);
                g2.backward(g3.grad, g1.a, taxa, true);
                // A entrada da rede não tem parâmetros: dispensa o grad de g1.
                g1.backward(g2.grad, x, taxa, false);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(perdas.data(), d_perdas, epocas * sizeof(double),
                              cudaMemcpyDeviceToHost));
        g1.devolver(l1);
        g2.devolver(l2);
        g3.devolver(l3);
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        g1.liberar();
        g2.liberar();
        g3.liberar();
        CUDA_CHECK(cudaFree(d_imagens));
        CUDA_CHECK(cudaFree(d_rotulos));
        CUDA_CHECK(cudaFree(d_gradiente));
        CUDA_CHECK(cudaFree(d_perdas));

        for (double& p : perdas) {
            p /= n;
        }

        // Mesma saída da versão CPU.
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
