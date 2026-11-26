#include "barenn.hpp"
#include <cuda_runtime.h>

namespace GPU {

void verificar(cudaError_t erro, const char* operacao) {
    if (erro != cudaSuccess)
        throw runtime_error(string(operacao) + ": " + cudaGetErrorString(erro));
}

template<class T> struct Buffer {
    T* ptr = nullptr;
    explicit Buffer(size_t n) {
        verificar(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)), "cudaMalloc");
    }
    ~Buffer() {
        if (ptr && cudaFree(ptr) != cudaSuccess)
            cerr << "Falha ao liberar memoria CUDA\n";
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

struct Evento {
    cudaEvent_t evento{};
    Evento() { verificar(cudaEventCreate(&evento), "cudaEventCreate"); }
    ~Evento() {
        if (cudaEventDestroy(evento) != cudaSuccess)
            cerr << "Falha ao liberar evento CUDA\n";
    }
    Evento(const Evento&) = delete;
    Evento& operator=(const Evento&) = delete;
};

template<int In, int Out, bool Final, int Stride = In>
__device__ void forward(const float* pesos, const float* bias,
                        const float* entrada, float* z, float* saida) {
    int i = threadIdx.x;
    if (i < Out) {
        float soma = bias[i];
        for (int j = 0; j < In; ++j)
            soma += entrada[j] * pesos[i * Stride + j];
        z[i] = soma;
        saida[i] = Final ? soma : fmaxf(0.0f, soma);
    }
    __syncthreads();
}

template<int In, int Out, bool Final, int Stride = In>
__device__ void backward(float* pesos, float* bias, const float* entrada,
                         const float* z, const float* grad_out,
                         float* delta, float* grad_in, float taxa) {
    int i = threadIdx.x;
    if (i < Out)
        delta[i] = Final ? grad_out[i]
            : grad_out[i] * (z[i] > 0.0f ? 1.0f : 0.0f);
    __syncthreads();
    if (i < In) {
        float soma = 0;
        for (int j = 0; j < Out; ++j)
            soma += delta[j] * pesos[j * Stride + i];
        grad_in[i] = soma;
    }
    // Todos terminam de ler os pesos antigos antes das atualizacoes.
    __syncthreads();
    if (i < Out) {
        for (int j = 0; j < In; ++j)
            pesos[i * Stride + j] -= taxa * (delta[i] * entrada[j]);
        bias[i] -= taxa * delta[i];
    }
    __syncthreads();
}

// Converte o indice compacto para linhas com uma coluna extra.
__device__ int indice_compartilhado(int i) {
    if (i < 2048) return i + i / 64;
    if (i < 2080) return i + 32;
    if (i < 2592) {
        int k = i - 2080;
        return 2112 + k + k / 32;
    }
    if (i < 2608) return i + 48;
    if (i < 2768) {
        int k = i - 2608;
        return 2656 + k + k / 16;
    }
    return i + 58;
}

__global__ void treinar_epoca(float* parametros_globais, const float* imagens,
                                const int* rotulos, int quantidade,
                                double* perdas, int epoca, float taxa) {
    __shared__ float parametros[2836];
    __shared__ float entrada[64], a1[32], a2[16], escores[10];
    __shared__ float z1[32], z2[16], z3[10], gradiente[10];
    __shared__ float d1[32], d2[16], d3[10];
    __shared__ float g1[64], g2[32], g3[16];
    for (int i = threadIdx.x; i < 2778; i += blockDim.x)
        parametros[indice_compartilhado(i)] = parametros_globais[i];
    __syncthreads();

    for (int amostra = 0; amostra < quantidade; ++amostra) {
        entrada[threadIdx.x] = imagens[amostra * 64 + threadIdx.x];
        __syncthreads();
        forward<64, 32, false, 65>(parametros, parametros + 2080, entrada, z1, a1);
        forward<32, 16, false, 33>(parametros + 2112, parametros + 2640, a1, z2, a2);
        forward<16, 10, true, 17>(parametros + 2656, parametros + 2826, a2, z3, escores);
        if (threadIdx.x == 0) {
            float total = 0;
            for (int i = 0; i < 10; ++i) {
                float erro = escores[i] - (i == rotulos[amostra] ? 1.0f : 0.0f);
                total += 0.5f * erro * erro;
                gradiente[i] = erro;
            }
            perdas[epoca] += static_cast<double>(total);
        }
        __syncthreads();
        backward<16, 10, true, 17>(parametros + 2656, parametros + 2826,
            a2, z3, gradiente, d3, g3, taxa);
        backward<32, 16, false, 33>(parametros + 2112, parametros + 2640,
            a1, z2, g3, d2, g2, taxa);
        backward<64, 32, false, 65>(parametros, parametros + 2080,
            entrada, z1, g2, d1, g1, taxa);
    }

    for (int i = threadIdx.x; i < 2778; i += blockDim.x)
        parametros_globais[i] = parametros[indice_compartilhado(i)];
}

struct Tempos { float treino_ms, h2d_ms, d2h_ms; };

Tempos treinar(Barenn::DenseLayer& l1, Barenn::DenseLayer& l2,
              Barenn::DenseLayer& l3, const vector<Barenn::Amostra>& treino,
              double* perdas, int epocas, float taxa) {
    vector<float> imagens;
    vector<int> rotulos;
    imagens.reserve(treino.size() * 64);
    rotulos.reserve(treino.size());
    for (const auto& amostra : treino) {
        imagens.insert(imagens.end(), amostra.imagem.begin(), amostra.imagem.end());
        rotulos.push_back(amostra.rotulo);
    }
    vector<float> parametros;
    for (const auto* camada : {&l1, &l2, &l3}) {
        auto p = camada->parametros();
        parametros.insert(parametros.end(), p.begin(), p.end());
    }
    Buffer<float> dp(parametros.size()), dx(imagens.size());
    Buffer<int> dy(rotulos.size());
    Buffer<double> dl(epocas);
    Evento inicio, fim;
    Tempos tempos{};
    verificar(cudaEventRecord(inicio.evento), "inicio H2D");
    verificar(cudaMemcpy(dp.ptr, parametros.data(), parametros.size() * sizeof(float), cudaMemcpyHostToDevice), "parametros H2D");
    verificar(cudaMemcpy(dx.ptr, imagens.data(), imagens.size() * sizeof(float), cudaMemcpyHostToDevice), "imagens H2D");
    verificar(cudaMemcpy(dy.ptr, rotulos.data(), rotulos.size() * sizeof(int), cudaMemcpyHostToDevice), "rotulos H2D");
    verificar(cudaEventRecord(fim.evento), "fim H2D");
    verificar(cudaEventSynchronize(fim.evento), "sincronizar H2D");
    verificar(cudaEventElapsedTime(&tempos.h2d_ms, inicio.evento, fim.evento), "tempo H2D");
    verificar(cudaMemset(dl.ptr, 0, epocas * sizeof(double)), "zerar perdas");
    verificar(cudaEventRecord(inicio.evento), "inicio treino");
    for (int epoca = 0; epoca < epocas; ++epoca) {
        treinar_epoca<<<1, 64>>>(dp.ptr, dx.ptr, dy.ptr,
            static_cast<int>(treino.size()), dl.ptr, epoca, taxa);
        verificar(cudaGetLastError(), "launch treinar_epoca");
    }
    verificar(cudaEventRecord(fim.evento), "fim treino");
    verificar(cudaEventSynchronize(fim.evento), "sincronizar treino");
    verificar(cudaEventElapsedTime(&tempos.treino_ms, inicio.evento, fim.evento), "tempo treino");
    verificar(cudaEventRecord(inicio.evento), "inicio D2H");
    verificar(cudaMemcpy(parametros.data(), dp.ptr, parametros.size() * sizeof(float), cudaMemcpyDeviceToHost), "parametros D2H");
    verificar(cudaMemcpy(perdas, dl.ptr, epocas * sizeof(double), cudaMemcpyDeviceToHost), "perdas D2H");
    verificar(cudaEventRecord(fim.evento), "fim D2H");
    verificar(cudaEventSynchronize(fim.evento), "sincronizar D2H");
    verificar(cudaEventElapsedTime(&tempos.d2h_ms, inicio.evento, fim.evento), "tempo D2H");
    for (int e = 0; e < epocas; ++e) perdas[e] /= treino.size();
    l1.carregar_parametros(parametros.data());
    l2.carregar_parametros(parametros.data() + 2080);
    l3.carregar_parametros(parametros.data() + 2608);
    return tempos;
}
} // namespace GPU

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return 1;
    }
    const string arquivo_treino = argc == 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc == 3 ? argv[2] : "optdigits.tes";

    try {
        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }

        constexpr int epocas = 51;
        constexpr float taxa = 0.02f;
        // Rede densa com dimensões 64 -> 32 -> 16 -> 10.
        Barenn::DenseLayer l1(Barenn::entradas, 32);
        Barenn::DenseLayer l2(32, 16);
        Barenn::DenseLayer l3(16, Barenn::classes, true);
        array<double, epocas> perdas{};

        const auto inicio = chrono::steady_clock::now();
        const auto gpu = GPU::treinar(l1, l2, l3, treino, perdas.data(), epocas, taxa);
        const chrono::duration<double> total = chrono::steady_clock::now() - inicio;
        cout << fixed << setprecision(6)
             << "CUDA H2D " << gpu.h2d_ms / 1000.0 << " s\n"
             << "CUDA D2H " << gpu.d2h_ms / 1000.0 << " s\n"
             << "CUDA total " << total.count() << " s\n";

        // Imprime as métricas e roda a avaliação após encerrar o cronômetro.
        cout << fixed << setprecision(4)
             << "TREINO | " << treino.size() << " imagens\n";
        for (int epoca = 0; epoca < epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << gpu.treino_ms / 1000.0 << " s\n";
        Barenn::avaliar(l1, l2, l3, teste);
#ifdef BARENN_VALIDAR
        for (const auto* camada : {&l1, &l2, &l3})
            for (float p : camada->parametros()) cerr << hexfloat << p << '\n';
#endif
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
