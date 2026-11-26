# Build reprodutível do Barenn. As flags ficam fixas aqui para que toda
# medição (bench, profiling) use exatamente o mesmo binário.
# -O3: habilita o vetorizador de laços.
# -march=x86-64-v3: AVX2/FMA, suportado pelo notebook (Zen 3) e pelos nós do
#   Franky (Raptor Lake). Não usamos -march=native porque no Franky o build
#   é feito no nó de login (Kaby Lake), diferente dos nós de computação.
# -ffp-contract=off: impede o compilador de fundir a*b+c em FMA, o que muda o
#   arredondamento; mantém a saída bit a bit idêntica à referência.
CXX      ?= g++
# -fopenmp: paraleliza a conversao dos arquivos de entrada (ler_dados).
CXXFLAGS ?= -O3 -march=x86-64-v3 -ffp-contract=off -fopenmp -g -std=c++17 -Wall -Wextra
TARGET   := barenn
SRC      := barenn.cpp
PROF_DIR := profiling

# Versão GPU. -arch=sm_86: Ampere, comum à RTX 3050 (notebook) e à RTX A1000
# (Franky). --fmad=false: sem FMA, mesma razão do -ffp-contract=off acima.
NVCC      ?= nvcc
NVCCFLAGS ?= -O3 -arch=sm_86 --fmad=false -lineinfo -std=c++17 \
             -Xcompiler -fopenmp,-O3
CUDA_TARGET := barenn_cuda

.PHONY: all run bench validate callgrind cachegrind clean \
        cuda bench-cuda validate-cuda nsys

all: $(TARGET)

$(TARGET): $(SRC) barenn.hpp
	$(CXX) $(CXXFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

# Média e desvio padrão do tempo de treino (N=10 por padrão).
bench: $(TARGET)
	./bench.sh ./$(TARGET) $(or $(N),10)

# Compara a saída com a referência do código legado.
validate: $(TARGET)
	./validate.sh ./$(TARGET)

callgrind: $(TARGET)
	mkdir -p $(PROF_DIR)
	valgrind --tool=callgrind --callgrind-out-file=$(PROF_DIR)/callgrind.out ./$(TARGET) > /dev/null
	callgrind_annotate $(PROF_DIR)/callgrind.out | head -40

cachegrind: $(TARGET)
	mkdir -p $(PROF_DIR)
	valgrind --tool=cachegrind --cache-sim=yes --cachegrind-out-file=$(PROF_DIR)/cachegrind.out ./$(TARGET) > /dev/null

cuda: $(CUDA_TARGET)

$(CUDA_TARGET): barenn_cuda.cu barenn.hpp
	$(NVCC) $(NVCCFLAGS) $< -o $@

bench-cuda: $(CUDA_TARGET)
	./bench.sh ./$(CUDA_TARGET) $(or $(N),10)

validate-cuda: $(CUDA_TARGET)
	./validate.sh ./$(CUDA_TARGET)

# Linha do tempo de kernels e transferências com Nsight Systems.
nsys: $(CUDA_TARGET)
	mkdir -p $(PROF_DIR)
	nsys profile --force-overwrite=true --trace=cuda -o $(PROF_DIR)/barenn_cuda ./$(CUDA_TARGET) > /dev/null
	# O nsys 2022.4 do pacote do Ubuntu não acha o próprio importador e deixa
	# só o .qdstrm: nesse caso a conversão é feita chamando-o diretamente.
	[ -f $(PROF_DIR)/barenn_cuda.nsys-rep ] || \
	    /usr/lib/nsight-systems/host-linux-x64/QdstrmImporter --input-file $(PROF_DIR)/barenn_cuda.qdstrm > /dev/null
	# Os nomes dos relatórios mudaram entre versões do nsys (2022: gpukernsum;
	# 2024+: cuda_gpu_kern_sum).
	if nsys stats --help-reports 2>/dev/null | grep -q cuda_gpu_kern_sum; then \
	    R=cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum; \
	else R=gpukernsum,gpumemtimesum,cudaapisum; fi; \
	nsys stats --force-export=true --report $$R $(PROF_DIR)/barenn_cuda.nsys-rep

clean:
	rm -f $(TARGET) $(CUDA_TARGET)
	rm -rf $(PROF_DIR)
