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

.PHONY: all run bench validate callgrind cachegrind clean

all: $(TARGET)

$(TARGET): $(SRC)
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

clean:
	rm -f $(TARGET)
	rm -rf $(PROF_DIR)
