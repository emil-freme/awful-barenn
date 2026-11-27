# -ffp-contract=off: sem FMA, para a saída bater bit a bit com a do legado
# -mavx2: vetores de 8 floats (no Franky o tempo de treino cai 20%, ver issue #3)
CXX      = g++
OPT      ?= -O3
CXXFLAGS  = -std=c++17 $(OPT) -mavx2 -ffp-contract=off -Wall -Wextra

# rpath: nos nós do Franky o LD_LIBRARY_PATH às vezes perde a libstdc++ do g++ 14
LIBDIR   := $(dir $(shell g++ -print-file-name=libstdc++.so))
LDFLAGS   = -Wl,-rpath,$(LIBDIR)

barenn: barenn.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f barenn

.PHONY: clean
