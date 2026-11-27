# -ffp-contract=off: sem FMA, para a saída bater bit a bit com a do legado
CXX      = g++
OPT      ?= -O3
CXXFLAGS  = -std=c++17 $(OPT) -ffp-contract=off -Wall -Wextra

# rpath: nos nós do Franky o LD_LIBRARY_PATH às vezes perde a libstdc++ do g++ 14
LIBDIR   := $(dir $(shell g++ -print-file-name=libstdc++.so))
LDFLAGS   = -Wl,-rpath,$(LIBDIR)

barenn: barenn.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f barenn

.PHONY: clean
