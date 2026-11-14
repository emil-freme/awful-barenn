# Optimization flag evaluation: `barenn.cpp`

| | |
|---|---|
| **File** | `barenn.cpp` |
| **Date/time** | 2026-11-14 15:28:00 -03 |
| **Mode** | Serial (g++) |
| **Compiler** | g++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0 |
| **CPU** | Intel(R) Core(TM) Ultra 9 185H (22 logical cores) |
| **OS** | Linux 6.8.0-149-generic |
| **Git revision** | 27f88c4 |
| **Working directory** | `/home/galazzi/Documents/Insper_7_Semestre/Super_Computação/APS2/awful-barenn-Gabriel-Galazzi` |

## Summary

| Level | Flags | Compile | Compile time (s) | Run exit | Wall time (s) | Reported time (s) | Accuracy | Speedup vs none | Same output as none |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| none | `(none)` | ok | 0.630 | 0 | 5.931 | 5.8755 | 95.5% | 1.00x | baseline |
| -O1 | `-O1` | ok | 0.628 | 0 | 0.889 | 0.8576 | 95.5% | 6.85x | yes |
| -O2 | `-O2` | ok | 0.727 | 0 | 0.713 | 0.6831 | 95.5% | 8.60x | yes |
| -O3 | `-O3` | ok | 0.935 | 0 | 0.483 | 0.4523 | 95.5% | 12.99x | yes |
| -Ofast | `-Ofast` | ok | 0.940 | 0 | 0.358 | 0.3283 | 95.5% | 17.90x | **no** |

- Wall time covers the whole run (data loading included). Reported time is what the program prints.
- Speedup is relative to the `none` level and based on program-reported time.
- "Same output as none" compares the full output, skipping lines that match `[Tt]empo|[Tt]ime`.

## Detailed results

### none

```bash
g++ -std=c++17 barenn.cpp -o barenn
```

Compiled in 0.630 s. Run exited with code 0 after 5.931 s.

Program output:

```text
TREINO | 3823 imagens
Epoca  1/51 | perda media 0.4129
Epoca  6/51 | perda media 0.0442
Epoca 11/51 | perda media 0.0312
Epoca 16/51 | perda media 0.0261
Epoca 21/51 | perda media 0.0227
Epoca 26/51 | perda media 0.0204
Epoca 31/51 | perda media 0.0184
Epoca 36/51 | perda media 0.0169
Epoca 41/51 | perda media 0.0155
Epoca 46/51 | perda media 0.0145
Epoca 51/51 | perda media 0.0135
Tempo de treino 5.8755 s

AVALIACAO FINAL
Acuracia 95.5% | 1717 acertos em 1797 imagens
Perda media 0.0432

ALGUMAS INFERENCIAS
Esperado  Previsto
       0         0
       1         1
       2         8
       3         3
       5         9
       7         9

CONFUSOES MAIS FREQUENTES
Esperado  Previsto  Ocorrencias
       7         9            9
       8         1            6
       3         2            5
```

### -O1

```bash
g++ -std=c++17 -O1 barenn.cpp -o barenn
```

Compiled in 0.628 s. Run exited with code 0 after 0.889 s.

Program output:

```text
TREINO | 3823 imagens
Epoca  1/51 | perda media 0.4129
Epoca  6/51 | perda media 0.0442
Epoca 11/51 | perda media 0.0312
Epoca 16/51 | perda media 0.0261
Epoca 21/51 | perda media 0.0227
Epoca 26/51 | perda media 0.0204
Epoca 31/51 | perda media 0.0184
Epoca 36/51 | perda media 0.0169
Epoca 41/51 | perda media 0.0155
Epoca 46/51 | perda media 0.0145
Epoca 51/51 | perda media 0.0135
Tempo de treino 0.8576 s

AVALIACAO FINAL
Acuracia 95.5% | 1717 acertos em 1797 imagens
Perda media 0.0432

ALGUMAS INFERENCIAS
Esperado  Previsto
       0         0
       1         1
       2         8
       3         3
       5         9
       7         9

CONFUSOES MAIS FREQUENTES
Esperado  Previsto  Ocorrencias
       7         9            9
       8         1            6
       3         2            5
```

### -O2

```bash
g++ -std=c++17 -O2 barenn.cpp -o barenn
```

Compiled in 0.727 s. Run exited with code 0 after 0.713 s.

Program output:

```text
TREINO | 3823 imagens
Epoca  1/51 | perda media 0.4129
Epoca  6/51 | perda media 0.0442
Epoca 11/51 | perda media 0.0312
Epoca 16/51 | perda media 0.0261
Epoca 21/51 | perda media 0.0227
Epoca 26/51 | perda media 0.0204
Epoca 31/51 | perda media 0.0184
Epoca 36/51 | perda media 0.0169
Epoca 41/51 | perda media 0.0155
Epoca 46/51 | perda media 0.0145
Epoca 51/51 | perda media 0.0135
Tempo de treino 0.6831 s

AVALIACAO FINAL
Acuracia 95.5% | 1717 acertos em 1797 imagens
Perda media 0.0432

ALGUMAS INFERENCIAS
Esperado  Previsto
       0         0
       1         1
       2         8
       3         3
       5         9
       7         9

CONFUSOES MAIS FREQUENTES
Esperado  Previsto  Ocorrencias
       7         9            9
       8         1            6
       3         2            5
```

### -O3

```bash
g++ -std=c++17 -O3 barenn.cpp -o barenn
```

Compiled in 0.935 s. Run exited with code 0 after 0.483 s.

Program output:

```text
TREINO | 3823 imagens
Epoca  1/51 | perda media 0.4129
Epoca  6/51 | perda media 0.0442
Epoca 11/51 | perda media 0.0312
Epoca 16/51 | perda media 0.0261
Epoca 21/51 | perda media 0.0227
Epoca 26/51 | perda media 0.0204
Epoca 31/51 | perda media 0.0184
Epoca 36/51 | perda media 0.0169
Epoca 41/51 | perda media 0.0155
Epoca 46/51 | perda media 0.0145
Epoca 51/51 | perda media 0.0135
Tempo de treino 0.4523 s

AVALIACAO FINAL
Acuracia 95.5% | 1717 acertos em 1797 imagens
Perda media 0.0432

ALGUMAS INFERENCIAS
Esperado  Previsto
       0         0
       1         1
       2         8
       3         3
       5         9
       7         9

CONFUSOES MAIS FREQUENTES
Esperado  Previsto  Ocorrencias
       7         9            9
       8         1            6
       3         2            5
```

### -Ofast

```bash
g++ -std=c++17 -Ofast barenn.cpp -o barenn
```

Compiled in 0.940 s. Run exited with code 0 after 0.358 s.

Program output:

```text
TREINO | 3823 imagens
Epoca  1/51 | perda media 0.4129
Epoca  6/51 | perda media 0.0442
Epoca 11/51 | perda media 0.0312
Epoca 16/51 | perda media 0.0261
Epoca 21/51 | perda media 0.0227
Epoca 26/51 | perda media 0.0204
Epoca 31/51 | perda media 0.0184
Epoca 36/51 | perda media 0.0169
Epoca 41/51 | perda media 0.0156
Epoca 46/51 | perda media 0.0145
Epoca 51/51 | perda media 0.0135
Tempo de treino 0.3283 s

AVALIACAO FINAL
Acuracia 95.5% | 1716 acertos em 1797 imagens
Perda media 0.0436

ALGUMAS INFERENCIAS
Esperado  Previsto
       0         0
       1         1
       2         8
       3         3
       5         9
       7         9

CONFUSOES MAIS FREQUENTES
Esperado  Previsto  Ocorrencias
       7         9           11
       8         1            6
       3         2            5
```
