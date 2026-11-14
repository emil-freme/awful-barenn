# Optimization flag evaluation: `barenn.cpp`

| | |
|---|---|
| **File** | `barenn.cpp` |
| **Date/time** | 2026-11-14 15:28:24 -03 |
| **Mode** | CUDA (nvcc) |
| **Compiler** | Cuda compilation tools, release 13.2, V13.2.86; host: g++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0 |
| **CPU** | Intel(R) Core(TM) Ultra 9 185H (22 logical cores) |
| **GPU** | NVIDIA GeForce RTX 4070 Laptop GPU, 595.99.02 |
| **OS** | Linux 6.8.0-149-generic |
| **Git revision** | 27f88c4 |
| **Working directory** | `/home/galazzi/Documents/Insper_7_Semestre/Super_Computação/APS2/awful-barenn-Gabriel-Galazzi` |

## Summary

| Level | Flags | Compile | Compile time (s) | Run exit | Wall time (s) | Reported time (s) | Accuracy | Speedup vs none | Same output as none |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| none | `(none)` | ok | 2.138 | 0 | 5.997 | 5.9416 | 95.5% | 1.00x | baseline |
| -O1 | `-O1` | ok | 1.991 | 0 | 0.940 | 0.9125 | 95.5% | 6.51x | yes |
| -O2 | `-O2` | ok | 2.111 | 0 | 0.719 | 0.6871 | 95.5% | 8.65x | yes |
| -O3 | `-O3` | ok | 2.292 | 0 | 0.444 | 0.4126 | 95.5% | 14.40x | yes |
| -Ofast | `-use_fast_math -Xcompiler -Ofast` | ok | 2.270 | 0 | 0.364 | 0.3332 | 95.5% | 17.83x | **no** |

- Wall time covers the whole run (data loading included). Reported time is what the program prints.
- Speedup is relative to the `none` level and based on program-reported time.
- "Same output as none" compares the full output, skipping lines that match `[Tt]empo|[Tt]ime`.

## Detailed results

### none

```bash
nvcc -std=c++17 -x cu barenn.cpp -o barenn
```

Compiled in 2.138 s. Run exited with code 0 after 5.997 s.

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
Tempo de treino 5.9416 s

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
nvcc -std=c++17 -x cu -O1 barenn.cpp -o barenn
```

Compiled in 1.991 s. Run exited with code 0 after 0.940 s.

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
Tempo de treino 0.9125 s

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
nvcc -std=c++17 -x cu -O2 barenn.cpp -o barenn
```

Compiled in 2.111 s. Run exited with code 0 after 0.719 s.

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
Tempo de treino 0.6871 s

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
nvcc -std=c++17 -x cu -O3 barenn.cpp -o barenn
```

Compiled in 2.292 s. Run exited with code 0 after 0.444 s.

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
Tempo de treino 0.4126 s

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
nvcc -std=c++17 -x cu -use_fast_math -Xcompiler -Ofast barenn.cpp -o barenn
```

Compiled in 2.270 s. Run exited with code 0 after 0.364 s.

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
Tempo de treino 0.3332 s

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
