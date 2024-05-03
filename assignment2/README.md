# CLE2_T3G3

## How to run

For the steps tutorial use the root directory of the project.

### Prog1

To compile the ex1 run the following command:

```bash
mpicc prog1/main.c prog1/utils.c -Wall -O3 -o prog1/main
```

To run the ex1 run the following command changing the values between < and > to your desired values:

```bash
mpiexec -n <number-of-processes> ./prog1/main -f <regex-path-to-file> -b <buffer-size>
```

### Prog2

To compile the ex2 run the following command:

```bash
mpicc prog2/main.c -Wall -O3 -o prog2/main -lm
```

To run the ex2 run the following command changing the values between < and > to your desired values:

```bash
mpiexec -n <number-of-processes> ./prog2/main <path-to-file>
```

