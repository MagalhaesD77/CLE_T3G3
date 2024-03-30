# CLE1_T3G3

## How to run

For the steps tutorial use the root directory of the project.

### Prog1

To compile the ex1 run the following command:

```bash
gcc prog1/ex1.c prog1/sharedRegion.c prog1/utils.c -Wall -O3 -o prog1/ex1 -pthread
```

To run the ex1 run the following command changing the values between < and > to your desired values:

```bash
./prog1/ex1 -f <regex-path-to-file> -t <number-of-threads> -b <buffer-size>
```

### Prog2

To compile the ex2 run the following command:

```bash
gcc prog2/bitonicSort.c prog2/synchronization.c -Wall -O3 -o prog2/bitonicSort -pthread
```

To run the ex2 run the following command changing the values between < and > to your desired values:

```bash
./prog2/bitonicSort <path-to-file> <number-of-threads>
```

