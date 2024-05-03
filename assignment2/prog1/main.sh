#!/bin/bash

clear

compile_output=$(mpicc prog1/main.c prog1/utils.c -Wall -O3 -o prog1/main)

if [ $? -eq 0 ]; then
    printf "Compilation successful\n\n"
    mpiexec -n 8 ./prog1/main "$@"
else
    printf "Compilation failed\n\n"
fi

