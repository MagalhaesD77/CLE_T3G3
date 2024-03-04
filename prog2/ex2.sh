#!/bin/bash


# gcc -Wall -O3 -o countWords countWords.c
# -Wall: All warnings should be printed
# -O3: Level 3 of optimization (Maximum level of optimization)
compile_output=$(gcc ex2.c -Wall -O3 -o ex2)

if [ $? -eq 0 ]; then
    echo "Compilation successful"
    ./ex2 "$@"
else
    echo "Compilation failed"
fi


