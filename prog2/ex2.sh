#!/bin/bash


# gcc -Wall -O3 -o countWords countWords.c
# -Wall: All warnings should be printed
# -O3: Level 3 of optimization (Maximum level of optimization)
compile_output=$(gcc prog2/ex2.c -Wall -O3 -o prog2/ex2)

if [ $? -eq 0 ]; then
    echo "Compilation successful"
    ./prog2/ex2 "$@"
else
    echo "Compilation failed"
fi


