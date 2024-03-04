#!/bin/bash


# gcc -Wall -O3 -o countWords countWords.c
# -Wall: All warnings should be printed
# -O3: Level 3 of optimization (Maximum level of optimization)
compile_output=$(gcc prog1/ex1.c -Wall -O3 -o prog1/ex1)

if [ $? -eq 0 ]; then
    echo "Compilation successful"
    ./prog1/ex1 "$@"
else
    echo "Compilation failed"
fi


