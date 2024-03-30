#!/bin/bash

clear

# gcc -Wall -O3 -o countWords countWords.c
# -Wall: All warnings should be printed
# -O3: Level 3 of optimization (Maximum level of optimization)
compile_output=$(gcc prog1/ex1.c prog1/sharedRegion.c prog1/utils.c -Wall -O3 -o prog1/ex1 -pthread)

if [ $? -eq 0 ]; then
    printf "Compilation successful\n\n"
    ./prog1/ex1 "$@"
else
    printf "Compilation failed\n\n"
fi


