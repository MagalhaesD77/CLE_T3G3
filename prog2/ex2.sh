#!/bin/bash

# Compile the C program
gcc -Wall -O3 -o bitonicSort bitonic-sort-multithread.c synchronization.c -lpthread

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. Running the program..."
    # Run the compiled program with command line arguments
    ./bitonicSort "$@"
else
    echo "Compilation failed. Please check your code."
fi