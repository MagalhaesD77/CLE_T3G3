/**
 *  \file main.c (implementation file)
 *
 *  \brief Problem name: Bitonic sort.
 *  
 *  CUDA
 * 
 *  \author Rafael Gil & Diogo Magalhães - May 2024
 */

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "common.h"
#include <cuda_runtime.h>

#define N 1024  // length of the row 

/**
 * \brief Read file and populate data array
 * 
 * \param fileName name of the file to read
 * \param lenNumberArray size of the array
 * \param numberArray array to be checked
*/
void readFile(char *fileName, int *lenNumberArray, int **numberArray);

/**
 * \brief checks if the number array is ordered in decreasing order
 * 
 * \param lenNumberArray size of the array
 * \param numberArray array to be checked
*/
void verifyIfSequenceIsOrdered(int lenNumberArray, int **numberArray);

/**
 *  \brief implementation of the imperitive bitonic sort - descending order.
 *
 *  \param array array of numbers to be sorted
 *  \param N length of the array
 *  \param startIndex index where sub-sequence starts
 *  \param endIndex index where sub-sequence ends
 */
__global__ static void bitonicSortOnGPU(int *array, int lenSubSequence, int iter);

/**
 *  \brief Get the process time that has elapsed since last call of this time.
 *
 *  \return process elapsed time
 */

static double get_delta_time(void);

/**
 *  \brief Main thread.
 *
 *  Starts the simulation.
 *
 *  \param argc number of words of the command line
 *  \param argv list of words of the command line
 *
 *  \return status of operation
 */
int main(){

    printf("%s Starting...\n", argv[0]);

    if (argc != 2){
        fprintf(stderr, "Invalid number of arguments\n");
        exit(EXIT_FAILURE);
    }

    // set up device
    int dev = 0;

    cudaDeviceProp deviceProp;
    CHECK (cudaGetDeviceProperties (&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK (cudaSetDevice (dev)); // the gpu its going to use

    // number sequence
    int *numberSequence = NULL;
    
    // length of number sequence
    int lenNumberSequence = 0;

    // read sequence from file
    readFile(argv[1], &lenNumberSequence, &numberSequence);

    if(lenNumberSequence > (size_t)5e9){
        fprintf(stderr, "Sequence too big. The GPU can only withstand 5Gb of data\n");
        exit(EXIT_FAILURE);
    }

    // alocate memory for the sequence in the GPU
    int *gpuSequence = NULL;
    CHECK(cudaMalloc((void**)&gpuSequence, lenNumberSequence * sizeof(int)));

    // copy sequence from host to gpu
    CHECK(cudaMemcpy(gpuSequence, numberSequence, lenNumberSequence * sizeof(int), cudaMemcpyHostToDevice));

    // kernel configuration
    unsigned int gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ;
	
	blockDimX = 1 << 0; // optimize!
	blockDimY = 1 << 0; // optimize!
	blockDimZ = 1 << 0; // do not change!
	gridDimX = 1 << 0; 	// optimize!
	gridDimY = 1 << 0;  // optimize!
	gridDimZ = 1 << 0;  // do not change!

    // define the number of iterations necessary
    int nIter = (int) (log2 (N) + 1.1);

    (void) get_delta_time();

    for (int i = 0; i < nIter; i++)
    {
        // DO THE ITERATION AND SORTING PROCESS

        dim3 grid(gridDimX, gridDimY, gridDimZ);
		dim3 block(blockDimX, blockDimY, blockDimZ);

        // calculate the size of the sub-sequence
        int lenSubSeqeunce = (int)(1 << i) * N;

        bitonicSortOnGPU<<<grid, block>>>(gpuSequence, lenSubSeqeunce, i);

        // synchronize and wait for all the threads
        CHECK(cudaDeviceSynchronize());

        // check if any error occured	
		CHECK(cudaGetLastError());
    }

    printf("\nElapsed time = %.6f s\n", get_delta_time());

    // copy sequence from gpu to host
    CHECK(cudaMemcpy(numberSequence, gpuSequence, lenNumberSequence * sizeof(int), cudaMemcpyDeviceToHost));

    // free alocated memory on the gpu
    CHECK(cudeFree(gpuSequence))

    // reset the gpu
    CHECK(cudaDeviceReset())

    //check if array is correctly ordered
    verifyIfSequenceIsOrdered(lenNumberSequence, &numberSequence);

    // free alocated memory
    free(numberSequence)

    exit(EXIT_SUCCESS);
}

void readFile(char *fileName, int *lenNumberArray, int **numberArray) {
    FILE *file = fopen(fileName, "rb");
    if (file == NULL) {
        fprintf(stderr, "Unable to open the file.\n");
        exit(EXIT_FAILURE);
    }

    int err = fread(lenNumberArray, sizeof(int), 1, file);
    if (err <= 0) {
        fprintf(stderr, "Error while reading file.\n");
        exit(EXIT_FAILURE);
    }

    *numberArray = malloc(*lenNumberArray * sizeof(int));
    if (*numberArray == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    if (fread(*numberArray, sizeof(int), *lenNumberArray, file) != *lenNumberArray) {
        fprintf(stderr, "Error while reading numbers to array.\n");
        exit(EXIT_FAILURE);
    }

    fclose(file);
}

void verifyIfSequenceIsOrdered(int lenNumberArray, int **numberArray) {
    for (int i = 0; i < lenNumberArray - 1; i++) {
        if ((*numberArray)[i] < (*numberArray)[i + 1]) {
            printf("Error in position %d between elements %d and %d\n", i, (*numberArray)[i], (*numberArray)[i + 1]);
            return;
        }
    }

    printf("Everything is OK!\n");
}

void imperativeBitonicSort(int *array, int N, int startIndex, int endIndex){
    // iterate through the powers of 2 up to N
    // simulates the layers of the algorithm
    for (int k = 2; k <= N; k = 2 * k) {
        // iterate through half of the current value of k
        // controls the length of the comparison between the numbers
        for (int j = k / 2; j > 0; j = j / 2) {
            // iterates through the partition of the array
            for (int i = startIndex; i <= endIndex; i++) {
                int ij = i ^ j;     // bitwise XOR, to calculate the index where to perform the comparison
                if ((ij) > i) {     // assure correct order
                    if (((i & k) == 0                               // bitwise AND to check if i-th index is in the lower half of the bitonic sequence
                                && array[i] < array[ij])            // check if i-th element is smaller than ij
                        || ((i & k) != 0                            // bitwise AND to check if i-th index is in the upper half of the bitonic sequence
                                && array[i] > array[ij])) {         // check if i-th element is greater than ij

                        // performs a common swap between the elements of the array
                        int aux = array[i];
                        array[i] = array[ij];
                        array[ij] = aux;
                    }
                }
            }
        }
    }
}


__global__ void bitonicSortOnGPU(int *array, int lenSubSequence, int iter)
{
    int x, y, idx;

    // calculate the segment of the sequence for the current thread 
    x = (int)threadIdx.x + (int)blockDim.x * (int)blockIdx.x;
    y = (int)threadIdx.y + (int)blockDim.y * (int)blockIdx.y;
    idx = (int)blockDim.x * (int)gridDim.x * y + x;

    if (lenSubSequence == N){   // bitonic sort
        // iterate through the powers of 2 up to N
        // simulates the layers of the algorithm
        for (int k = 2; k <= lenSubSequence; k = 2 * k) {
            // iterate through half of the current value of k
            // controls the length of the comparison between the numbers
            for (int j = k / 2; j > 0; j = j / 2) {
                // iterates through the partition of the array
                for (int i = 0; i <= endIndex; i++) {
                    int elem = N * (1 << iter) * idx + i;
                    int ij = elem ^ j;     // bitwise XOR, to calculate the index where to perform the comparison
                    if ((ij) > elem) {     // assure correct order
                        if (((elem & k) == 0                               // bitwise AND to check if i-th index is in the lower half of the bitonic sequence
                                    && array[elem] < array[ij])            // check if i-th element is smaller than ij
                            || ((elem & k) != 0                            // bitwise AND to check if i-th index is in the upper half of the bitonic sequence
                                    && array[elem] > array[ij])) {         // check if i-th element is greater than ij

                            // performs a common swap between the elements of the array
                            int aux = array[elem];
                            array[elem] = array[ij];
                            array[ij] = aux;
                        }
                    }
                }
            }
        }
    }else{  // bitonic merge
        int k = lenSubSequence;
        for (int j = k / 2; j > 0; j = j / 2) {
                // iterates through the partition of the array
                for (int i = 0; i <= endIndex; i++) {
                    int elem = N * (1 << iter) * idx + i;
                    int ij = elem ^ j;     // bitwise XOR, to calculate the index where to perform the comparison
                    if ((ij) > elem) {     // assure correct order
                        if (((elem & k) == 0                               // bitwise AND to check if i-th index is in the lower half of the bitonic sequence
                                    && array[elem] < array[ij])            // check if i-th element is smaller than ij
                            || ((elem & k) != 0                            // bitwise AND to check if i-th index is in the upper half of the bitonic sequence
                                    && array[elem] > array[ij])) {         // check if i-th element is greater than ij

                            // performs a common swap between the elements of the array
                            int aux = array[elem];
                            array[elem] = array[ij];
                            array[ij] = aux;
                        }
                    }
                }
        }
    }
    
}

static double get_delta_time(void)
{
    static struct timespec t0, t1;

    t0 = t1;

    if (clock_gettime(CLOCK_MONOTONIC, &t1) != 0)
    {
        perror("clock_gettime");
        exit(1);
    }

    return (double)(t1.tv_sec - t0.tv_sec) + 1.0e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
}