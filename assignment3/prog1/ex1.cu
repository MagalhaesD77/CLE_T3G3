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
#include <getopt.h>
#include "common.h"
#include <cuda_runtime.h>

#define DEFAULT_NUMBER_THREADS 1024 // Default length of the row

/**
 *  \brief Argument Parser.
 *
 *  \param argc number of command line arguments
 *  \param argv array of command line arguments
 *  \param numThreads number of GPU threads to use
 *  \param fileName name of the file to read
 */
void cli_parser(int argc, char *argv[], int *numThreads, char **fileName);

/**
 *  \brief Print Usage of the program.
 *
 *  \param cmdName command name
 */
static void printUsage (char *cmdName);

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
void verifyIfSequenceIsOrdered(int *lenNumberArray, int **numberArray);

/**
 *  \brief implementation of the imperitive bitonic sort - descending order.
 *
 *  \param array array of numbers to be sorted
 *  \param N length of the array
 *  \param startIndex index where sub-sequence starts
 *  \param endIndex index where sub-sequence ends
 */
__global__ static void bitonicSortOnGPU(int *numArray, int N);

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
int main(int argc, char **argv)
{
    int numThreads;
    char *fileName;

    // Parse the command line arguments
    cli_parser(argc, argv, &numThreads, &fileName);

    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;

    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev)); // the gpu its going to use

    // number sequence
    int *numberSequence = NULL;

    // length of number sequence
    int lenNumberSequence = 0;

    // read sequence from file
    readFile(fileName, &lenNumberSequence, &numberSequence);

    if (lenNumberSequence > (size_t)5e9)
    {
        fprintf(stderr, "Sequence is too big. The GPU can only withstand 5Gb of data\n");
        exit(EXIT_FAILURE);
    }

    // alocate memory for the sequence in the GPU
    int *gpuSequence = NULL;
    CHECK(cudaMalloc((void **)&gpuSequence, lenNumberSequence * sizeof(int)));

    // copy sequence from host to gpu
    CHECK(cudaMemcpy(gpuSequence, numberSequence, lenNumberSequence * sizeof(int), cudaMemcpyHostToDevice));

    // kernel configuration
    unsigned int gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ;

    blockDimX = numThreads;
    blockDimY = 1 << 0;
    blockDimZ = 1 << 0;
    gridDimX = 1 << 0;
    gridDimY = 1 << 0;
    gridDimZ = 1 << 0;

    dim3 grid(gridDimX, gridDimY, gridDimZ);
    dim3 block(blockDimX, blockDimY, blockDimZ);

    (void)get_delta_time();

    bitonicSortOnGPU<<<grid, block>>>(gpuSequence, lenNumberSequence);

    // synchronize and wait for all the threads
    CHECK(cudaDeviceSynchronize());
    
    printf("\nElapsed time = %.6f s\n", get_delta_time());

    // check if any error occured
    CHECK(cudaGetLastError());

    // copy sequence from gpu to host
    CHECK(cudaMemcpy(numberSequence, gpuSequence, lenNumberSequence * sizeof(int), cudaMemcpyDeviceToHost));

    // free alocated memory on the gpu
    CHECK(cudaFree(gpuSequence));

    // reset the gpu
    CHECK(cudaDeviceReset());

    // check if array is correctly ordered
    verifyIfSequenceIsOrdered(&lenNumberSequence, &numberSequence);

    // free alocated memory
    free(numberSequence);

    exit(EXIT_SUCCESS);
}


void readFile(char *fileName, int *lenNumberArray, int **numberArray)
{
    FILE *file = fopen(fileName, "rb");
    if (file == NULL)
    {
        fprintf(stderr, "Unable to open the file.\n");
        exit(EXIT_FAILURE);
    }

    int err = fread(lenNumberArray, sizeof(int), 1, file);
    if (err <= 0)
    {
        fprintf(stderr, "Error while reading file.\n");
        exit(EXIT_FAILURE);
    }

    *numberArray = (int *)malloc(*lenNumberArray * sizeof(int));
    if (*numberArray == NULL)
    {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    if (fread(*numberArray, sizeof(int), *lenNumberArray, file) != *lenNumberArray)
    {
        fprintf(stderr, "Error while reading numbers to array.\n");
        exit(EXIT_FAILURE);
    }

    fclose(file);
}


void verifyIfSequenceIsOrdered(int *lenNumberArray, int **numberArray)
{
    int N_ = 5;
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
        printf("%d\t", (*numberArray)[i * N_ + j]);
        }
        printf("\n");
    }

    for (int i = 0; i < *lenNumberArray - 1; i++)
    {
        if ((*numberArray)[i] < (*numberArray)[i + 1])
        {
            printf("Error in position %d between elements %d and %d\n", i, (*numberArray)[i], (*numberArray)[i + 1]);
            return;
        }
    }

    printf("Everything is OK!\n");
}


__global__ void bitonicSortOnGPU(int *numArray, int N)
{
    int x, y, idx;

    // calculate the segment of the sequence for the current thread
    x = (int)threadIdx.x + (int)blockDim.x * (int)blockIdx.x;
    y = (int)threadIdx.y + (int)blockDim.y * (int)blockIdx.y;
    idx = (int)blockDim.x * (int)gridDim.x * y + x;

    // Calculate the number of threads
    int K = blockDim.x * gridDim.x;

    // If the number of threads is greater than N/2, it will never sort the sequence so it is necessary to limit the number of threads
    if (K > N/2)
        K = N/2;

    // define the number of iterations necessary
    int nIter = (int)(log2(K) + 1.1);

    for (int iter = 0; iter < nIter; iter++)
    {
        // synchronize all threads
        __syncthreads();

        if (idx >= (K >> iter)) // check if the thread is active
        {
            continue;
        }

        if (iter == 0){
            // iterate through the powers of 2 up to N
            // simulates the layers of the algorithm
            for (int k = 2; k <= (1 << iter) * N/K; k *= 2){    // power of 2 until len of the subsequence
            // iterate through half of the current value of k
            // controls the length of the comparison between the numbers
                for (int j = k / 2; j > 0; j /= 2){
                    // iterates through the partition of the array
                    for (int i = 0; i < (1 << iter) * N/K; i++){
                        int elem = N/K * (1 << iter) * idx + i;
                        int l = elem ^ j;           // bitwise XOR, to calculate the index where to perform the comparison
                        if (l > elem){              // assure correct order
                            if ((((elem & k) == 0)                          // bitwise AND to check if i-th index is in the lower half of the bitonic sequence
                                && (numArray[elem] < numArray[l]))      // check if i-th element is smaller than ij
                                || (((elem & k) != 0)                   // bitwise AND to check if i-th index is in the upper half of the bitonic sequence
                                && (numArray[elem] > numArray[l]))){     // check if i-th element is greater than ij
                            
                                // performs a common swap between the elements of the array
                                int aux = numArray[elem];
                                numArray[elem] = numArray[l];
                                numArray[l] = aux;
                            }
                        }
                    }
                }
            }
        }
        else{
            int k = (1 << iter) * N/K;
            for (int j = k / 2; j > 0; j /= 2){
                // iterates through the partition of the array
                for (int i = 0; i < (1 << iter) * N/K; i++){
                    int elem = N/K * (1 << iter) * idx + i;
                    int l = elem ^ j;           // bitwise XOR, to calculate the index where to perform the comparison
                    if (l > elem){              // assure correct order
                        if ((((elem & k) == 0)                          // bitwise AND to check if i-th index is in the lower half of the bitonic sequence
                            && (numArray[elem] < numArray[l]))      // check if i-th element is smaller than ij
                            || (((elem & k) != 0)                   // bitwise AND to check if i-th index is in the upper half of the bitonic sequence
                            && (numArray[elem] > numArray[l]))){     // check if i-th element is greater than ij
                        
                            // performs a common swap between the elements of the array
                            int aux = numArray[elem];
                            numArray[elem] = numArray[l];
                            numArray[l] = aux;
                        }
                    }
                }
            }
        }
    }
}


void cli_parser(int argc, char *argv[], int *numThreads, char **fileName)
{
  int t_flag = 0;
  int numFiles = 0;
  int opt;
  while ((opt = getopt(argc, argv, "f:t:")) != -1) {
    switch (opt) {
    case 'f':
      // get the text file names by processing the command line and storing them in the shared region (This work with multiple arguments for example using dataset1/text*.txt)
      for (int i = optind - 1; i < argc && argv[i][0] != '-'; ++i) {
        
        if (numFiles > 0){
          fprintf(stderr, "Error: Only one file can be processed\n");
          exit(EXIT_FAILURE);
        }

        (*fileName) = argv[i];
        
        numFiles++;
      }
      break;

    case 't':
      // Only allow the -b flag to be used once
      if(t_flag == 1){
        fprintf(stderr, "Error: -t flag can only be used once\n");
        exit(EXIT_FAILURE);
      }

      // Check if the buffer size is a valid positive integer and a power of 2
        if (atoi(optarg) <= 0 || (atoi(optarg) & (atoi(optarg) - 1)) != 0){
            fprintf(stderr, "Error: Buffer size must be a positive integer and a power of 2\n");
            exit(EXIT_FAILURE);
        }

      // Set the buffer size
      t_flag = 1;
      (*numThreads) = atoi(optarg);
      break;

    default:
      printUsage(basename (argv[0]));
      exit(EXIT_FAILURE);
    }
  }

  // Check if the -f flag was used
  if (numFiles <= 0)
  {
      printf("No file provided\n");
      printUsage(basename (argv[0]));
      exit(EXIT_FAILURE);
  }

  // Check if the -b flag was used
  if(t_flag == 0){
      (*numThreads) = DEFAULT_NUMBER_THREADS;
      printf("-t not defined. Using default value of %d buffer size\n\n", *numThreads);
  }
    
}


static void printUsage (char *cmdName)
{
  fprintf (stderr, "\nSynopsis: %s [OPTIONS]\n"
           "  OPTIONS:\n"
           "  -f fileName   --- name of the file or multiple files to be processed\n"
           "  -t numThreads --- number of threads to be used\n"
           "  -h            --- print this help\n", cmdName);
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
