/**
 *  \file main.c (implementation file)
 *
 *  \brief Problem name: Bitonic sort.
 *  
 *  Synchronization based MPI (Message Passing Interface).
 *
 *  \author Rafael Gil & Diogo Magalhães - April 2024
 */

// compile:
//      mpicc -Wall -O3 -o main main.c -lm
// run:
//      mpiexec -n <number of processes> ./main <path to file>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <mpich/mpi.h>

#include "main.h"


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
int main(int argc, char *argv[]) {
    int gMemb[8];
    int rank, nProc, nProcNow, nIter, lenNumberArray;
    int *numberArray = NULL, *recNumberArray;

    MPI_Comm presentComm, nextComm;
    MPI_Group presentGroup, nextGroup;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProc);

    if(nProc != 1 && nProc != 2 && nProc != 4 && nProc != 8){
        if(rank == 0){
            fprintf(stderr, "Invalid number of threads\nValid: 1 / 2 / 4 / 8\n");
            printf("Exiting program...\n");
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    if (argc < 2){
        if(rank == 0){
            fprintf(stderr, "Invalid number of arguments\n");
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
	}

    if (rank == 0) {
        readFile(argv[1], &lenNumberArray, &numberArray);
    }

    // start timer
    (void) get_delta_time();

    // send the length to all processes
    MPI_Bcast(&lenNumberArray, 1, MPI_INT, 0, MPI_COMM_WORLD);

    recNumberArray = malloc(lenNumberArray * sizeof(int));

    if (recNumberArray == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    // define the number of iterations, based on the number of processes
    nIter = (int) (log2 (nProc) + 1.1);

    nProcNow = nProc;
    presentComm = MPI_COMM_WORLD;
    MPI_Comm_group (presentComm, &presentGroup);
    for (int i = 0; i < 8; i++){
        gMemb[i] = i;
    }

    for (int i = 0; i < nIter; i++){
        if (i > 0){
            MPI_Group_incl (presentGroup, nProcNow, gMemb, &nextGroup);
            MPI_Comm_create(presentComm, nextGroup, &nextComm);
            presentGroup = nextGroup;
            presentComm = nextComm;
            if (rank >= nProcNow){ 
                free (recNumberArray);
                MPI_Finalize ();
                return EXIT_SUCCESS;
            }
        }

        MPI_Comm_size (presentComm, &nProc);

        int offset = rank * (lenNumberArray / nProcNow);
        int endOffset = (offset + (lenNumberArray / nProcNow)) - 1;

        MPI_Scatter (numberArray + offset, lenNumberArray / nProcNow, MPI_INT, recNumberArray + offset, lenNumberArray / nProcNow, MPI_INT, 0, presentComm);

        if (i == 0){
            imperativeBitonicSort(recNumberArray, lenNumberArray / nProcNow, offset, endOffset);
        }else{
            merge(recNumberArray, lenNumberArray / nProcNow, offset, endOffset);
        }

        MPI_Gather (recNumberArray + offset, lenNumberArray / nProcNow, MPI_INT, numberArray + offset, lenNumberArray / nProcNow, MPI_INT, 0, presentComm);

        // equivalent to nProcNow = nProcNow / 2 but more efficient since its bit manipulation
        nProcNow = nProcNow >> 1;
    }

    // finish timer
    printf ("\nElapsed time = %.6f s\n", get_delta_time ());
    
    if(rank == 0){
        //check if array is correctly ordered
        verifyIfSequenceIsOrdered(lenNumberArray, &numberArray);
        free(numberArray);
    }

    MPI_Finalize();
    exit (EXIT_SUCCESS);
}

/**
 * \brief Read file and populate data array
 * 
 * \param fileName name of the file to read
 * \param lenNumberArray size of the array
 * \param numberArray array to be checked
*/
void readFile(char *fileName, int *lenNumberArray, int **numberArray) {
    FILE *file = fopen(fileName, "rb");
    if (file == NULL) {
        fprintf(stderr, "Unable to open the file.\n");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    int err = fread(lenNumberArray, sizeof(int), 1, file);
    if (err <= 0) {
        fprintf(stderr, "Error while reading file.\n");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    *numberArray = malloc(*lenNumberArray * sizeof(int));
    if (*numberArray == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    if (fread(*numberArray, sizeof(int), *lenNumberArray, file) != *lenNumberArray) {
        fprintf(stderr, "Error while reading numbers to array.\n");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    fclose(file);
}


/**
 * \brief checks if the number array is ordered in decreasing order
 * 
 * \param lenNumberArray size of the array
 * \param numberArray array to be checked
*/
void verifyIfSequenceIsOrdered(int lenNumberArray, int **numberArray) {
    for (int i = 0; i < lenNumberArray - 1; i++) {
        if ((*numberArray)[i] < (*numberArray)[i + 1]) {
            printf("Error in position %d between elements %d and %d\n", i, (*numberArray)[i], (*numberArray)[i + 1]);
            return;
        }
    }

    printf("Everything is OK!\n");
}


/**
 * \brief Print a formatted array
 * 
 * \param arr array to be printed
 * \param size size of the array 
*/
void print_int_array(int **arr, int size) {
    printf("Received array: [");
    for (int i = 0; i < size; i++) {
        if (i != 0) {
            printf(", ");
        }
        printf("%d", (*arr)[i]);
    }
    printf("]\n");
}


/**
 *  \brief implementation of the imperitive bitonic sort - descending order.
 *
 *  \param array array of numbers to be sorted
 *  \param N length of the array
 *  \param startIndex index where sub-sequence starts
 *  \param endIndex index where sub-sequence ends
 */
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



/**
 * \brief implements bitonic merge
 * 
 * \param array array of numbers to be sorted
 * \param N length of the array
 * \param startIndex index where sub-sequence starts
 * \param endIndex index where sub-sequence ends
*/
void merge(int *array, int N, int startIndex, int endIndex){
    int k = N;
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



/**
 *  \brief Get the process time that has elapsed since last call of this time.
 *
 *  \return process elapsed time
 */

static double get_delta_time(void){
  static struct timespec t0, t1;

  t0 = t1;

  if(clock_gettime (CLOCK_MONOTONIC, &t1) != 0){
    perror ("clock_gettime");
    exit(1);
  }

  return (double) (t1.tv_sec - t0.tv_sec) + 1.0e-9 * (double) (t1.tv_nsec - t0.tv_nsec);
}