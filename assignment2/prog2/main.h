/**
 *  \file main.h (interface file)
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


#ifndef PROBCONST_H_
#define PROBCONST_H_

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
 * \brief Print a formatted array
 * 
 * \param arr array to be printed
 * \param size size of the array 
 */
void print_int_array(int **arr, int size);

/**
 *  \brief implementation of the imperitive bitonic sort - descending order.
 *
 *  \param array array of numbers to be sorted
 *  \param N length of the array
 *  \param startIndex index where sub-sequence starts
 *  \param endIndex index where sub-sequence ends
 */
void imperativeBitonicSort(int *array, int N, int startIndex, int endIndex);

/**
 * \brief implements bitonic merge
 * 
 * \param array array of numbers to be sorted
 * \param N length of the array
 * \param startIndex index where sub-sequence starts
 * \param endIndex index where sub-sequence ends
*/
void merge(int *array, int N, int startIndex, int endIndex);

/**
 *  \brief Get the process time that has elapsed since last call of this time.
 *
 *  \return process elapsed time
 */
static double get_delta_time(void);

#endif /* PROBCONST_H_ */

