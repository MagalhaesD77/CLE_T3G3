/**
 *  \file fifo.c (implementation file)
 *
 *  \brief Problem name: Producers / Consumers.
 *
 *  Synchronization based on monitors.
 *  Both threads and the monitor are implemented using the pthread library which enables the creation of a
 *  monitor of the Lampson / Redell type.
 *
 *  Data transfer region implemented as a monitor.
 *
 *  Definition of the operations carried out by the producers / consumers:
 *     \li putVal
 *     \li getVal.
 *
 *  \author António Rui Borges - March 2023
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <pthread.h>
#include <errno.h>

#include "constants.h"

/** \brief distributor thread return status */
extern int statusDistributor;

/** \brief workers thread return status arry */
extern int *statusWorkers;

/** \brief distributor thread return status */
extern int statusDistributor;

/** \brief array to store the numbers read from the file */
static int *numberArray;

/** \brief locking flag which warrants mutual exclusion inside the monitor */
static pthread_mutex_t accessCR = PTHREAD_MUTEX_INITIALIZER;

/** \brief flag which warrants that the data transfer region is initialized exactly once */
static pthread_once_t init = PTHREAD_ONCE_INIT;;

/** \brief workers synchronization point when waiting to receive workload */
static pthread_cond_t waitForWorkAtribution;

/** \brief distributor synchronization point when the workers ask for a new workload */
static pthread_cond_t waitForWorkRequest;

/** \brief distributor synchronization point when workers signal the completion of their work */
static pthread_cond_t waitForWorkCompletion;

/**
 *  \brief Initialization of the data transfer region.
 *
 *  Internal monitor operation.
 */

static void initialization (void)
{
	pthread_cond_init (&waitForWorkAtribution, NULL);				  // initialize distributor synchronization point
	pthread_cond_init (&waitForWorkCompletion, NULL);			    // initialize distributor synchronization point
	pthread_cond_init (&waitForWorkRequest, NULL);			      // initialize workers synchronization point
}

/**
 * \brief Print a formatted array
 * 
 * \param arr array to be printed
 * \param size size of the array 
*/
static void print_int_array(int *arr, int size){
    printf("Printing array:\nSize: %d\n", size);
    printf("[");
    for (int i = 0; i < size; i++){
        if (i != 0){
            printf(", ");
        }
        printf("%d", arr[i]);
    }
    printf("]\n");
}

/**
 * \brief Read file and populate data array
 * 
 * \param fileName name of the file to read
*/
void readFile(char *fileName){
    FILE *file;
    file = fopen (fileName, "rb");
        
    // checkif it was able to open the file
    if(file == NULL ) {
        printf("Not able to open the file.\n");
        exit(1);
    }

    // read the first byte from the file
    // which represents the ammount of numbers the file has
    int array_size;
    int err = fread(&array_size, sizeof(array_size), 1, file);
    if(err <= 0){
        printf("Error while reading file");
        exit(1);
    }

    // allocate memory for the array     
    // that will store the numbers read from the file
    numberArray = malloc(array_size * sizeof(int));
    if(numberArray == NULL){
        printf("Error while allocating memory");
        exit(1);
    }

    // read every number 
    if(fread(numberArray, sizeof(int), array_size, file) != array_size){
        printf("Error while reading numbers to array");
        exit(1);
    }

    // close file pointer
    fclose(file);

    print_int_array(numberArray, array_size);
}