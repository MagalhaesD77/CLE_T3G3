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

/** \brief length of the number array*/
static int lenNumberArray;

/** \brief length of the sub sequences */
static int lenSubSequences;

/** \brief number of sub sequences left */
static int toBeProcessed;

/** \brief number of work requests made*/
static int requestsMade;

/** \brief flag to signal the end of the program*/
static int finished;

/** \brief number of workers waiting for workload*/
static unsigned int lookingForWork;

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
    finished = 0;                                                       // initialize flag
    toBeProcessed = 0;                                                  // initialize flag
    lenNumberArray = 0;                                                 // initialize flag
    lenSubSequences = 0;                                                // initialize flag
    requestsMade = 0;                                                   // initialize flag

	pthread_cond_init (&waitForWorkAtribution, NULL);				    // initialize distributor synchronization point
	pthread_cond_init (&waitForWorkCompletion, NULL);			        // initialize distributor synchronization point
	pthread_cond_init (&waitForWorkRequest, NULL);			            // initialize workers synchronization point
}

/**
 * \brief Print a formatted array
 * 
 * \param arr array to be printed
 * \param size size of the array 
*/
/* static void print_int_array(int *arr, int size){
    printf("Printing array:\nSize: %d\n", size);
    printf("[");
    for (int i = 0; i < size; i++){
        if (i != 0){
            printf(", ");
        }
        printf("%d", arr[i]);
    }
    printf("]\n");
} */

/**
 * \brief Read file and populate data array
 * 
 * \param fileName name of the file to read
*/
void readFile(char *fileName){
    if ((statusDistributor = pthread_mutex_lock(&accessCR)) != 0){							/* enter monitor */
		errno = statusDistributor;															/* save error in errno */
		perror("error on entering monitor(CF)");
		statusDistributor = EXIT_FAILURE;
		pthread_exit(&statusDistributor);
	}
	pthread_once(&init, initialization); 

    FILE *file;
    file = fopen (fileName, "rb");
        
    // checkif it was able to open the file
    if(file == NULL ) {
        printf("Not able to open the file.\n");
        exit(1);
    }

    // read the first byte from the file
    // which represents the ammount of numbers the file has
    int err = fread(&lenNumberArray, sizeof(lenNumberArray), 1, file);
    if(err <= 0){
        printf("Error while reading file");
        exit(1);
    }

    // allocate memory for the array     
    // that will store the numbers read from the file
    numberArray = malloc(lenNumberArray * sizeof(int));
    if(numberArray == NULL){
        printf("Error while allocating memory");
        exit(1);
    }

    // read every number 
    if(fread(numberArray, sizeof(int), lenNumberArray, file) != lenNumberArray){
        printf("Error while reading numbers to array");
        exit(1);
    }

    // close file pointer
    fclose(file);

    if ((statusDistributor = pthread_mutex_unlock (&accessCR)) != 0){                                   /* exit monitor */
        errno = statusDistributor;                                                             /* save error in errno */
        perror ("error on exiting monitor(CF)");
        statusDistributor = EXIT_FAILURE;
        pthread_exit (&statusDistributor);
    }
}

/** 
 * \brief define the number of sub-sequences to be distributed 
 * 
 * \param numWorkers number of worker threads
*/
void defineSubSequences(int numWorkers){
    if ((statusDistributor = pthread_mutex_lock(&accessCR)) != 0){							/* enter monitor */
		errno = statusDistributor;															/* save error in errno */
		perror("error on entering monitor(CF)");
		statusDistributor = EXIT_FAILURE;
		pthread_exit(&statusDistributor);
	}
	pthread_once(&init, initialization);

    lenSubSequences = lenNumberArray / numWorkers;
    toBeProcessed = numWorkers;

    if ((statusDistributor = pthread_mutex_unlock (&accessCR)) != 0){                                   /* exit monitor */
        errno = statusDistributor;                                                             /* save error in errno */
        perror ("error on exiting monitor(CF)");
        statusDistributor = EXIT_FAILURE;
        pthread_exit (&statusDistributor);
    }
}

/**
 * \brief Calculate the number of sub-sequences to attribute to the workers and distribute them
 * 
 * \return flag that signals the end of the program
*/
int distributeWorkloads(){
    if ((statusDistributor = pthread_mutex_lock(&accessCR)) != 0){							/* enter monitor */
		errno = statusDistributor;															/* save error in errno */
		perror("error on entering monitor(CF)");
		statusDistributor = EXIT_FAILURE;
		pthread_exit(&statusDistributor);
	}
	pthread_once(&init, initialization);                                                    /* internal data initialization */

    // if there are no workers looking for work
    if(lookingForWork == 0){
        printf("Distributor is waiting for work requests\n");
        // wait to receive a work request
        if((statusDistributor = pthread_cond_wait(&waitForWorkRequest, &accessCR)) != 0){
            errno = statusDistributor;                            							/* save error in errno */
		    perror("error on waiting in waitForWorkRequest");
		    statusDistributor = EXIT_FAILURE;
		    pthread_exit(&statusDistributor);
        }
    }

    printf("Distributor is going to distribute work\n");

    // decrement the number of workers looking for work
    lookingForWork--;
    // increment the number of work requests made
    requestsMade++;
    // check if there have been enough requests to complete all the workloads
    if(requestsMade == toBeProcessed){
        finished = 1;
    }

    // signal that there is a new workload ready
    if((statusDistributor = pthread_cond_signal(&waitForWorkAtribution)) != 0){
        errno = statusDistributor;                                                                  /* save error in errno */
        perror("error on signaling in waitForWorkAttribution");
        statusDistributor = EXIT_FAILURE;
        pthread_exit (&statusDistributor);
    } 

    if ((statusDistributor = pthread_mutex_unlock (&accessCR)) != 0){                                   /* exit monitor */
        errno = statusDistributor;                                                             /* save error in errno */
        perror ("error on exiting monitor(CF)");
        statusDistributor = EXIT_FAILURE;
        pthread_exit (&statusDistributor);
    }

    return finished;
}

/**
 * \brief Request sequence to sort
 * 
 * \return the array of numbers to be process or -1 to signal the end of the program
*/

int* askForWorkload(int workerId, int *length, int *startIndex, int *endIndex){
    if ((statusWorkers[workerId] = pthread_mutex_lock(&accessCR)) != 0){								/* enter monitor */
		errno = statusWorkers[workerId];															/* save error in errno */
		perror("error on entering monitor(CF)");
		statusWorkers[workerId] = EXIT_FAILURE;
		pthread_exit(&statusWorkers[workerId]);
	}
	pthread_once(&init, initialization);                                                    /* internal data initialization */

    // add to the waiting line
    lookingForWork++;

    printf("Worker %d is asking for work\n", workerId);

    // signal distributor that the worker needs a workload
    if ((statusWorkers[workerId] = pthread_cond_signal(&waitForWorkRequest)) != 0){
		errno = statusWorkers[workerId];                         									/* save error in errno */
		perror ("error on signaling waitForWorkRequest");
		statusWorkers[workerId] = EXIT_FAILURE;
		pthread_exit(&statusWorkers[workerId]);
	}

    // wait for a workload attribution
    if((statusWorkers[workerId] = pthread_cond_wait(&waitForWorkAtribution, &accessCR)) != 0){
        errno = statusWorkers[workerId];                                                                /* save error in errno */
        perror("error on waiting for waitForWorkAttribution");
        statusWorkers[workerId] = EXIT_FAILURE;
        pthread_exit(&statusWorkers[workerId]);
    }

    printf("Worker %d was attributed work\n", workerId);

    //define length of the sub-sequence to sort
    *length = lenSubSequences;
    //define the index to the start of the sub-sequence
    *startIndex = (requestsMade - 1) * lenSubSequences;
    //define the index to the end of the sub-sequence
    *startIndex = (requestsMade* lenSubSequences) - 1;


    if ((statusWorkers[workerId] = pthread_mutex_unlock(&accessCR)) != 0){								/* exit monitor */
		errno = statusWorkers[workerId];															/* save error in errno */
		perror("error on entering monitor(CF)");
		statusWorkers[workerId] = EXIT_FAILURE;
		pthread_exit(&statusWorkers[workerId]);
	}

    if(finished == 1){
        return NULL;
    }

    return numberArray;
}