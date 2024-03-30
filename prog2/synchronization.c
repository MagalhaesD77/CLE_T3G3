/**
 *  \file synchronization.c (implementation file)
 *
 *  \brief Problem name: Bitonic sort.
 *
 *  Synchronization based on monitors.
 *  Both threads and the monitor are implemented using the pthread library which enables the creation of a
 *  monitor of the Lampson / Redell type.
 *
 *  Data transfer region implemented as a monitor.
 *
 *  Definition of the operations carried out by the distributor / workers:
 *     \li read file
 *     \li defineSubSequences
 *     \li distributeWorkloadsimplementation
 *     \li askForWorkloads
 *
 *  \author Rafael Gil & Diogo Magalhães - March 2024
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>

#include "constants.h"

/** \brief distributor thread return status */
extern int statusDistributor;

/** \brief workers thread return status array */
extern int *statusWorkers;

/** 
 * \brief workers activity status
 *
 * 1 --> completed work
 * 0 --> waiting for work
 * -1 --> exited
 */
static int *activeWorkers;

/** \brief initial number of workers*/
static int initialNumberWorkers;

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

/** \brief number of workloads taken by workers */
static int workRequestsAttributed; 

/** \brief number of the current iteration */
static int currentIteration; 

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
    workRequestsAttributed = 0;                                         // initialize flag
    currentIteration = 0;                                               // initialize flag

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
static void print_int_array(int *arr, int size){
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
 * \brief checks if the number array is ordered in decreasing order
*/
void verifyIfSequenceIsOrdered(){
    for (int i = 0; i < lenNumberArray - 1; i++) {
        if (numberArray[i] < numberArray[i + 1]) {
            printf ("Error in position %d between element %d and %d\n", i, numberArray[i], numberArray[i+1]);
            return;
        }
    }
    
    printf ("Everything is OK!\n");
}

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
        exit(EXIT_FAILURE);
    }

    // read the first byte from the file
    // which represents the ammount of numbers the file has
    int err = fread(&lenNumberArray, sizeof(lenNumberArray), 1, file);
    if(err <= 0){
        printf("Error while reading file");
        exit(EXIT_FAILURE);
    }

    // allocate memory for the array     
    // that will store the numbers read from the file
    numberArray = malloc(lenNumberArray * sizeof(int));
    if(numberArray == NULL){
        printf("Error while allocating memory");
        exit(EXIT_FAILURE);
    }

    // read every number 
    if(fread(numberArray, sizeof(int), lenNumberArray, file) != lenNumberArray){
        printf("Error while reading numbers to array");
        exit(EXIT_FAILURE);
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
    toBeProcessed = initialNumberWorkers = numWorkers;
    if((activeWorkers = malloc(numWorkers * sizeof(int))) == NULL){
        fprintf(stderr, "Error allocating memory for activeWorkers array\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < numWorkers; i++)
    {
        activeWorkers[i] = 0;               // every worker is waiting for work at the start
    }

    if ((statusDistributor = pthread_mutex_unlock (&accessCR)) != 0){                                   /* exit monitor */
        errno = statusDistributor;                                                             /* save error in errno */
        perror ("error on exiting monitor(CF)");
        statusDistributor = EXIT_FAILURE;
        pthread_exit (&statusDistributor);
    }
}

/**
 * \brief check if every active worker has completed their assigned work
 * 
 * distributor auxiliary function 
*/
static int checkIfEveryWorkerHasCompletedWork(){
    for (int i = 0; i < initialNumberWorkers; i++)
    {
        if(activeWorkers[i] == 0){
            return 0;
        }
    }

    return 1;
}

/**
 * \brief will update the activity status of the workers, selecting which workers will be working for the next iteration
 * 
 * distributor auxiliary function 
*/
void updateWorkerActivityStatus(){
    int amountToNextIteration = 0;
    for (int i = 0; i < initialNumberWorkers; i++){
        if(amountToNextIteration == toBeProcessed){
            break;
        }
        if(activeWorkers[i] == 1){
            activeWorkers[i] = 0;
            amountToNextIteration++;
        }
    }

    // If it is the last iteration
    if(toBeProcessed == 1){
        for (int  i = 0; i < initialNumberWorkers; i++){
            if(activeWorkers[i] != -1){
                activeWorkers[i] = -1;
            }
        }   
        print_int_array(activeWorkers, initialNumberWorkers);
        return;
    }
    int changed = 0;
    for (int  i = 0; i < initialNumberWorkers; i++){
        if(activeWorkers[i] == -1){
            continue;
        }
        if(changed == (toBeProcessed/2)){
		    break;
        }
        activeWorkers[i] = -1;
        changed++;
    }
    print_int_array(activeWorkers, initialNumberWorkers);
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

    // decrement the number of workers looking for work
    lookingForWork--;

    // signal that there is a new workload ready
    if((statusDistributor = pthread_cond_signal(&waitForWorkAtribution)) != 0){
        errno = statusDistributor;                                                                  /* save error in errno */
        perror("error on signaling in waitForWorkAttribution");
        statusDistributor = EXIT_FAILURE;
        pthread_exit (&statusDistributor);
    } 

    // check if there have been enough requests to complete all the workloads of the current iteraction
    // and if they have been taken by workers
    if(requestsMade >= toBeProcessed && workRequestsAttributed >= toBeProcessed){
        // wait for every active worker to finish their work
        while (checkIfEveryWorkerHasCompletedWork() == 0)
        {
            printf("Distributor is waiting for every worker to finish their work\n");
            if((statusDistributor = pthread_cond_wait(&waitForWorkCompletion, &accessCR)) != 0){
                errno = statusDistributor;                            							/* save error in errno */
                perror("error on waiting in waitForWorkRequest");
                statusDistributor = EXIT_FAILURE;
                pthread_exit(&statusDistributor);
            }
        }

        printf("Workers activity status:\n");
        updateWorkerActivityStatus();
        lookingForWork = 0;
        requestsMade = 0;
        workRequestsAttributed = 0;
        toBeProcessed = toBeProcessed/2;
        currentIteration++;
        if(toBeProcessed >= 1) lenSubSequences = lenNumberArray / toBeProcessed; 

        // unblock workers
        if((statusDistributor = pthread_cond_signal(&waitForWorkAtribution)) != 0){
            errno = statusDistributor;                                                                  /* save error in errno */
            perror("error on signaling in waitForWorkAttribution");
            statusDistributor = EXIT_FAILURE;
            pthread_exit (&statusDistributor);
        }
    }

    // true if all sub-sequences have been processed
    if(toBeProcessed < 1){
        finished = 1;
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
 * \param workerId id of the worker
 * \param length length of the sub-sequence to be processed
 * \param startIndex index from which the sub-sequence starts
 * \param endIndex index at which the sub-sequence ends
 * \param iteration current iteration of the algorithm
 * 
 * \return the array of numbers to be process or NULL to signal the end of the program
*/
int* askForWorkload(int workerId, int *length, int *startIndex, int *endIndex, int *iteration){
    if ((statusWorkers[workerId] = pthread_mutex_lock(&accessCR)) != 0){								/* enter monitor */
		errno = statusWorkers[workerId];															/* save error in errno */
		perror("error on entering monitor(CF)");
		statusWorkers[workerId] = EXIT_FAILURE;
		pthread_exit(&statusWorkers[workerId]);
	}
	pthread_once(&init, initialization);                                                    /* internal data initialization */

    // block worker here while its activity status is equal to 1
    while(activeWorkers[workerId] == 1){
        printf("Worker %d is waiting for next iteration\n", workerId);
        if((statusWorkers[workerId] = pthread_cond_wait(&waitForWorkAtribution, &accessCR)) != 0){
            errno = statusWorkers[workerId];                                                                /* save error in errno */
            perror("error on waiting for waitForWorkAttribution");
            statusWorkers[workerId] = EXIT_FAILURE;
            pthread_exit(&statusWorkers[workerId]);
        }
    }

    // check to see whether or not it was chosen to work
    // ending the thread if it was not chosen
    if(activeWorkers[workerId] == -1){
        if ((statusWorkers[workerId] = pthread_cond_signal(&waitForWorkAtribution)) != 0){
            errno = statusWorkers[workerId];                         									/* save error in errno */
            perror ("error on signaling waitForWorkRequest");
            statusWorkers[workerId] = EXIT_FAILURE;
            pthread_exit(&statusWorkers[workerId]);
        }
        if ((statusWorkers[workerId] = pthread_mutex_unlock(&accessCR)) != 0){								
            errno = statusWorkers[workerId];															/* save error in errno */
            perror("error on entering monitor(CF)");
            statusWorkers[workerId] = EXIT_FAILURE;
            pthread_exit(&statusWorkers[workerId]);
        }

        return NULL;
    }

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

    // save the current request number, preventing incorrect 
    // and increment the number of work requests made
    int myCurrentRequestNumber = ++requestsMade;

    // wait for a workload attribution
    if((statusWorkers[workerId] = pthread_cond_wait(&waitForWorkAtribution, &accessCR)) != 0){
        errno = statusWorkers[workerId];                                                                /* save error in errno */
        perror("error on waiting for waitForWorkAttribution");
        statusWorkers[workerId] = EXIT_FAILURE;
        pthread_exit(&statusWorkers[workerId]);
    }

    workRequestsAttributed++;
    if ((statusWorkers[workerId] = pthread_cond_signal(&waitForWorkRequest)) != 0){
		errno = statusWorkers[workerId];                         									/* save error in errno */
		perror ("error on signaling waitForWorkRequest");
		statusWorkers[workerId] = EXIT_FAILURE;
		pthread_exit(&statusWorkers[workerId]);
	}

    //define length of the sub-sequence to sort
    *length = lenSubSequences;
    //define the index to the start of the sub-sequence
    *startIndex = (myCurrentRequestNumber - 1) * lenSubSequences;
    //define the index to the end of the sub-sequence
    *endIndex = (myCurrentRequestNumber * lenSubSequences) - 1;
    //save the iteration 
    *iteration = currentIteration;

    printf("Worker %d was attributed work. Len: %d. Start: %d. End: %d\n", workerId, *length, *startIndex, *endIndex);


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

/**
 * \brief Notify that the work is done
 * 
 * \param workerId id of the worker
*/
void workFinished(int workerId){
    if ((statusWorkers[workerId] = pthread_mutex_lock(&accessCR)) != 0){								/* enter monitor */
		errno = statusWorkers[workerId];															/* save error in errno */
		perror("error on entering monitor(CF)");
		statusWorkers[workerId] = EXIT_FAILURE;
		pthread_exit(&statusWorkers[workerId]);
	}
	pthread_once(&init, initialization);                                                    /* internal data initialization */

    // update status has completed work
    activeWorkers[workerId] = 1;

    // signal distributor that work is finished
    if((statusWorkers[workerId] = pthread_cond_signal(&waitForWorkCompletion)) != 0){
        errno = statusWorkers[workerId];                         									/* save error in errno */
		perror ("error on signaling waitForWorkCompletion");
		statusWorkers[workerId] = EXIT_FAILURE;
		pthread_exit(&statusWorkers[workerId]);
    }

    if ((statusWorkers[workerId] = pthread_mutex_unlock(&accessCR)) != 0){								/* exit monitor */
		errno = statusWorkers[workerId];															/* save error in errno */
		perror("error on entering monitor(CF)");
		statusWorkers[workerId] = EXIT_FAILURE;
		pthread_exit(&statusWorkers[workerId]);
	}
}

/**
 * \brief clean-up function. release memory and destroy mutex and conditional variables
*/
void cleanup(){
    free(numberArray);
    free(activeWorkers);
    pthread_mutex_destroy(&accessCR);
    pthread_cond_destroy(&waitForWorkAtribution);
    pthread_cond_destroy(&waitForWorkCompletion);
    pthread_cond_destroy(&waitForWorkRequest);
}