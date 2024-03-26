#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

#include "constants.h"
#include "synchronization.h"

char *fileName;                 // name of the file to read 
int nThreads;                   // number of worker threads

int statusDistributor;          // return status of distributor thread
int *statusWorkers;             // return status of worker threads



void *distributor(void *data);
void *worker(void *data);
static double get_delta_time(void);
void imperativeBitonicSort(int* array, int N, int startIndex, int endIndex);


int main(int argc, char *argv[]){
    // check if there were arguments passed
    if (argc < 3)
    {
        printf("Provide the name of the file first and then the number of threads to use\n");
        exit(EXIT_FAILURE);
    }  

    // save the name of the file to read
    fileName = argv[1];

    // process command line arguments in order to get the number of threads
    if((nThreads = (int) atoi (argv[2])) <= 0){
        fprintf(stderr, "Error getting the number of threads\nPass them as a cmd param or check the order of the params\n");
        char inp[2];
        do
        {
            printf("Use the default of 4 threads? (y --> Yes / n --> No)");
            if(fgets(inp, sizeof(inp), stdin) == NULL){
                fprintf(stderr, "Error reading input\n");
                exit(EXIT_FAILURE);
            }
        } while (inp[0] != 'y' && inp[0] != 'n');
        
		if(inp[0] == 'n'){
            printf("Exiting program...\n");
            exit(EXIT_FAILURE);
        }

        nThreads = NUM_THREADS_DEFAULT;
    }

    if(nThreads != 1 && nThreads != 2 && nThreads != 4 && nThreads != 8){
        fprintf(stderr, "Invalid number of threads\n");
		char inp[2];
        do
        {
            printf("Use the default of 4 threads? (y --> Yes / n --> No)");
            if(fgets(inp, sizeof(inp), stdin) == NULL){
                fprintf(stderr, "Error reading input\n");
                exit(EXIT_FAILURE);
            }
        } while (inp[0] != 'y' && inp[0] != 'n');
        
		if(inp[0] == 'n'){
            printf("Exiting program...\n");
            exit(EXIT_FAILURE);
        }

        nThreads = NUM_THREADS_DEFAULT;
    }

    pthread_t tIdDistributor;       // distributor internal thread id array
    pthread_t *tIdWorkers;          // workers internal thread id array
    unsigned int *workersId;        // distributor application defined thread id array
    int *pStatus;                   // pointer to execution status

    // alocate memory for workers arrays
    if((tIdWorkers = malloc(nThreads * sizeof(pthread_t))) == NULL 
        || (workersId = malloc(nThreads * sizeof(unsigned int))) == NULL
        || (statusWorkers = malloc(nThreads * sizeof(int))) == NULL)
    {
        fprintf(stderr, "Error allocating memory for workers arrays\n");
		exit(EXIT_FAILURE);
    }

    // attribute an index for each worker
    for (int i = 0; i <= nThreads; i++)
    {
        workersId[i] = i;
    }

    // start timer
    (void) get_delta_time();

    // initialise distributor thread
    if(pthread_create(&tIdDistributor, NULL, distributor, NULL) != 0){
        printf("Failed creating distributor thread\n");
        exit(EXIT_FAILURE);
    }

    // initialise worker threads
    for (int i = 0; i < nThreads; i++){
		if (pthread_create(&tIdWorkers[i], NULL, worker, &workersId[i]) != 0)
		{
			printf("Failed creating worker %u thread\n", i);
			exit(EXIT_FAILURE);
		}
	}

    // finalise distributor thread
    if(pthread_join(tIdDistributor, (void *) &pStatus) != 0){
        printf("Failed waiting for distributor thread\n");
        exit(EXIT_FAILURE);
	}
	printf("thread distributor has terminated: ");
	printf("its status was %d\n", *pStatus);

    // finalise worker threads
    for (int i = 0; i < nThreads; i++){
		if (pthread_join(tIdWorkers[i], (void *) &pStatus) != 0)
		{
			printf("Failed waiting for worker %u thread\n", i);
			exit(EXIT_FAILURE);
		}
		printf("Worker thread %u has finished: ", i);
		printf("its status was %d\n", *pStatus);
	}

    //check if array is correctly ordered
    verifyIfSequenceIsOrdered();

    // free resources
    cleanup();

    // finish timer
    printf ("\nElapsed time = %.6f s\n", get_delta_time ());

    exit (EXIT_SUCCESS);
}

/**
 *  \brief Definition of distributor thread.
 *
 *  Its role is to simulate the life cycle of a distributor.
 *
 *  \param par pointer to application defined worker identification
 */
void *distributor(void *data){
    readFile(fileName);

    defineSubSequences(nThreads);
    
    while (distributeWorkloads() == 0);

    statusDistributor = EXIT_SUCCESS;
	pthread_exit(&statusDistributor);
}

/**
 *  \brief Definition of distributor thread.
 *
 *  Its role is to simulate the life cycle of a distributor.
 *
 *  \param par pointer to application defined worker identification
 */
void *worker(void *data){
    unsigned int id = *((unsigned int *) data);

    int *numberArray;
    int lenSubSequence;
    int startIndex;
    int endIndex;
    int update = 0;

    while ((numberArray = askForWorkload(id, &lenSubSequence, &startIndex, &endIndex, &update)) != NULL)
    {
        if(update == 1){
            update = 0;
            continue;
        }

        imperativeBitonicSort(numberArray, lenSubSequence, startIndex, endIndex);
        workFinished(id);
    }
    
    printf("Worker %d has accomplished its functions. Will be terminated...\n", id);

    statusWorkers[id] = EXIT_SUCCESS;
	pthread_exit(&statusWorkers[id]);
}

/**
 *  \brief implementation of the imperitive bitonic sort - descending order.
 *
 *  \param array array of numbers to be sorted
 *  \param N length of the array
 *  \param startIndex index where sub-sequence starts
 *  \param endIndex index where sub-sequence ends
 */
void imperativeBitonicSort(int* array, int N, int startIndex, int endIndex){
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
 *  \brief Get the process time that has elapsed since last call of this time.
 *
 *  \return process elapsed time
 */
static double get_delta_time(void)
{
  static struct timespec t0, t1;

  t0 = t1;
  if(clock_gettime (CLOCK_MONOTONIC, &t1) != 0)
  {
    perror ("clock_gettime");
    exit(1);
  }
  return (double) (t1.tv_sec - t0.tv_sec) + 1.0e-9 * (double) (t1.tv_nsec - t0.tv_nsec);
}