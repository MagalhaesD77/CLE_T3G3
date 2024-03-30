/**
 *  \file sharedRegion.h (implementation file)
 *
 *  \brief Problem name: Portuguese Text processing.
 *
 *  Synchronization based on monitors.
 *  Both threads and the monitor are implemented using the pthread library which enables the creation of a
 *  monitor of the Lampson / Redell type.
 *
 *  Data transfer region implemented as a monitor.
 * 
 *  Problem Data Structures.
 *     \li customFile
 *     \li workerData
 *
 *  Definition of the operations carried out by the workers / main threads:
 *     \li initializeCountings
 *     \li add_file
 *     \li joinResults
 *     \li printResults
 *     \li getData
 *     \li mutex_lock
 *     \li mutex_unlock
 *     \li add_thread_counts
 *
 *  \author Diogo Magalhães & Rafael Gil - March 2024
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <errno.h>

#include "sharedRegion.h"
#include "utils.h"

/** \brief worker threads return status array */
extern int *workerStatus;

/** \brief worker threads current file array */
int *workerFileStatus;

/** \brief max number of bytes per chunk */
extern int bufferSize;

/** \brief number of threads */
extern int nThreads;

/** \brief words parcial results */
int **wordsCount;

/** \brief multi consonant words parcial results */
int **multiConsWordsCount;

/** \brief mutex to access the current files index */
static pthread_mutex_t accessCR = PTHREAD_MUTEX_INITIALIZER;

/** \brief storage region */
struct customFile *files;

/** \brief current file index being processed */
int currFileIndex = 0;

/** \brief number of files to process */
int numFiles = 0;



/**
 *  \brief Join the counting results from each thread for each file.
 */

void joinResults()
{
    for (int i = 0; i < numFiles; i++)
    {
        // int totalWords = 0;
        // int totalMultiConsWords = 0;
        for (int j = 0; j < nThreads; j++)
        {
            files[i].numWords += wordsCount[i][j];
            files[i].numMultiConsWords += multiConsWordsCount[i][j];
        }
    }
}

/**
 *  \brief Print the results of the counting.
 */

void printResults()
{
    for (int i = 0; i < numFiles; i++)
    {
        printf("File name: %s\n", files[i].fileName);
        printf("Total number words = %d\n", files[i].numWords);
        printf("Total number of words with at least two instances of the same consonant = %d\n", files[i].numMultiConsWords);
        printf("\n");
    }
}

/**
 *  \brief Add the partial counting results from a thread for a file to the shared region.
 *
 *  \param workerId worker identification
 *  \param words number of words
 *  \param multiConsWords number of words with at least two instances of the same consonant
 */

void add_thread_counts(unsigned int workerId, int words, int multiConsWords)
{
    wordsCount[workerFileStatus[workerId]][workerId] += words;
    multiConsWordsCount[workerFileStatus[workerId]][workerId] += multiConsWords;

}

/**
 *  \brief Initialize the counting arrays for the threads in the shared region.
 */

void initializeCountings(){
    // Memory allocation for the wordsCount and multiConsWordsCount arrays
    wordsCount = (int **)malloc(numFiles * sizeof(int *));
    multiConsWordsCount = (int **)malloc(numFiles * sizeof(int *));

    for (int i = 0; i < numFiles; i++)
    {
        wordsCount[i] = (int *)malloc(nThreads * sizeof(int));
        multiConsWordsCount[i] = (int *)malloc(nThreads * sizeof(int));
    }

    // Memory allocation for the workerFileStatus array
    workerFileStatus = (int *)malloc(nThreads * sizeof(int));

}

/**
 *  \brief Get next buffer of data from the active file in the shared region.
 *
 *  \param workerId worker identification
 *  \param buffer buffer to store the data
 *  \param returnStatus return status
 */

void getData(unsigned int workerId, char *buffer, int *returnStatus, int *currentFileIndex)
{
    mutex_lock(&accessCR, workerId);

    workerFileStatus[workerId] = currFileIndex;

    if (currFileIndex >= numFiles)
    {
        mutex_unlock(&accessCR, workerId);
        *returnStatus = 1;
        return;
    }

    size_t bytes_read = fread(buffer, 1, bufferSize, files[currFileIndex].file);

    // if EOF reached there is no need to find the last outside word char
    if (!feof(files[currFileIndex].file)){

        int file_pointer_change = find_last_outside_word_char_position(buffer, bytes_read);

        if (file_pointer_change == -1){
            printf("Error: Found word bigger than the size of the buffer. Please increase the buffer size.\n");
            // Maybe keep reading the word (Not adding the letters to buffer until a outside word char is found) (Need to check if two consonant word to change the word in the buffer if needed)
            files[currFileIndex].failed = 1;
            currFileIndex++;
            mutex_unlock(&accessCR, workerId);
            *returnStatus = 2;
            return;
        }

        // change the file pointer to byte current_position - (BUFFER_SIZE - file_pointer_change)
        fseek(files[currFileIndex].file, - bufferSize + file_pointer_change + 1, SEEK_CUR);

        // Remove from buffer the last bytes after file_pointer_change (The outside word char is excluded from the buffer)
        memset(buffer + file_pointer_change, '\0', bufferSize - file_pointer_change + 1);

    }
    else{
        currFileIndex++;
    }

    mutex_unlock(&accessCR, workerId);

    *returnStatus = 0;
    return;

}

/**
 *  \brief Lock the mutex.
 *
 *  \param mutex mutex to be locked
 *  \param workerId worker identification
 */

void mutex_lock(pthread_mutex_t *mutex, unsigned int workerId){
    if ((workerStatus[workerId] = pthread_mutex_lock(mutex)) != 0) /* enter monitor */
    {
        errno = workerStatus[workerId]; /* save error in errno */
        perror("error on entering monitor(CF)");
        workerStatus[workerId] = EXIT_FAILURE;
        pthread_exit(&workerStatus[workerId]);
    }

}

/**
 *  \brief Unlock the mutex.
 *
 *  \param mutex mutex to be unlocked
 *  \param workerId worker identification
 */

void mutex_unlock(pthread_mutex_t *mutex, unsigned int workerId){
    if ((workerStatus[workerId] = pthread_mutex_unlock(mutex)) != 0) /* exit monitor */
    {
        errno = workerStatus[workerId]; /* save error in errno */
        perror("error on exiting monitor(CF)");
        workerStatus[workerId] = EXIT_FAILURE;
        pthread_exit(&workerStatus[workerId]);
    }

}

/**
 *  \brief Add a file to the shared region.
 *
 *  \param filename file to be added
 */

int add_file(char *fileName)
{
    // Open the file
    FILE *file = fopen(fileName, "r");
    if (file == NULL)
    {
        printf("Error: Could not open file %s\n", fileName);
        return -1;
    }

    // Allocate memory for the files array
    files = (struct customFile *)realloc(files, (numFiles + 1) * sizeof(struct customFile));

    // allocate memory for the file in the customFile array
    files[numFiles].fileName = fileName;
    files[numFiles].file = file;
    files[numFiles].numWords = 0;
    files[numFiles].numMultiConsWords = 0;
    files[numFiles].failed = 0;

    numFiles++;

    return 0;
}
