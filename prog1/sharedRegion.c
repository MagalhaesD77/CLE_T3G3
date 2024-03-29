/**
 *  \file sharedRegion.c (implementation file)
 *
 *  \brief Problem name: Portuguese Text processing.
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
 *  \author Diogo Magalhães & Rafael Gil - March 2024
 */

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <pthread.h>
#include <errno.h>

#include "sharedRegion.h"
#include "utils.h"

/** \brief worker threads return status array */
extern int *workerStatus;

/** \brief max number of bytes per chunk */
extern int bufferSize;

/** \brief number of threads */
extern int nThreads;

/** \brief words parcial results */
int **wordsCount;

/** \brief multi consonant words parcial results */
int **multiConsWordsCount;


// Real shared region variables
/** \brief mutex to access the current files index */
static pthread_mutex_t accessCR = PTHREAD_MUTEX_INITIALIZER;

/** \brief storage region */
struct customFile *files;

/** \brief current file index being processed */
int currFileIndex = 0;

/** \brief number of files to process */
int numFiles = 0;


void add_thread_counts(unsigned int workerId, int fileIndex, int words, int multiConsWords)
{
    wordsCount[fileIndex][workerId] += words;
    multiConsWordsCount[fileIndex][workerId] += multiConsWords;

}

int get_thread_words_count(unsigned int workerId, int fileIndex)
{
    return wordsCount[fileIndex][workerId];
}

int get_thread_multi_cons_words_count(unsigned int workerId, int fileIndex)
{
    return multiConsWordsCount[fileIndex][workerId];
}

void initializeCountings(){
    wordsCount = (int **)malloc(numFiles * sizeof(int *));
    multiConsWordsCount = (int **)malloc(numFiles * sizeof(int *));

    for (int i = 0; i < numFiles; i++)
    {
        wordsCount[i] = (int *)malloc(nThreads * sizeof(int));
        multiConsWordsCount[i] = (int *)malloc(nThreads * sizeof(int));
    }

}

void getData(unsigned int workerId, char *buffer, int *returnStatus, int *currentFileIndex)
{
    mutex_lock(&accessCR, workerId);

    *currentFileIndex = currFileIndex;

    if (currFileIndex >= numFiles)
    {
        mutex_unlock(&accessCR, workerId);
        *returnStatus = 1;
        return;
    }

    // printf("Initial buffer: %s\n", buffer);
    size_t bytes_read = fread(buffer, 1, bufferSize, files[currFileIndex].file);
    // printf("Buffer read: %s\n", buffer);

    // if EOF reached there is no need to find the last outside word char
    if (!feof(files[currFileIndex].file)){

        int file_pointer_change = find_last_outside_word_char_position(buffer, bytes_read);

        // printf("File pointer change: %d\n", file_pointer_change);

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


void mutex_lock(pthread_mutex_t *mutex, unsigned int workerId){
    if ((workerStatus[workerId] = pthread_mutex_lock(mutex)) != 0) /* enter monitor */
    {
        errno = workerStatus[workerId]; /* save error in errno */
        perror("error on entering monitor(CF)");
        workerStatus[workerId] = EXIT_FAILURE;
        pthread_exit(&workerStatus[workerId]);
    }

}

void mutex_unlock(pthread_mutex_t *mutex, unsigned int workerId){
    if ((workerStatus[workerId] = pthread_mutex_unlock(mutex)) != 0) /* exit monitor */
    {
        errno = workerStatus[workerId]; /* save error in errno */
        perror("error on exiting monitor(CF)");
        workerStatus[workerId] = EXIT_FAILURE;
        pthread_exit(&workerStatus[workerId]);
    }

}


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

void printFileNames()
{
    printf("%d files to process:\n", numFiles);
    // Print the file names without using numFiles
    for (int i = 0; i < numFiles; i++)
    {
        printf("%s\n", files[i].fileName);
    }
    
}

void printArguments()
{
    printf("Number of threads: %d\n", nThreads);
    printf("Buffer size: %d\n", bufferSize);
}





