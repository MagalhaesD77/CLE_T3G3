/**
 *  \file sharedRegion.h (interface file)
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

#ifndef SHAREDREGION_H
#define SHAREDREGION_H


/**
 *  \brief Custom file structure.
 *
 *   File structure with word count and multi consonant word count.
 */

struct customFile
{
    FILE *file;
    char *fileName;
    int numWords;
    int numMultiConsWords;
    int failed;
};

/**
 *  \brief Worker data structure.
 *
 *   structure to store any information needed by the worker threads.
 */

struct workerData
{
    int id;
};

/**
 *  \brief Initialize the counting arrays for the threads in the shared region.
 */

void initializeCountings();

/**
 *  \brief Add a file to the shared region.
 *
 *  \param filename file to be added
 */

int add_file(char *fileName);

/**
 *  \brief Join the counting results from each thread for each file.
 */

void joinResults();

/**
 *  \brief Print the results of the counting.
 */

void printResults();

/**
 *  \brief Get next buffer of data from the active file in the shared region.
 *
 *  \param workerId worker identification
 *  \param buffer buffer to store the data
 *  \param returnStatus return status
 */

void getData(unsigned int workerId, char *buffer, int *returnStatus);

/**
 *  \brief Lock the mutex.
 *
 *  \param mutex mutex to be locked
 *  \param workerId worker identification
 */

void mutex_lock(pthread_mutex_t *mutex, unsigned int workerId);

/**
 *  \brief Unlock the mutex.
 *
 *  \param mutex mutex to be unlocked
 *  \param workerId worker identification
 */

void mutex_unlock(pthread_mutex_t *mutex, unsigned int workerId);

/**
 *  \brief Add the partial counting results from a thread for a file to the shared region.
 *
 *  \param workerId worker identification
 *  \param words number of words
 *  \param multiConsWords number of words with at least two instances of the same consonant
 */

void add_thread_counts(unsigned int workerId, int words, int multiConsWords);

#endif /* SHAREDREGION_H */
