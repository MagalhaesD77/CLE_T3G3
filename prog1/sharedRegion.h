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
 *  Definition of the operations carried out by the producers / consumers:
 *     \li putVal
 *     \li getVal.
 *
 *  \author Diogo Magalhães & Rafael Gil - March 2024
 */

#ifndef SHAREDREGION_H
#define SHAREDREGION_H

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

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

struct workerData
{
    int id;
};

void initializeCountings();
int add_file(char *fileName);

void printFileNames();
void printArguments();
void getData(unsigned int workerId, char *buffer, int *returnStatus, int *currentFileIndex);
void mutex_lock(pthread_mutex_t *mutex, unsigned int workerId);
void mutex_unlock(pthread_mutex_t *mutex, unsigned int workerId);

void add_thread_counts(unsigned int workerId, int fileIndex, int words, int multiConsWords);
int get_thread_words_count(unsigned int workerId, int fileIndex);
int get_thread_multi_cons_words_count(unsigned int workerId, int fileIndex);

#endif /* SHAREDREGION_H */


