/**
 *  \file synchronization.c (interface file)
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
 *     \li distributeWorkloads
 *     \li askForWorkloads
 *
 *  \author Rafael Gil & Diogo Magalhães - March 2024
 */

#ifndef SYNCHRONIZATION_H
#define SYNCHRONIZATION_H

/**
 * \brief Read file and populate data array
 * 
 * \param fileName name of the file to read
*/
extern void readFile(char *fileName);

/** 
 * \brief define the number of sub-sequences to be distributed 
 * 
 * \param numWorkers number of worker threads
*/
extern void defineSubSequences(int numWorkers);

/**
 * \brief Calculate the number of sub-sequences to attribute to the workers and distribute them
 * 
 * \return flag that signals the end of the program
*/
extern int distributeWorkloads();

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
extern int* askForWorkload(int workerId, int *length, int *startIndex, int *endIndex, int *iteration);


/**
 * \brief Notify that the work is done
 * 
 * \param workerId id of the worker
*/
extern void workFinished(int workerId);

/**
 * \brief checks if the number array is ordered in decreasing order
*/
extern void verifyIfSequenceIsOrdered();

/**
 * \brief clean-up function. release memory and destroy mutex and conditional variables
*/
extern void cleanup();

#endif /* SYNCHRONIZATION_H */
