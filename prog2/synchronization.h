/**
 *  \file fifo.h (interface file)
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

#ifndef SYNCHRONIZATION_H
#define SYNCHRONIZATION_H

/**
 * \brief Read file and populate data array
 * 
 * \param fileName name of the file to read
*/
extern void readFile(char *fileName);



#endif /* SYNCHRONIZATION_H */
