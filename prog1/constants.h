/**
 *  \file ex1.h (interface file)
 *
 *  \brief Problem name: Portuguese Text processing.
 *
 *  Problem parameters.
 *
 *  \author Diogo Magalhães & Rafael Gil - March 2024
 */

#ifndef PROBCONST_H_
#define PROBCONST_H_

/* Generic parameters */

/** \brief default size of reading buffer */
#define BUFFER_SIZE_DEFAULT 8192;

/** \brief default number of worker threads */
#define N_THREADS_DEFAULT 4;


// Functions

/**
 *  \brief Argument Parser.
 *
 *  Operation carried out by the main thread.
 *
 *  \param argc number of command line arguments
 *  \param argv array of command line arguments
 */
void cli_parser(int argc, char *argv[]);

/**
 *  \brief Print Usage of the program.
 *
 *  Operation carried out by the main thread.
 *
 *  \param cmdName command name
 */
static void printUsage (char *cmdName);

#endif /* PROBCONST_H_ */


