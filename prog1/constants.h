/**
 *  \file constants.h (interface file)
 *
 *  \brief Problem name: Portuguese Text processing.
 * 
 *  Constants used in the program.
 *
 *  Problem main parameters.
 *     \li N_THREADS_DEFAULT
 *     \li BUFFER_SIZE_DEFAULT.
 *
 *  Definition of the initial operations carried out by the main / worker threads:
 *     \li cli_parser
 *     \li printUsage
 *     \li worker.
 *
 *  \author Diogo Magalhães & Rafael Gil - March 2024
 */

#ifndef PROBCONST_H_
#define PROBCONST_H_

/** \brief default size of reading buffer */
#define BUFFER_SIZE_DEFAULT 8192;

/** \brief default number of worker threads */
#define N_THREADS_DEFAULT 4;


/**
 *  \brief Argument Parser.
 *
 *  \param argc number of command line arguments
 *  \param argv array of command line arguments
 */

void cli_parser(int argc, char *argv[]);

/**
 *  \brief Print Usage of the program.
 *
 *  \param cmdName command name
 */

static void printUsage (char *cmdName);

/**
 *  \brief Worker Thread Function.
 *
 *  \param prodId producer identification
 *  \param val value to be stored
 */

static void *worker (void *id);

#endif /* PROBCONST_H_ */


