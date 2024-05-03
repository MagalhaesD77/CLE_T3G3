/**
 *  \file main.h (interface file)
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
 *  \author Diogo Magalhães & Rafael Gil - May 2024
 */

#ifndef PROBCONST_H_
#define PROBCONST_H_

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
    int opened;
    int failed;
};

struct workerToDispatcherMessage
{
    int numWords;
    int numMultiConsWords;
};


/** \brief default size of reading buffer */
#define BUFFER_SIZE_DEFAULT 8192;


void get_data(struct customFile *files, int numFiles, int *currFileIndex, char *buffer, int bufferSize);

/**
 *  \brief Argument Parser.
 *
 *  \param argc number of command line arguments
 *  \param argv array of command line arguments
 *  \param numFiles number of files
 *  \param files array of files
 *  \param bufferSize buffer size
 */

void cli_parser(int argc, char *argv[], int *numFiles, struct customFile **files, int *bufferSize);

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

void worker(int rank, int size, int bufferSize);
void dispatcher(int rank, int size, int numFiles, struct customFile *files, int bufferSize);

#endif /* PROBCONST_H_ */


