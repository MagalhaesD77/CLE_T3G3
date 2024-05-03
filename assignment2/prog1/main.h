/**
 *  \file main.h (interface file)
 *
 *  \brief Problem name: Portuguese Text processing.
 * 
 *  Constants used in the program.
 *
 *  Problem main parameters.
 *     \li BUFFER_SIZE_DEFAULT.
 *
 *  Definition of the initial operations carried out by the dispatcher / worker processes:
 *     \li cli_parser
 *     \li printUsage
 *     \li dispatcher
 *     \li worker
 *     \li get_delta_time
 *     \li printResults.
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


/**
 *  \brief Worker to Dispatcher Message.
 *
 *  Message structure from worker to dispatcher.
 */

struct workerToDispatcherMessage
{
    int numWords;
    int numMultiConsWords;
};


/** \brief default size of reading buffer */
#define BUFFER_SIZE_DEFAULT 8192;


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
 * \brief Dispatcher Process Function.
 * 
 * \param rank process rank
 * \param size number of processes
 * \param numFiles number of files
 * \param files array of files
 * \param bufferSize buffer size
 */

void dispatcher(int rank, int size, int numFiles, struct customFile *files, int bufferSize);


/**
 * \brief Worker Process Function.
 * 
 * \param rank process rank
 * \param size number of processes
 * \param bufferSize buffer size
 */

void worker(int rank, int size, int bufferSize);


/**
 *  \brief Get next buffer of data from the active file in the shared region.
 *
 *  \param files array of files
 *  \param numFiles number of files
 *  \param currFileIndex index of the current file
 *  \param buffer buffer to store the data
 *  \param bufferSize buffer size
 */

void get_data(struct customFile *files, int numFiles, int *currFileIndex, char *buffer, int bufferSize);


/**
 *  \brief Print the results of the counting.
 *  
 *  \param files array of files
 *  \param numFiles number of files
 */

void printResults(struct customFile *files, int numFiles);

#endif /* PROBCONST_H_ */

