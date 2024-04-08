/**
 *  \file ex1.c (implementation file)
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

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <pthread.h>
#include <libgen.h>
#include <time.h>
#include <getopt.h>
#include <stdlib.h>

#include "constants.h"
#include "sharedRegion.h"
#include "utils.h"

/** \brief buffer size */
int bufferSize;

/** \brief number of threads */
int nThreads;

/** \brief number of files */
extern int numFiles;

/** \brief worker threads return status array */
int *workerStatus;



/** \brief array with codes of alphanumeric characters and underscore */
char alphanumeric_chars_underscore[] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_'};

/** \brief size of the alphanumeric characters and underscore array */
int alphanumeric_chars_underscore_array_size = sizeof(alphanumeric_chars_underscore)/sizeof(char);

/** \brief array with codes of consonants */
char consonants[] = {'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'z'};

/** \brief size of the consonants array */
int consonants_array_size = sizeof(consonants)/sizeof(char);

/** \brief array with codes of outside word characters */
char outside_word_chars[] = {0x20, 0x9, 0xD, 0xA, 0x2d, 0x22, 0x5b, 0x5d, 0x28, 0x29, 0x2e, 0x2c, 0x3a, 0x3b, 0x3f, 0x21};

/** \brief size of the outside word characters array */
int outside_word_array_size = sizeof(outside_word_chars)/sizeof(char);



/** \brief execution time measurement */
static double get_delta_time(void);



/**
 *  \brief Main thread.
 *
 *  Its role is creating the threads that will process the files and count the words
 *  and multi consonant words, and wait for them to finish to output the results.
 *
 *  \param argc number of words of the command line
 *  \param argv list of words of the command line
 *
 *  \return status of operation
 */

int main(int argc, char *argv[])
{
    init_c3_bytes_to_char();
    init_e2_2_bytes_to_char();

    // 1. Parse the command line arguments
    cli_parser(argc, argv);
    
    (void) get_delta_time ();
    // 2. to create the worker threads
    pthread_t workers[nThreads];
    for (int i = 0; i < nThreads; i++)
    {
        struct workerData *data = (struct workerData *)malloc(sizeof(struct workerData));
        data->id = i;
        if (pthread_create(&workers[i], NULL, worker, (void *)data) != 0)
        {
            fprintf(stderr, "Error: Could not create worker thread %d\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // 3. Wait for the worker threads to finish
    for (int i = 0; i < nThreads; i++)
    {
        void *status;
        if (pthread_join(workers[i], &status) != 0)
        {
            fprintf(stderr, "Error: Could not join worker thread %d\n", i);
            exit(EXIT_FAILURE);
        }
    }

    double elapsed_time = get_delta_time ();

    // 4. Join the results from the worker threads for each file and Print the results
    joinResults();
    printResults();

    // free resources
    cleanup();

    printf ("Elapsed time = %.6f s\n", elapsed_time);

    return 0;
}



/**
 *  \brief Worker Thread Function.
 *
 *  \param prodId producer identification
 *  \param val value to be stored
 */

static void *worker (void *data)
{
    struct workerData *workerData = (struct workerData *)data;
    int id = workerData->id;
    
    char buffer[bufferSize];
    int returnStatus;
    char word_chars[sizeof(consonants)/sizeof(char)];

    while(1){

        // Clear the buffer
        memset(buffer, '\0', sizeof(buffer));

        // Get the data from the shared region
        getData(id, buffer, &returnStatus);
        // returnStatus: 0 - Success, 1 - No more data, 2 - Error (Retry again)
        if (returnStatus == 0){
            // Process the data
            char c;
            int index = 0;
            int count_word = 0;
            int count_two_consoant_words = 0;
            int two_consoant_bool = 0;
            memset(word_chars, '\0', sizeof(word_chars));   // Clear the array
            int inside_word = 0;
            while ((c = read_next_char_from_array(buffer, &index, bufferSize)) != EOF)
            {
                c = tolower(c);

                // Check if char c is in outside_word_chars array
                if (contains(outside_word_chars, outside_word_array_size, c))
                {
                    if (inside_word != 0){
                        inside_word = 0;
                        two_consoant_bool = 0;
                        memset(word_chars, '\0', sizeof(word_chars));   // Clear the array
                    }
                }
                else
                {   
                    // If it is a second consonant or if it is not an alphanumeric char or underscore, ignore it
                    if (two_consoant_bool != 0 || contains(alphanumeric_chars_underscore, alphanumeric_chars_underscore_array_size, c) == 0){
                        continue;
                    }

                    // If outside word, change to inside word
                    if (inside_word == 0)
                    {
                        count_word++;
                        inside_word = 1;
                    }

                    // Check if consonant
                    if (contains(consonants, consonants_array_size, c)){
                        // if consonant, check if it is the second one
                        if (contains(word_chars, sizeof(word_chars)/sizeof(char), c)){
                            count_two_consoant_words++;
                            two_consoant_bool = 1;
                        }
                        else{
                            // add the caracter to the array
                            add(word_chars, sizeof(word_chars)/sizeof(char), c);
                        }
                    }

                }

            }

            add_thread_counts(id, count_word, count_two_consoant_words);
            continue;
        }
        if (returnStatus == 1){
            break;
        }
        if (returnStatus == 2){
            continue;
        }

    }

    workerStatus[id] = EXIT_SUCCESS;
    pthread_exit (&workerStatus[id]);
}

/**
 *  \brief Argument Parser.
 *
 *  \param argc number of command line arguments
 *  \param argv array of command line arguments
 */

void cli_parser(int argc, char *argv[])
{
    int t_flag = 0;
    int b_flag = 0;
    int opt;
    while ((opt = getopt(argc, argv, "f:t:b:h")) != -1) {
        switch (opt) {
        case 'f':
            // get the text file names by processing the command line and storing them in the shared region (This work with multiple arguments for example using dataset1/text*.txt)
            for (int i = optind - 1; i < argc && argv[i][0] != '-'; ++i) {
                // Open the file
                if (add_file(argv[i]) == -1)
                {
                    exit(EXIT_FAILURE);
                }
            }
            break;
        case 't':
            if(t_flag == 1){
                fprintf(stderr, "Error: -t flag can only be used once\n");
                exit(EXIT_FAILURE);
            }
            if (atoi(optarg) <= 0)
            {
                fprintf(stderr, "Error: -t flag must be a positive integer\n");
                exit(EXIT_FAILURE);
            }
            t_flag = 1;
            nThreads = atoi(optarg);
            break;
        case 'b':
            if(b_flag == 1){
                fprintf(stderr, "Error: -b flag can only be used once\n");
                exit(EXIT_FAILURE);
            }
            if (atoi(optarg) <= 0)
            {
                fprintf(stderr, "Error: -b flag must be a positive integer\n");
                exit(EXIT_FAILURE);
            }
            b_flag = 1;
            bufferSize = atoi(optarg);
            break;
        case 'h':
            printUsage(basename (argv[0]));
            exit(EXIT_SUCCESS);
        default:
            printUsage(basename (argv[0]));
            exit(EXIT_FAILURE);
        }
    }

    if (numFiles <= 0)
    {
        printf("No file provided\n");
        printUsage(basename (argv[0]));
        exit(EXIT_FAILURE);
    }

    // Check if the -t flag was used
    if(t_flag == 0){
        nThreads = N_THREADS_DEFAULT;
        printf("-t not defined. Using default value of %d threads\n\n", nThreads);
    }

    // Check if the -b flag was used
    if(b_flag == 0){
        bufferSize = BUFFER_SIZE_DEFAULT;
        printf("-b not defined. Using default value of %d buffer size\n\n", bufferSize);
    }

    if ((workerStatus = malloc (nThreads * sizeof (int))) == NULL){
        fprintf (stderr, "error on allocating space to the return status arrays of producer / consumer threads\n\n");
        exit (EXIT_FAILURE);
    }

    initializeCountings();
    
}

/**
 *  \brief Print Usage of the program.
 *
 *  \param cmdName command name
 */

static void printUsage (char *cmdName)
{
  fprintf (stderr, "\nSynopsis: %s [OPTIONS]\n"
           "  OPTIONS:\n"
           "  -f fileName   --- name of the file or multiple files to be processed\n"
           "  -t nThreads   --- set the number of threads to be created (default: 4)\n"
           "  -b bufferSize --- set the buffer size (default: 8192)\n"
           "  -h            --- print this help\n", cmdName);
}

/**
 *  \brief Get the process time that has elapsed since last call of this time.
 *
 *  \return process elapsed time
 */

static double get_delta_time(void)
{
  static struct timespec t0, t1;

  t0 = t1;
  if(clock_gettime (CLOCK_MONOTONIC, &t1) != 0)
  {
    perror ("clock_gettime");
    exit(1);
  }
  return (double) (t1.tv_sec - t0.tv_sec) + 1.0e-9 * (double) (t1.tv_nsec - t0.tv_nsec);
}
