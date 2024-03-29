#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <pthread.h>
#include <libgen.h>
#include <math.h>
#include <time.h>
#include <getopt.h>

#include <stdlib.h> // For exit() function, might be removed later

#include "constants.h"
#include "sharedRegion.h"
#include "utils.h"


// variables that should be passed by argument
int bufferSize;
int nThreads;
extern int numFiles;
int *workerStatus;


extern char alphanumeric_chars_underscore;
extern int alphanumeric_chars_underscore_array_size;
extern char consonants;
extern int consonants_array_size;
extern char outside_word_chars;
extern int outside_word_array_size;
extern char c3_bytes_to_char;
extern char e2_2_bytes_to_char;


/** \brief worker life cycle routine */
static void *worker (void *id);

int main(int argc, char *argv[])
{
    init_c3_bytes_to_char();
    init_e2_2_bytes_to_char();

    // 1. Parse the command line arguments
    cli_parser(argc, argv);

        // Print the file names
        // printFileNames();
        // printArguments();
    
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

    // 4. Join the results from the worker threads for each file and Print the results
    for (int i = 0; i < numFiles; i++)
    {
        int totalWords = 0;
        int totalMultiConsWords = 0;
        for (int j = 0; j < nThreads; j++)
        {
            totalWords += get_thread_words_count(j, i);
            totalMultiConsWords += get_thread_multi_cons_words_count(j, i);
        }

        printf("File name: %d\nTotal number words = %d\nTotal number of words with at least two instances of the same consonant = %d\n", i, totalWords, totalMultiConsWords);
    }


    return 0;
}

static void *worker (void *data)
{
    struct workerData *workerData = (struct workerData *)data;
    int id = workerData->id;
    // printf("Worker %d\n", id);
    
    char buffer[bufferSize];
    int returnStatus, currentFile;
    char word_chars[consonants_array_size];

    while(true){

        // Clear the buffer
        memset(buffer, '\0', sizeof(buffer));

        // Get the data from the shared region
        getData(id, buffer, &returnStatus, &currentFile);
        // returnStatus: 0 - Success, 1 - No more data, 2 - Error (Retry again)
        if (returnStatus == 0){
            // Process the data
            
            // char read_next_char_from_array(const char *array, int *index, int array_size);
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
                if (contains(&outside_word_chars, outside_word_array_size, c))
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
                    if (two_consoant_bool != 0 || contains(&alphanumeric_chars_underscore, alphanumeric_chars_underscore_array_size, c) == 0){
                        continue;
                    }

                    // If outside word, change to inside word
                    if (inside_word == 0)
                    {
                        count_word++;
                        inside_word = 1;
                    }

                    // Check if consonant
                    if (contains(&consonants, consonants_array_size, c)){
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

            add_thread_counts(id, currentFile, count_word, count_two_consoant_words);
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
            t_flag = 1;
            nThreads = atoi(optarg);
            break;
        case 'b':
            if(b_flag == 1){
                fprintf(stderr, "Error: -b flag can only be used once\n");
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


static void printUsage (char *cmdName)
{
  fprintf (stderr, "\nSynopsis: %s [OPTIONS]\n"
           "  OPTIONS:\n"
           "  -f fileName   --- name of the file or multiple files to be processed\n"
           "  -t nThreads   --- set the number of threads to be created (default: 4)\n"
           "  -b bufferSize --- set the buffer size (default: 8192)\n"
           "  -h            --- print this help\n", cmdName);
}