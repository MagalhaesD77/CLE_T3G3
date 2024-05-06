/**
 *  \file main.c (implementation file)
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

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <mpi.h>
#include <libgen.h>
#include <time.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdbool.h>

#include "main.h"
#include "utils.h"

/** \brief execution time measurement */
static double get_delta_time(void);


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


/**
 *  \brief Main.
 *
 *  Its role is creating the processes that will process the files and count the words
 *  and multi consonant words, and wait for them to finish to output the results.
 *
 *  \param argc number of words of the command line
 *  \param argv list of words of the command line
 *
 *  \return status of operation
 */

int main(int argc, char *argv[]){

  int rank, size;
  int bufferSize;

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);

  // We will have at least 2 processes (one dispatcher and atlist one worker)
  if (size < 2){
    if (rank == 0) printf ("Too few processes!\n");

    MPI_Finalize ();
    return EXIT_FAILURE;
  }

  if (rank == 0){
    // Dispatcher
    int numFiles = 0;
    static struct customFile *files;
    
    // Parse the command line arguments
    cli_parser(argc, argv, &numFiles, &files, &bufferSize);

    /* Tell each worker the maximum number of bytes each chunk will have so they can initialize the buffer */
    MPI_Bcast(&bufferSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    (void) get_delta_time (); // Initialize the time measurement

    // Run the dispatcher workflow
    dispatcher(rank, size, numFiles, files, bufferSize);

    double elapsed_time = get_delta_time ();  // Get the elapsed time

    printResults(files, numFiles);
    printf("Dispatcher: Elapsed time = %.6f s\n", elapsed_time);
  }
  else{
    // Worker
    MPI_Bcast(&bufferSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    worker(rank, size, bufferSize);
  }

  MPI_Finalize ();
  return EXIT_SUCCESS;

}


/**
 * \brief Dispatcher Process Function.
 * 
 * \param rank process rank
 * \param size number of processes
 * \param numFiles number of files
 * \param files array of files
 * \param bufferSize buffer size
 */

void dispatcher(int rank, int size, int numFiles, struct customFile *files, int bufferSize){
  
  int currFileIndex = 0;
  // struct workerToDispatcherMessage *recData[size-1]; // 0 - Need work, 1 - Results
  struct workerToDispatcherMessage **recData = (struct workerToDispatcherMessage **)malloc((size-1) * sizeof(struct workerToDispatcherMessage *));
  char **buffer = (char **)malloc((size-1) * sizeof(char *)); // Tag 0 - Work, Tag 1 - Stop
  bool allMsgRec, recVal, msgRec[size-1];
  MPI_Request reqSnd[size-1], reqRec[size-1];
  int workerFileStatus[size-1];

  // Allocate memory for the worker messages
  for (int i = 0; i < size-1; i++){
    // recData[i] = malloc(sizeof(struct workerToDispatcherMessage));
    recData[i] = (struct workerToDispatcherMessage *)malloc(sizeof(struct workerToDispatcherMessage));
  }

  // Allocate memory for the buffer
  for (int i = 0; i < size-1; i++){
    buffer[i] = (char *)malloc(bufferSize * sizeof(char));
  }

  // Initialize the active workers
  int activeWorkers[size-1];
  int activeWorkersCount = size-1;
  for (int i = 0; i < size-1; i++){
    activeWorkers[i] = 1;
  }

  while (activeWorkersCount > 0){

    for (int i = (rank + 1) % size; i < size; i++){
      if (activeWorkers[i-1] == 0)
        continue;

      // clear buffer
      memset(buffer[i-1], '\0', bufferSize);

      // Get data to send to worker
      workerFileStatus[i-1] = currFileIndex;
      get_data(files, numFiles, &currFileIndex, buffer[i-1], bufferSize);

      if (buffer[i-1][0] == '\0'){
        activeWorkers[i-1] = 0;
        activeWorkersCount--;
      }

      MPI_Isend(buffer[i-1], bufferSize, MPI_CHAR, i, 0, MPI_COMM_WORLD, &reqSnd[i-1]);
    }

    // Receive messages from workers in a non-blocking manner
    for (int i = (rank + 1) % size; i < size; i++){
      if (activeWorkers[i-1] == 0)
        continue;
        
      MPI_Irecv ((struct workerToDispatcherMessage *) recData[i-1], sizeof (struct workerToDispatcherMessage), MPI_BYTE, i, 0, MPI_COMM_WORLD, &reqRec[i-1]);
      msgRec[i-1] = false;
    }
    do{ // Wait for all messages to be received
      allMsgRec = true;
      for (int i = (rank + 1) % size; i < size; i++){
        if (activeWorkers[i-1] == 0)
          continue;

        if (!msgRec[i-1]){
          recVal = false;
          MPI_Test(&reqRec[i-1], (int *) &recVal, MPI_STATUS_IGNORE);
          if (recVal){
            files[workerFileStatus[i-1]].numWords += recData[i-1]->numWords;
            files[workerFileStatus[i-1]].numMultiConsWords += recData[i-1]->numMultiConsWords;

            msgRec[i-1] = true;
          }
          else
            allMsgRec = false;
        }
      }
    } while (!allMsgRec);
  }

}


/**
 * \brief Worker Process Function.
 * 
 * \param rank process rank
 * \param size number of processes
 * \param bufferSize buffer size
 */

void worker(int rank, int size, int bufferSize){

  init_c3_bytes_to_char();
  init_e2_2_bytes_to_char();

  struct workerToDispatcherMessage sndData = {0}; // 0 - Need work, 1 - Results
  char recData[bufferSize];
  MPI_Status status;
  char word_chars[consonants_array_size];

  while(true){

    MPI_Recv ((char *) &recData, bufferSize, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);

    if (recData[0] == '\0'){
      // No more work to do
      break;
    }

    // Process the data
    char c;
    int index = 0;
    int count_word = 0;
    int count_two_consoant_words = 0;
    int two_consoant_bool = 0;
    memset(word_chars, '\0', sizeof(word_chars));   // Clear the array
    int inside_word = 0;
    while ((c = read_next_char_from_array(recData, &index, bufferSize)) != EOF)
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

    // sndData.msgType = rank;
    sndData.numWords = count_word;
    sndData.numMultiConsWords = count_two_consoant_words;

    MPI_Send ((struct workerToDispatcherMessage *) &sndData, sizeof (struct workerToDispatcherMessage), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    
  }

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


/**
 *  \brief Argument Parser.
 *
 *  \param argc number of command line arguments
 *  \param argv array of command line arguments
 *  \param numFiles number of files
 *  \param files array of files
 *  \param bufferSize buffer size
 */

void cli_parser(int argc, char *argv[], int *numFiles, struct customFile **files, int *bufferSize)
{
  int b_flag = 0;
  int opt;
  while ((opt = getopt(argc, argv, "f:t:b:h")) != -1) {
    switch (opt) {
    case 'f':
      // get the text file names by processing the command line and storing them in the shared region (This work with multiple arguments for example using dataset1/text*.txt)
      for (int i = optind - 1; i < argc && argv[i][0] != '-'; ++i) {
        
        // Open the file
        FILE *file = fopen(argv[i], "r");

        // Check if the file was opened successfully
        if (file == NULL)
        {
          fprintf(stderr, "Error: Could not open file %s. Ignoring it\n", argv[i]);
          continue;
        }

        // Allocate memory for the files array
        *files = (struct customFile *)realloc(*files, (*numFiles + 1) * sizeof(struct customFile));

        // Allocate memory for the file in the customFile array
        (*files)[*numFiles].fileName = argv[i];
        (*files)[*numFiles].file = file;
        (*files)[*numFiles].numWords = 0;
        (*files)[*numFiles].numMultiConsWords = 0;
        (*files)[*numFiles].failed = 0;

        (*numFiles)++;
      }
      break;

    case 'b':
      // Only allow the -b flag to be used once
      if(b_flag == 1){
        fprintf(stderr, "Error: -b flag can only be used once\n");
        exit(EXIT_FAILURE);
      }

      // Check if the buffer size is a valid positive integer
      if (atoi(optarg) <= 0)
      {
        fprintf(stderr, "Error: -b flag must be a positive integer\n");
        exit(EXIT_FAILURE);
      }

      // Set the buffer size
      b_flag = 1;
      (*bufferSize) = atoi(optarg);
      break;

    default:
      printUsage(basename (argv[0]));
      exit(EXIT_FAILURE);
    }
  }

  if (*numFiles <= 0)
  {
      printf("No valid files provided\n");
      printUsage(basename (argv[0]));
      exit(EXIT_FAILURE);
  }

  // Check if the -b flag was used
  if(b_flag == 0){
      (*bufferSize) = BUFFER_SIZE_DEFAULT;
      printf("-b not defined. Using default value of %d buffer size\n\n", *bufferSize);
  }
    
}


/**
 *  \brief Get next buffer of data from the active file in the shared region.
 *
 *  \param files array of files
 *  \param numFiles number of files
 *  \param currFileIndex index of the current file
 *  \param buffer buffer to store the data
 *  \param bufferSize buffer size
 */

void get_data(struct customFile *files, int numFiles, int *currFileIndex, char *buffer, int bufferSize)
{
  if ((*currFileIndex) >= numFiles){
    // No more files to process
    return;
  }

  // Read buffer from file
  size_t bytes_read = fread(buffer, 1, bufferSize, files[(*currFileIndex)].file);

  // if EOF reached there is no need to find the last outside word char (which is the else part of the if statement)
  if (!feof(files[(*currFileIndex)].file)){

    int file_pointer_change = find_last_outside_word_char_position(buffer, bytes_read);

    if (file_pointer_change == -1){ // Failed to analyze the file since bufferSize is too small
      files[(*currFileIndex)].failed = 1;
      (*currFileIndex)++;
      return;
    }

    // change the file pointer to byte current_position - (BUFFER_SIZE - file_pointer_change)
    fseek(files[(*currFileIndex)].file, - bufferSize + file_pointer_change + 1, SEEK_CUR);

    // Remove from buffer the last bytes after file_pointer_change (The outside word char is excluded from the buffer)
    memset(buffer + file_pointer_change, '\0', bufferSize - file_pointer_change + 1);

  }
  else{
    (*currFileIndex)++;
  }
  
  return;

}


/**
 *  \brief Print the results of the counting.
 *  
 *  \param files array of files
 *  \param numFiles number of files
 */

void printResults(struct customFile *files, int numFiles)
{
    for (int i = 0; i < numFiles; i++)
    {
        printf("File name: %s\n", files[i].fileName);
        if (files[i].failed){
            printf("Error: File could not be processed\n");
            printf("\n");
            continue;
        }
        printf("Total number words = %d\n", files[i].numWords);
        printf("Total number of words with at least two instances of the same consonant = %d\n", files[i].numMultiConsWords);
        printf("\n");
    }
}


