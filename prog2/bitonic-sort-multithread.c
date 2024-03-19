#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

#define NUM_THREADS_DEFAULT 4

pthread_t distributorThread;        // distributor thread
pthread_t *threadWorkers;          // workers application defined thread array
int *statusWorkers;                // return status of worker threads

int *numArray;                  // array to store numbers read from file
char *fileName;                 // name of the file to read 
int numThreads;                 // number of threads to use

int processIteraction;          // current iteraction of the bitonic process

void *distributor(void *data);
void *worker(void *data);
void print_int_array(int *arr, int size);


int main(int argc, char *argv[]){
    // check if there were arguments passed
    if (argc < 2)
    {
        printf("No file provided\n");
        return 1;
    }

    fileName = argv[1];

    int opt;
    do{
        switch ((opt = getopt(argc, argv, "t")))
        {
        case 't':
            int value = atoi(optarg);

            if (value != 1 || value != 2 || value != 4|| value != 8){
                printf("Invalid number of threads - will use the default number of threads\n");
                numThreads = NUM_THREADS_DEFAULT;
                break;
            }
            numThreads = value;
            break;
        case '?':
            printf("Invalid usage\n");
            exit(1);
        case -1:
            break;       
        default:
            break;
        }
    }while(opt != -1);

    if(pthread_create(&distributorThread, NULL, distributor, NULL) != 0){
        printf("Failed creating distributor thread\n");
        exit(1);
    }

    if(pthread_join(distributorThread, NULL) != 0){
        printf("Error on waiting thread\n");
        exit(1);
    }

    free(numArray);

}

void *distributor(void *data){
    FILE *file;
    file = fopen (fileName, "rb");
        
    // checkif it was able to open the file
    if(file == NULL ) {
        printf("Not able to open the file.\n");
        exit(1);
    }

    // read the first byte from the file
    // which represents the ammount of numbers the file has
    int array_size;
    int err = fread(&array_size, sizeof(array_size), 1, file);
    if(err <= 0){
        printf("Error while reading file");
        exit(1);
    }

    // allocate memory for the array     
    // that will store the numbers read from the file
    numArray = malloc(array_size * sizeof(int));
    if(numArray == NULL){
        printf("Error while allocating memory");
        exit(1);
    }

    // read every number 
    if(fread(numArray, sizeof(int), array_size, file) != array_size){
        printf("Error while reading numbers to array");
        exit(1);
    }

    // close file pointer
    fclose(file);

    print_int_array(numArray, array_size);

    pthread_exit(EXIT_SUCCESS);
}

void *worker(void *data){
    int id = (int*)data;

    // lock until it receives work
    if(processIteraction > 1){          // if its the first iteraction it is sort, otherwise it is merge

    }
    

    statusWorkers[id] = EXIT_SUCCESS;
    pthread_exit(&statusWorkers[id]);
}

// fucntion to print a formatted array to the console
void print_int_array(int *arr, int size){
    printf("Printing array:\nSize: %d\n", size);
    printf("[");
    for (int i = 0; i < size; i++){
        if (i != 0){
            printf(", ");
        }
        printf("%d", arr[i]);
    }
    printf("]\n");
}