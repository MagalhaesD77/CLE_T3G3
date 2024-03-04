#include <stdio.h>
#include <stdlib.h>

void print_int_array(int *arr, int size);
void bitonicMerge(int arr[], int low, int count, int dir);
void bitonicSort(int arr[], int low, int count, int dir);
void sort(int arr[], int n, int up);

int main(int argc, char *argv[])
{
    // print the arguments provided
    for (int i = 0; i < argc; i++)
    {
        printf("arg %d: %s\n", i, argv[i]);
    }
    printf("\n");

    if (argc < 2)
    {
        printf("No file provided\n");
        return 1;
    }


    // Try to open each file
    FILE *file;
    for (int i = 1; i < argc; i++)
    {
        file = fopen (argv[i], "rb");

        if(file == NULL ) {
            printf("Not able to open the file.");
            continue;
            // return 1;
        }

        int array_size;
        int err = fread(&array_size, sizeof(array_size), 1, file);
        if(err <= 0){
            printf("Error while reading file");
            return 1;
        }

        int* number_array;
        number_array = malloc(array_size * sizeof(int));
        if(number_array == NULL){
            printf("Error while allocating memory");
            exit(1);
        }

        if(fread(number_array, sizeof(int), array_size, file) != array_size){
            // printf("%d ", num_read);
            printf("Error while reading numbers to array");
            exit(1);
        }

        //print_int_array(array, array_size);
        sort(number_array ,array_size, 0);
        //print_int_array(array, array_size);

        for (int i = 1; i < array_size; i++){
            if (number_array[i - 1] < number_array[i]){ 
                printf ("Error in position %d between element %d and %d\n", i, number_array[i - 1], number_array[i]);
                break;
            }
            if (i == (array_size - 1)){
                printf ("Everything is OK!\n");
            }
        }
    }



    return 1;
}


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

void bitonicMerge(int arr[], int low, int count, int dir) {
    if (count > 1) {
        int k = count / 2;
        for (int i = low; i < low + k; i++) {
            if ((arr[i] > arr[i + k]) == dir) {
                // Swap elements if they are in the wrong order
                int temp = arr[i];
                arr[i] = arr[i + k];
                arr[i + k] = temp;
            }
        }
        bitonicMerge(arr, low, k, dir);
        bitonicMerge(arr, low + k, k, dir);
    }
}

// Function to perform bitonic sort recursively
void bitonicSort(int arr[], int low, int count, int dir) {
    if (count > 1) {
        int k = count / 2;
        
        // Sort in ascending order
        bitonicSort(arr, low, k, 0);
        
        // Sort in descending order
        bitonicSort(arr, low + k, k, 1);
        
        // Merge both sorted halves
        bitonicMerge(arr, low, count, dir);
    }
}

void sort(int arr[], int n, int up) {
    bitonicSort(arr, 0, n, up);
}