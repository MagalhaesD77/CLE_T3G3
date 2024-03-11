#include <stdio.h>
#include <stdlib.h>

void print_int_array(int *arr, int size);
void imperativeBitonicSort(int* a, int N);

int main(int argc, char *argv[])
{
    // check if there were arguments passed
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
        
        // checkif it was able to open the file
        if(file == NULL ) {
            printf("Not able to open the file.\n");
            continue;       // if it was not able to open the file, go to the next iter and open the next file
        }

        // read the first byte from the file
        // which represents the ammount of numbers the file has
        int array_size;
        int err = fread(&array_size, sizeof(array_size), 1, file);
        if(err <= 0){
            printf("Error while reading file");
            return 1;
        }

        // allocate memory for the array 
        // that will store the numbers read from the file
        int* number_array;
        number_array = malloc(array_size * sizeof(int));
        if(number_array == NULL){
            printf("Error while allocating memory");
            exit(1);
        }

        // read every number 
        if(fread(number_array, sizeof(int), array_size, file) != array_size){
            printf("Error while reading numbers to array");
            exit(1);
        }

        // perform iterative bitonic sort
        impBitonicSort(number_array, array_size);

        // verify if the array is ordered
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

    return 0;
}

void imperativeBitonicSort(int* array, int N){
    // iterate through the powers of 2 up to N
    // simulates the layers of the algorithm
    for (int k=2; k<=N; k=2*k) {
        // iterate throught half of the current value of k
        // controls the length of the comparison between the numbers
        for (int j=k/2; j>0; j=j/2) {
            // iterates through the whole array
            for (int i=0; i<N; i++) {
                int ij=i^j;     // bitewise XOR, to calculate the index where to perform the comparison
                if ((ij)>i) {   // assure correct order
                    if (((i&k)==0                                       // bitwise AND to check if i-th index is in the lower half of the bitonic sequence
                                && array[i] > array[ij])                // check if i-th element is bigger than ij

                    || ((i&k)!=0                                        // bitwise AND to check if i-th index is in the upper half of the bitonic sequence
                                && array[i] < array[ij])) {             // check if i-th element is lower than ij
                        
                        // performs a common swap between the elements of the array
                        int aux = array[i];
                        array[i] = array[j];
                        array[j] = aux;
                    }
                }
            }
        }
    }
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