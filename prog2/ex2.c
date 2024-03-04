#include <stdio.h>

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

        if(file == NULL) {
            printf("Not able to open the file.");
            continue;
            // return 1;
        }

        int num;
        int *dynamicIntArray = NULL;
        int count = 0;
        while (fread(&num, sizeof(int), 1, file) != 0){   // fread will return 1 when it is able to read a number and 0 when it is not able to read a number (end of file)
            
            // allocate memory for the array and add the number to the array
            dynamicIntArray = (int *)realloc(dynamicIntArray, ++count * sizeof(int));
            if (dynamicIntArray == NULL) {
                printf("Memory allocation failed. Exiting program.\n");
                return 1; // Indicates an error
            }
            dynamicIntArray[count-1] = num;
        }

        // print int array
        print_int_array(dynamicIntArray, count);

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


