#include <stdio.h>
#include <string.h>

#define MAX 100 // Example of global variable

int max = 100; // Example of local global variable

// Define the list of outside word characters
char outside_word_chars[] = {0x20, 0x9, 0xD, 0xA};
char consonants[] = {'b', 'c'};

int contains(char c, char *array);
void add(char *array, char c);

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

        printf("File: %s opened\n", argv[i]);

        // Read file caracter by caracter
        char c;
        int count_word = 0;
        int count_two_consoant_words = 0;
        int two_consoant_bool = 0;
        char word_chars[] = {'a', 'b'};
        int inside_word = 0;
        while ((c = fgetc(file)) != EOF)
        {
            // Convert to lower case
            if (c >= 65 && c <= 90)
            {
                c += 32;
            }

            // Check if char c is in outside_word_chars array
            if (contains(c, outside_word_chars))
            {
                inside_word = 0;
                two_consoant_bool = 0;
                memset(word_chars, '\0', sizeof(word_chars));   // Clear the array
                // printf(" ");
            }
            else
            {
                if (inside_word == 0)
                {
                    count_word++;
                    inside_word = 1;
                    
                }
                else{
                    if (contains(c, consonants) && two_consoant_bool == 0){
                        if (contains(c, word_chars)){

                            count_two_consoant_words++;
                            two_consoant_bool = 1;

                        }
                        else{
                            // add the caracter to the array
                            add(word_chars, c);
                        }
                    }
                }

            }

        }


        printf("\nNumber of words: %d\n\n", count_word);
        printf("\nNumber of words with two consonants: %d\n\n", count_two_consoant_words);
    }
    fclose(file);

 
    return 0;
}


int contains(char c, char *array)
{
    for (int i = 0; i < sizeof(array); i++)
    {
        if (c == array[i])
        {
            return 1;
        }
    }
    return 0;
}


void add(char *array, char c)
{
    for (int i = 0; i < sizeof(array); i++)
    {
        if (array[i] == '\0')
        {
            array[i] = c;
            break;
        }
    }
}

