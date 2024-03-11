#include <stdio.h>
#include <string.h>
#include <ctype.h>

// Functions declaration
int contains(char *array, int array_size, char c);
void add(char *array, int array_size, char c);
char read_next_char(FILE *file);
int bytes_to_read(char c);
void init_c3_bytes_to_char();
void init_e2_2_bytes_to_char();


// Global variables
char alphanumeric_chars_underscore[] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_'};
int alphanumeric_chars_underscore_array_size = sizeof(alphanumeric_chars_underscore)/sizeof(char);

char consonants[] = {'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'z'};
int consonants_array_size = sizeof(consonants)/sizeof(char);

char outside_word_chars[] = {0x20, 0x9, 0xD, 0xA, 0x2d, 0x22, 0x5b, 0x5d, 0x28, 0x29, 0x2e, 0x2c, 0x3a, 0x3b, 0x3f, 0x21};  // Contains white space symbols (From 0x20 to 0xA), separation symbols (From 0x2d to 0x29 ), punctuation symbols (From 0x2e to 0x21)
int outside_word_array_size = sizeof(outside_word_chars)/sizeof(char);


// Conversion tables
char c3_bytes_to_char[256] = {};
char e2_2_bytes_to_char[256] = {};


int main(int argc, char *argv[])
{
    // Initialize the c3 conversion table
    init_c3_bytes_to_char();
    init_e2_2_bytes_to_char();

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
            printf("Not able to open the file named: %s\n", argv[i]);
            continue;
            // return 1;
        }
        printf("File name: %s\n", argv[i]);

        // Read file caracter by caracter
        char c;
        int count_word = 0;
        int count_two_consoant_words = 0;
        int two_consoant_bool = 0;
        char word_chars[consonants_array_size];
        int inside_word = 0;
        while ((c = read_next_char(file)) != EOF)
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


        printf("Number of words: %d\n", count_word);
        printf("Number of words with two consonants: %d\n\n", count_two_consoant_words);
    }
    fclose(file);

 
    return 0;
}


char read_next_char(FILE *file){
    // Create clean 
    char c[4] = {};
    c[0] = fgetc(file);

    // Check if EOF
    if (c[0]==EOF){
        return EOF;
    }

    // Read more bytes if necessary
    int read_bytes = bytes_to_read(c[0]);
    for(int i = 0; i < read_bytes - 1; i++){
        c[i+1] = fgetc(file);
    }

    // Convert to char
    switch ((unsigned char) c[0])
    {
    case 0xc3:
        return c3_bytes_to_char[(unsigned char) c[1]];
    case 0xe2:
        return e2_2_bytes_to_char[(unsigned char) c[2]];
    default:
        // unicode letter
        return c[0];
    }
}


int bytes_to_read(char c){
    //Return the number of set bytes
    int count = 0;

    // Count number of initial 1s
    int i = 7;
    while(((c >> i) & 1) != 0){
        count++;
        i--;
    }

    return count;
}


void init_c3_bytes_to_char(){
    c3_bytes_to_char[0xa1] = c3_bytes_to_char[0xa0] = c3_bytes_to_char[0xa2] = c3_bytes_to_char[0xa3] = 'a';
    c3_bytes_to_char[0x81] = c3_bytes_to_char[0x80] = c3_bytes_to_char[0x82] = c3_bytes_to_char[0x83] = 'A';

    c3_bytes_to_char[0xa9] = c3_bytes_to_char[0xa8] = c3_bytes_to_char[0xaa] = 'e';
    c3_bytes_to_char[0x89] = c3_bytes_to_char[0x88] = c3_bytes_to_char[0x8a] = 'E';

    c3_bytes_to_char[0xad] = c3_bytes_to_char[0xac] = 'i';
    c3_bytes_to_char[0x8d] = c3_bytes_to_char[0x8c] = 'I';

    c3_bytes_to_char[0xb3] = c3_bytes_to_char[0xb2] = c3_bytes_to_char[0xb4] = c3_bytes_to_char[0xb5] = 'o';
    c3_bytes_to_char[0x93] = c3_bytes_to_char[0x92] = c3_bytes_to_char[0x94] = c3_bytes_to_char[0x95] = 'O';

    c3_bytes_to_char[0xba] = c3_bytes_to_char[0xb9] = 'u';
    c3_bytes_to_char[0x9a] = c3_bytes_to_char[0x99] = 'U';

    c3_bytes_to_char[0xa7] = 'c';
    c3_bytes_to_char[0x87] = 'C';
}

void init_e2_2_bytes_to_char(){
    e2_2_bytes_to_char[0x9c] = e2_2_bytes_to_char[0x9d] = 0x22;

    e2_2_bytes_to_char[0x93] = e2_2_bytes_to_char[0xa6] = 0x2e;

    e2_2_bytes_to_char[0x98] = e2_2_bytes_to_char[0x99] = 0x27;

}


int contains(char *array, int array_size, char c)
{
    for (int i = 0; i < array_size; i++)
    {
        if (c == array[i])
        {
            return 1;
        }
    }

    return 0;
}


void add(char *array, int array_size, char c)
{
    for (int i = 0; i < array_size; i++)
    {
        if (array[i] == '\0')
        {
            array[i] = c;
            return;
        }
    }
}
