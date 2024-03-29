
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <pthread.h>
#include <errno.h>

#include "utils.h"



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



int find_last_outside_word_char_position(char *buffer, int buffer_size){
    int n = buffer_size - 1;

    // Check if EOF. This should not happen.
    if (buffer[n]==EOF){
        return n;
    }
    
    char c[4];
    char single_byte_char;
    while (n >= 0){
        // Initialize the array with null characters
        memset(c, '\0', sizeof(c));   // Clear the array
        c[0] = buffer[n];

        // Read more bytes if necessary
        int read_bytes = bytes_to_read(c[0]);
        // printf("Quantity of bytes to read: %d\n", read_bytes);
        // printf("Bytes to read: %d\n", read_bytes);
        for(int i = 1; i < read_bytes; i++){
            c[i] = buffer[n+i];
        }

        single_byte_char = convert_to_char(c);

        // make lower case
        single_byte_char = tolower(single_byte_char);

        if (contains(outside_word_chars, outside_word_array_size, single_byte_char)){
            return n + read_bytes;
        }

        n--;
    }

    return -1;
}


char convert_to_char(char *c){
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

char read_next_char_from_array(const char *array, int *index, int array_size){
    // Create clean 
    char c[4] = {};
    c[0] = array[(*index)++];

    // Check if index exceeds array size
    if (*index >= array_size || c[0] == '\0'){
        return EOF; // Returning null character to indicate end of array
    }

    // Read more bytes if necessary
    int read_bytes = bytes_to_read(c[0]);
    for(int i = 0; i < read_bytes - 1; i++){
        c[i+1] = array[(*index)++];
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
    // Init array used for conversion of multiple bytes, starting with 0xc3 byte, chars into single byte chars
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
    // Init array used for conversion of multiple bytes, starting with 0xe2 byte, chars into single byte chars
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
