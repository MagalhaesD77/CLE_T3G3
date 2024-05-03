/**
 *  \file utils.c (implementation file)
 *
 *  \brief Problem name: Portuguese Text processing.
 *
 *  Text processing utility functions.
 *
 *  Definition of the initial operations carried out by the dispatcher / worker processes:
 *     \li find_last_outside_word_char_position
 *     \li convert_to_char
 *     \li read_next_char_from_array
 *     \li bytes_to_read
 *     \li init_c3_bytes_to_char
 *     \li init_e2_2_bytes_to_char
 *     \li contains
 *     \li add
 *
 *  \author Diogo Magalhães & Rafael Gil - May 2024
 */


#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "utils.h"

/** \brief array with codes of outside word characters */
extern char outside_word_chars[];

/** \brief size of the outside word characters array */
extern int outside_word_array_size;

/** \brief array with conversion of multiple bytes, starting with 0xc3 byte, chars into single byte chars */
char c3_bytes_to_char[256] = {};

/** \brief array with conversion of multiple bytes, starting with 0xe2 byte, chars into single byte chars */
char e2_2_bytes_to_char[256] = {};



/**
 *  \brief Find the last char position of the last full word in the buffer.
 *
 *  \param buffer buffer to find the last word position
 *  \param buffer_size size of the buffer
 */

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

/**
 *  \brief Convert a list of char a single byte char.
 *
 *  \param c character to be converted
 */

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

/**
 *  \brief Read the next char from the array.
 *
 *  \param array array to work with
 *  \param index pointer to start reading
 *  \param array_size size of the array
 * 
 */

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

/**
 *  \brief Return the number of extra bytes needed to read the full char.
 *
 *  \param c character
 */

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

/**
 *  \brief Initialize the c3 chars conversion array.
 */

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

/**
 *  \brief Initialize the e2 chars conversion array.
 */

void init_e2_2_bytes_to_char(){
    // Init array used for conversion of multiple bytes, starting with 0xe2 byte, chars into single byte chars
    e2_2_bytes_to_char[0x9c] = e2_2_bytes_to_char[0x9d] = 0x22;

    e2_2_bytes_to_char[0x93] = e2_2_bytes_to_char[0xa6] = 0x2e;

    e2_2_bytes_to_char[0x98] = e2_2_bytes_to_char[0x99] = 0x27;

}

/**
 *  \brief Check if a char is in the array.
 *
 *  \param array array with content
 *  \param array_size size of the array
 *  \param c char to find
 */

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

/**
 *  \brief Add a char to the array.
 *
 *  \param array array to add the char
 *  \param array_size size of the array
 *  \param c char to add
 */

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
