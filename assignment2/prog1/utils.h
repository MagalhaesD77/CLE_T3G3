/**
 *  \file utils.h (interface file)
 *
 *  \brief Problem name: Portuguese Text processing.
 *
 *  Text processing utility functions.
 *
 *  Definition of the initial operations carried out by the main / worker threads:
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

#ifndef UTILS_H
#define UTILS_H

/**
 *  \brief Find the last char position of the last full word in the buffer.
 *
 *  \param buffer buffer to find the last word position
 *  \param buffer_size size of the buffer
 */

int find_last_outside_word_char_position(char *buffer, int buffer_size);

/**
 *  \brief Convert a list of char a single byte char.
 *
 *  \param c character to be converted
 */

char convert_to_char(char *c);

/**
 *  \brief Read the next char from the array.
 *
 *  \param array array to work with
 *  \param index pointer to start reading
 *  \param array_size size of the array
 * 
 */

char read_next_char_from_array(const char *array, int *index, int array_size);

/**
 *  \brief Return the number of extra bytes needed to read the full char.
 *
 *  \param c character
 */

int bytes_to_read(char c);

/**
 *  \brief Initialize the c3 chars conversion array.
 */

void init_c3_bytes_to_char();

/**
 *  \brief Initialize the e2 chars conversion array.
 */

void init_e2_2_bytes_to_char();

/**
 *  \brief Check if a char is in the array.
 *
 *  \param array array with content
 *  \param array_size size of the array
 *  \param c char to find
 */

int contains(char *array, int array_size, char c);

/**
 *  \brief Add a char to the array.
 *
 *  \param array array to add the char
 *  \param array_size size of the array
 *  \param c char to add
 */

void add(char *array, int array_size, char c);

#endif /* UTILS_H */
