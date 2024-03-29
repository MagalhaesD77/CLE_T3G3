/**
 *  \file sharedRegion.h (interface file)
 *
 *  \brief Problem name: Portuguese Text processing.
 *
 *  Synchronization based on monitors.
 *  Both threads and the monitor are implemented using the pthread library which enables the creation of a
 *  monitor of the Lampson / Redell type.
 *
 *  Data transfer region implemented as a monitor.
 *
 *  Definition of the operations carried out by the producers / consumers:
 *     \li putVal
 *     \li getVal.
 *
 *  \author Diogo Magalhães & Rafael Gil - March 2024
 */

#ifndef UTILS_H
#define UTILS_H




int find_last_outside_word_char_position(char *buffer, int buffer_size);
char convert_to_char(char *c);
char read_next_char(FILE *file);
int bytes_to_read(char c);
int bytes_to_read(char c);
void init_c3_bytes_to_char();
void init_e2_2_bytes_to_char();
int contains(char *array, int array_size, char c);
void add(char *array, int array_size, char c);
char read_next_char_from_array(const char *array, int *index, int array_size);

#endif /* UTILS_H */


