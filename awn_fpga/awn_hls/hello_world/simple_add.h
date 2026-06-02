#ifndef SIMPLE_ADD_H
#define SIMPLE_ADD_H

#include <ap_int.h>

#define N 128

typedef ap_int<8>  data_t;
typedef ap_int<16> acc_t;

void simple_add(const data_t a[N], const data_t b[N], acc_t c[N]);

#endif
