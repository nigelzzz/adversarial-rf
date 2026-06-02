#include "simple_add.h"

void simple_add(const data_t a[N], const data_t b[N], acc_t c[N]) {
#pragma HLS INTERFACE ap_ctrl_hs port=return
#pragma HLS INTERFACE bram port=a
#pragma HLS INTERFACE bram port=b
#pragma HLS INTERFACE bram port=c

    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        c[i] = (acc_t)a[i] + (acc_t)b[i];
    }
}
