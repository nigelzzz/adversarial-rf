#include "simple_add.h"
#include <cstdio>
#include <cstdlib>

int main() {
    data_t a[N], b[N];
    acc_t  c[N], ref[N];

    for (int i = 0; i < N; i++) {
        a[i] = (data_t)((i % 7) - 3);     // small int8 values
        b[i] = (data_t)(((i * 5) % 11) - 5);
        ref[i] = (acc_t)a[i] + (acc_t)b[i];
    }

    simple_add(a, b, c);

    int err = 0;
    for (int i = 0; i < N; i++) {
        if (c[i] != ref[i]) {
            printf("MISMATCH i=%d  hw=%d  ref=%d\n",
                   i, (int)c[i], (int)ref[i]);
            err++;
        }
    }

    if (err == 0) {
        printf("*** PASS: %d elements all match ***\n", N);
        return 0;
    } else {
        printf("*** FAIL: %d mismatches ***\n", err);
        return 1;
    }
}
