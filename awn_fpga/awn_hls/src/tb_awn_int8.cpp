// Phase 4 testbench: drive awn_forward() with the golden input from quant.npz
// and assert argmax matches EXPECTED_ARGMAX (class 1 = AM-DSB for x_fp).
//
// Also prints int8 logits + dequantized fp32 logits side-by-side with the
// reference fp32 logits, so we can eyeball quantization noise.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "awn_int8.h"
#include "golden_io.h"

int main() {
    int8_t logits_q[AWN_NUM_CLASSES];

    // X_Q is shape [2][128] in golden_io.h — same layout as awn_forward expects.
    awn_forward(X_Q, logits_q);

    // ---- dequantize & argmax ----
    int8_t  best_i = 0;
    int8_t  best_v = -128;
    for (int i = 0; i < AWN_NUM_CLASSES; ++i) {
        if (logits_q[i] > best_v) { best_v = logits_q[i]; best_i = (int8_t)i; }
    }

    printf("AWN int8 HLS forward — golden input from awn_fpga/build/quant.npz\n");
    printf("--------------------------------------------------------------------\n");
    printf("  class | int8 logit | dequant (q*S)  | fp32 ref       | abs err\n");
    printf("--------+------------+----------------+----------------+--------\n");
    float max_abs_err = 0.0f;
    for (int i = 0; i < AWN_NUM_CLASSES; ++i) {
        float dq  = (float)logits_q[i] * S_FC2_OUT;
        float ref = FP_LOGITS[i];
        float err = fabsf(dq - ref);
        if (err > max_abs_err) max_abs_err = err;
        printf("  %3d   | %+5d      | %+12.4f   | %+12.4f   | %.4f\n",
               i, (int)logits_q[i], dq, ref, err);
    }
    printf("--------------------------------------------------------------------\n");
    printf("max |dequant - fp_ref| = %.4f\n", max_abs_err);
    printf("hw argmax = %d   |  expected = %d   |  match=%s\n",
           (int)best_i, EXPECTED_ARGMAX,
           (best_i == EXPECTED_ARGMAX) ? "YES" : "NO");

    if (best_i != EXPECTED_ARGMAX) {
        printf("\n*** FAIL: argmax mismatch ***\n");
        return 1;
    }
    printf("\n*** PASS: argmax == %d (AM-DSB) ***\n", EXPECTED_ARGMAX);
    return 0;
}
