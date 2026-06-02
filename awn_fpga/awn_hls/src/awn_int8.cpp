// AWN int8 forward pass — HLS C++ port of awn_fpga/sw/refmodel.py main().
// All requantize multipliers reproduce sw/refmodel.py:q_multiplier exactly,
// so csim is bit-exact against the iverilog flow and the npz golden file.

#include "awn_int8.h"
#include "awn_weights.h"
#include "awn_biases.h"
#include "awn_qparams.h"

#include <stdint.h>

// ----- low-level arithmetic primitives --------------------------------------

// (acc * mul + half) >> shift, with arithmetic right shift on signed int.
// Matches sw/refmodel.py:srdhm exactly.
static inline int32_t srdhm(int32_t acc, int32_t mul, int8_t shift) {
    int64_t prod = (int64_t)acc * (int64_t)mul;
    int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : (int64_t)0;
    int64_t sum  = prod + half;
    return (int32_t)(sum >> shift);
}

static inline int8_t sat8(int32_t v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return (int8_t)v;
}

static inline int8_t requant_s8(int32_t acc, int32_t mul, int8_t shift) {
    return sat8(srdhm(acc, mul, shift));
}

// LeakyReLU with alpha = 328/2^15 ≈ 0.01001 (matches refmodel.py default).
static inline int8_t leaky_relu_s8(int8_t v) {
    if (v >= 0) return v;
    int32_t neg = ((int32_t)v * 328 + (1 << 14)) >> 15;
    return sat8(neg);
}

static inline int8_t relu_s8(int8_t v) { return v < 0 ? (int8_t)0 : v; }

static inline int8_t lut_s8(int8_t v, const int8_t table[LUT_SIZE]) {
    return table[(int)v + 128];
}


// ----- conv1: ZeroPad((3,3,0,0)) + Conv2d(1->64, k=(2,7)) + LeakyReLU --------
static void conv1_block(const int8_t x[AWN_IN_CH][AWN_IN_LEN],
                              int8_t y[CONV1_OUT_CH][AWN_IN_LEN]) {
#pragma HLS ARRAY_PARTITION variable=W1 dim=3 complete   // kh = 2
#pragma HLS ARRAY_PARTITION variable=W1 dim=4 complete   // kw = 7
#pragma HLS ARRAY_PARTITION variable=x  dim=1 complete   // 2 IQ channels
    for (int oc = 0; oc < CONV1_OUT_CH; ++oc) {
        for (int w = 0; w < AWN_IN_LEN; ++w) {
#pragma HLS PIPELINE II=1
            int32_t acc = b1[oc];
            for (int h = 0; h < CONV1_KH; ++h) {
                for (int kw = 0; kw < CONV1_KW; ++kw) {
                    int w_in = w - 3 + kw;
                    int8_t xv = (w_in >= 0 && w_in < AWN_IN_LEN)
                                ? x[h][w_in] : (int8_t)0;
                    acc += (int32_t)xv * (int32_t)W1[oc][0][h][kw];
                }
            }
            int8_t r = requant_s8(acc, CONV1_MUL, CONV1_SHIFT);
            y[oc][w] = leaky_relu_s8(r);
        }
    }
}

// ----- conv2: Conv1d(64->64, k=5, pad=2) + LeakyReLU -------------------------
static void conv2_block(const int8_t x[CONV1_OUT_CH][AWN_IN_LEN],
                              int8_t y[CONV2_OUT_CH][AWN_IN_LEN]) {
#pragma HLS ARRAY_PARTITION variable=W2 dim=3 complete   // unroll kt=5
#pragma HLS ARRAY_PARTITION variable=x  dim=2 cyclic factor=5
    for (int oc = 0; oc < CONV2_OUT_CH; ++oc) {
        for (int t = 0; t < AWN_IN_LEN; ++t) {
#pragma HLS PIPELINE II=1
            int32_t acc = b2[oc];
            for (int ic = 0; ic < CONV1_OUT_CH; ++ic) {
                for (int kt = 0; kt < CONV2_K; ++kt) {
                    int t_in = t - 2 + kt;
                    int8_t xv = (t_in >= 0 && t_in < AWN_IN_LEN)
                                ? x[ic][t_in] : (int8_t)0;
                    acc += (int32_t)xv * (int32_t)W2[oc][ic][kt];
                }
            }
            int8_t r = requant_s8(acc, CONV2_MUL, CONV2_SHIFT);
            y[oc][t] = leaky_relu_s8(r);
        }
    }
}

// ----- Lifting U-branch ------------------------------------------------------
static void u_branch(const int8_t odd_q[LIFT_CH][64],
                           int8_t y_tanh[LIFT_CH][64]) {
#pragma HLS ARRAY_PARTITION variable=Wu1 dim=3 complete   // kt=3
#pragma HLS ARRAY_PARTITION variable=Wu4 dim=3 complete
    int8_t u1[LIFT_CH][66];
    for (int oc = 0; oc < LIFT_CH; ++oc) {
        for (int t = 0; t < 66; ++t) {
#pragma HLS PIPELINE II=1
            int32_t acc = bu1[oc];
            for (int ic = 0; ic < LIFT_CH; ++ic) {
                for (int kt = 0; kt < LIFT_K; ++kt) {
                    int idx = t - 2 + kt;
                    if (idx < 0)        idx = -idx;
                    else if (idx >= 64) idx = 2 * 63 - idx;
                    acc += (int32_t)odd_q[ic][idx] * (int32_t)Wu1[oc][ic][kt];
                }
            }
            int8_t r = requant_s8(acc, U1_MUL, U1_SHIFT);
            u1[oc][t] = leaky_relu_s8(r);
        }
    }
    for (int oc = 0; oc < LIFT_CH; ++oc) {
        for (int t = 0; t < 64; ++t) {
#pragma HLS PIPELINE II=1
            int32_t acc = bu4[oc];
            for (int ic = 0; ic < LIFT_CH; ++ic) {
                for (int kt = 0; kt < LIFT_K; ++kt) {
                    acc += (int32_t)u1[ic][t + kt] * (int32_t)Wu4[oc][ic][kt];
                }
            }
            int8_t r = requant_s8(acc, U4_MUL, U4_SHIFT);
            y_tanh[oc][t] = lut_s8(r, TANH_U_LUT);
        }
    }
}

// ----- Lifting P-branch ------------------------------------------------------
static void p_branch(const int8_t c_q[LIFT_CH][64],
                           int8_t y_tanh[LIFT_CH][64]) {
#pragma HLS ARRAY_PARTITION variable=Wp1 dim=3 complete
#pragma HLS ARRAY_PARTITION variable=Wp4 dim=3 complete
    int8_t p1[LIFT_CH][66];
    for (int oc = 0; oc < LIFT_CH; ++oc) {
        for (int t = 0; t < 66; ++t) {
#pragma HLS PIPELINE II=1
            int32_t acc = bp1[oc];
            for (int ic = 0; ic < LIFT_CH; ++ic) {
                for (int kt = 0; kt < LIFT_K; ++kt) {
                    int idx = t - 2 + kt;
                    if (idx < 0)        idx = -idx;
                    else if (idx >= 64) idx = 2 * 63 - idx;
                    acc += (int32_t)c_q[ic][idx] * (int32_t)Wp1[oc][ic][kt];
                }
            }
            int8_t r = requant_s8(acc, P1_MUL, P1_SHIFT);
            p1[oc][t] = leaky_relu_s8(r);
        }
    }
    for (int oc = 0; oc < LIFT_CH; ++oc) {
        for (int t = 0; t < 64; ++t) {
#pragma HLS PIPELINE II=1
            int32_t acc = bp4[oc];
            for (int ic = 0; ic < LIFT_CH; ++ic) {
                for (int kt = 0; kt < LIFT_K; ++kt) {
                    acc += (int32_t)p1[ic][t + kt] * (int32_t)Wp4[oc][ic][kt];
                }
            }
            int8_t r = requant_s8(acc, P4_MUL, P4_SHIFT);
            y_tanh[oc][t] = lut_s8(r, TANH_P_LUT);
        }
    }
}

// ----- AvgPool1d over 64 samples (rescaled by mul/shift) ---------------------
static void avgpool_64(const int8_t x[LIFT_CH][64],
                             int8_t y[LIFT_CH],
                             int32_t mul, int8_t shift) {
    for (int c = 0; c < LIFT_CH; ++c) {
#pragma HLS PIPELINE II=1
        int32_t acc = 0;
        for (int t = 0; t < 64; ++t) acc += (int32_t)x[c][t];
        y[c] = requant_s8(acc, mul, shift);
    }
}

// ----- Linear (GEMM with single column vector input) -------------------------
template <int M, int K>
static void linear_acc(const int8_t W[M][K],
                       const int32_t *bias,
                       const int8_t x[K],
                             int32_t out[M]) {
    for (int m = 0; m < M; ++m) {
#pragma HLS PIPELINE II=1
        int32_t acc = (bias != 0) ? bias[m] : (int32_t)0;
        for (int k = 0; k < K; ++k) {
            acc += (int32_t)W[m][k] * (int32_t)x[k];
        }
        out[m] = acc;
    }
}


// ============================================================================
//  TOP function
// ============================================================================
void awn_forward(const int8_t  x_q[AWN_IN_CH][AWN_IN_LEN],
                       int8_t  logits_q[AWN_NUM_CLASSES]) {
    int8_t y1[CONV1_OUT_CH][AWN_IN_LEN];
    conv1_block(x_q, y1);

    int8_t y2[CONV2_OUT_CH][AWN_IN_LEN];
    conv2_block(y1, y2);

    int8_t even_q[LIFT_CH][64];
    int8_t odd_q [LIFT_CH][64];
    for (int c = 0; c < LIFT_CH; ++c) {
        for (int t = 0; t < 64; ++t) {
            even_q[c][t] = y2[c][2 * t];
            odd_q [c][t] = y2[c][2 * t + 1];
        }
    }

    int8_t u_tanh[LIFT_CH][64];
    u_branch(odd_q, u_tanh);

    int8_t c_q[LIFT_CH][64];
    for (int ch = 0; ch < LIFT_CH; ++ch) {
        for (int t = 0; t < 64; ++t) {
            int8_t er = requant_s8((int32_t)even_q[ch][t], EVEN_MUL, EVEN_SHIFT);
            c_q[ch][t] = sat8((int32_t)er + (int32_t)u_tanh[ch][t]);
        }
    }

    int8_t p_tanh[LIFT_CH][64];
    p_branch(c_q, p_tanh);

    int8_t d_q[LIFT_CH][64];
    for (int ch = 0; ch < LIFT_CH; ++ch) {
        for (int t = 0; t < 64; ++t) {
            int8_t orq = requant_s8((int32_t)odd_q[ch][t], ODD_MUL, ODD_SHIFT);
            d_q[ch][t] = sat8((int32_t)orq - (int32_t)p_tanh[ch][t]);
        }
    }

    int8_t avg_d_q[LIFT_CH];
    int8_t avg_c_q[LIFT_CH];
    avgpool_64(d_q, avg_d_q, AVG_D_MUL, AVG_D_SHIFT);
    avgpool_64(c_q, avg_c_q, AVG_C_MUL, AVG_C_SHIFT);

    int8_t cat_q[SE_IN];
    for (int c = 0; c < LIFT_CH; ++c) {
        cat_q[c]            = requant_s8((int32_t)avg_d_q[c],
                                         D_TO_CONCAT_MUL, D_TO_CONCAT_SHIFT);
        cat_q[c + LIFT_CH]  = requant_s8((int32_t)avg_c_q[c],
                                         C_TO_CONCAT_MUL, C_TO_CONCAT_SHIFT);
    }

    int32_t se0_acc[SE_HID];
    linear_acc<SE_HID, SE_IN>(Wse0, /*bias=*/(const int32_t*)0, cat_q, se0_acc);
    int8_t se0_q[SE_HID];
    for (int i = 0; i < SE_HID; ++i) {
        int8_t r = requant_s8(se0_acc[i], SE0_MUL, SE0_SHIFT);
        se0_q[i] = relu_s8(r);
    }

    int32_t se3_acc[SE_IN];
    linear_acc<SE_IN, SE_HID>(Wse3, /*bias=*/(const int32_t*)0, se0_q, se3_acc);
    int8_t sig_q[SE_IN];
    for (int i = 0; i < SE_IN; ++i) {
        int8_t r = requant_s8(se3_acc[i], SE3_MUL, SE3_SHIFT);
        sig_q[i] = lut_s8(r, SIGMOID_LUT);
    }

    int8_t se_mul_q[SE_IN];
    for (int i = 0; i < SE_IN; ++i) {
        int32_t prod = (int32_t)sig_q[i] * (int32_t)cat_q[i];
        se_mul_q[i] = requant_s8(prod, SE_MUL_MUL, SE_MUL_SHIFT);
    }

    int32_t fc0_acc[FC0_OUT];
    linear_acc<FC0_OUT, FC0_IN>(Wfc0, bfc0, se_mul_q, fc0_acc);
    int8_t fc0_q[FC0_OUT];
    for (int i = 0; i < FC0_OUT; ++i) {
        int8_t r = requant_s8(fc0_acc[i], FC0_MUL, FC0_SHIFT);
        fc0_q[i] = leaky_relu_s8(r);
    }

    int32_t fc2_acc[FC2_OUT];
    linear_acc<FC2_OUT, FC0_OUT>(Wfc2, bfc2, fc0_q, fc2_acc);
    for (int i = 0; i < FC2_OUT; ++i) {
        logits_q[i] = requant_s8(fc2_acc[i], FC2_MUL, FC2_SHIFT);
    }
}
