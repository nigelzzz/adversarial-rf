#ifndef AWN_DIMS_H
#define AWN_DIMS_H

// RML2016.10a: 2 IQ channels, 128 samples, 11 modulation classes.
#define AWN_IN_CH       2
#define AWN_IN_LEN      128
#define AWN_NUM_CLASSES 11

// conv1: in=1 (IQ stacked as 2x7 kernel), out=64
#define CONV1_OUT_CH    64
#define CONV1_KH        2
#define CONV1_KW        7

// conv2 (1D): 64 -> 64, kernel 5
#define CONV2_OUT_CH    64
#define CONV2_K         5

// lifting Updator / Predictor: 64 -> 64, kernel 3
#define LIFT_CH         64
#define LIFT_K          3

// SE attention
#define SE_IN           128
#define SE_HID          32

// FC
#define FC0_IN          128
#define FC0_OUT         320
#define FC2_OUT         AWN_NUM_CLASSES

// LUT size (full int8 domain)
#define LUT_SIZE        256

#endif
