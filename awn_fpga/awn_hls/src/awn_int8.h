#ifndef AWN_INT8_H
#define AWN_INT8_H

#include <stdint.h>
#include "awn_dims.h"

// Top function: forward pass over RML2016.10a 2x128 IQ window.
//   x_q: pre-quantized int8 input, shape [AWN_IN_CH][AWN_IN_LEN]
//   logits_q: int8 logits, shape [AWN_NUM_CLASSES]
// To get fp32 logits multiply by S_FC2_OUT (golden_io.h).
void awn_forward(const int8_t  x_q[AWN_IN_CH][AWN_IN_LEN],
                       int8_t  logits_q[AWN_NUM_CLASSES]);

#endif
