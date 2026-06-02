// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.2 (64-bit)
// Tool Version Limit: 2023.10
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef __awn_forward_mac_muladd_7s_8s_14s_14_4_1__HH__
#define __awn_forward_mac_muladd_7s_8s_14s_14_4_1__HH__
#include "awn_forward_mac_muladd_7s_8s_14s_14_4_1_DSP48_22.h"

template<
    int ID,
    int NUM_STAGE,
    int din0_WIDTH,
    int din1_WIDTH,
    int din2_WIDTH,
    int dout_WIDTH>
SC_MODULE(awn_forward_mac_muladd_7s_8s_14s_14_4_1) {
    sc_core::sc_in_clk clk;
    sc_core::sc_in<sc_dt::sc_logic> reset;
    sc_core::sc_in<sc_dt::sc_logic> ce;
    sc_core::sc_in< sc_dt::sc_lv<din0_WIDTH> >   din0;
    sc_core::sc_in< sc_dt::sc_lv<din1_WIDTH> >   din1;
    sc_core::sc_in< sc_dt::sc_lv<din2_WIDTH> >   din2;
    sc_core::sc_out< sc_dt::sc_lv<dout_WIDTH> >   dout;



    awn_forward_mac_muladd_7s_8s_14s_14_4_1_DSP48_22 awn_forward_mac_muladd_7s_8s_14s_14_4_1_DSP48_22_U;

    SC_CTOR(awn_forward_mac_muladd_7s_8s_14s_14_4_1):  awn_forward_mac_muladd_7s_8s_14s_14_4_1_DSP48_22_U ("awn_forward_mac_muladd_7s_8s_14s_14_4_1_DSP48_22_U") {
        awn_forward_mac_muladd_7s_8s_14s_14_4_1_DSP48_22_U.clk(clk);
        awn_forward_mac_muladd_7s_8s_14s_14_4_1_DSP48_22_U.rst(reset);
        awn_forward_mac_muladd_7s_8s_14s_14_4_1_DSP48_22_U.ce(ce);
        awn_forward_mac_muladd_7s_8s_14s_14_4_1_DSP48_22_U.in0(din0);
        awn_forward_mac_muladd_7s_8s_14s_14_4_1_DSP48_22_U.in1(din1);
        awn_forward_mac_muladd_7s_8s_14s_14_4_1_DSP48_22_U.in2(din2);
        awn_forward_mac_muladd_7s_8s_14s_14_4_1_DSP48_22_U.dout(dout);

    }

};

#endif //
