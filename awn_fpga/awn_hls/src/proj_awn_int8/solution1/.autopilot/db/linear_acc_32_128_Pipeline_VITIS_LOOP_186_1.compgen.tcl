# This script segment is generated automatically by AutoPilot

set name awn_forward_mul_7s_8s_15_1_1
if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $name BINDTYPE {op} TYPE {mul} IMPL {auto} LATENCY 0 ALLOW_PRAGMA 1
}


set id 10915
set name awn_forward_mac_muladd_7s_8s_15s_15_4_1
set corename simcore_mac
set op mac
set stage_num 4
set clk_width 1
set clk_signed 0
set reset_width 1
set reset_signed 0
set in0_width 7
set in0_signed 1
set in1_width 8
set in1_signed 1
set in2_width 15
set in2_signed 1
set ce_width 1
set ce_signed 0
set out_width 15
set arg_lists {i0 {7 1 +} i1 {8 1 +} m {15 1 +} i2 {15 1 +} p {15 1 +} c_reg {1} rnd {0} acc {0} }
set TrueReset 0
if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $name BINDTYPE {op} TYPE {all} IMPL {dsp_slice} LATENCY 3 ALLOW_PRAGMA 1
}


set op mac
set corename DSP48
if {${::AESL::PGuard_autocg_gen} && ${::AESL::PGuard_autocg_ipmgen}} {
if {[info proc ::AESL_LIB_VIRTEX::xil_gen_dsp48] == "::AESL_LIB_VIRTEX::xil_gen_dsp48"} {
eval "::AESL_LIB_VIRTEX::xil_gen_dsp48 { \
    id ${id} \
    name ${name} \
    corename ${corename} \
    op ${op} \
    reset_level 1 \
    sync_rst true \
    true_reset ${TrueReset} \
    stage_num ${stage_num} \
    clk_width ${clk_width} \
    clk_signed ${clk_signed} \
    reset_width ${reset_width} \
    reset_signed ${reset_signed} \
    in0_width ${in0_width} \
    in0_signed ${in0_signed} \
    in1_width ${in1_width} \
    in1_signed ${in1_signed} \
    in2_width ${in2_width} \
    in2_signed ${in2_signed} \
    ce_width ${ce_width} \
    ce_signed ${ce_signed} \
    out_width ${out_width} \
    arg_lists {${arg_lists}} \
}"
} else {
puts "@W \[IMPL-101\] Cannot find ::AESL_LIB_VIRTEX::xil_gen_dsp48, check your platform lib"
}
}


set id 10917
set name awn_forward_mac_muladd_6s_8s_15s_15_4_1
set corename simcore_mac
set op mac
set stage_num 4
set clk_width 1
set clk_signed 0
set reset_width 1
set reset_signed 0
set in0_width 6
set in0_signed 1
set in1_width 8
set in1_signed 1
set in2_width 15
set in2_signed 1
set ce_width 1
set ce_signed 0
set out_width 15
set arg_lists {i0 {6 1 +} i1 {8 1 +} m {14 1 +} i2 {15 1 +} p {15 1 +} c_reg {1} rnd {0} acc {0} }
set TrueReset 0
if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $name BINDTYPE {op} TYPE {all} IMPL {dsp_slice} LATENCY 3 ALLOW_PRAGMA 1
}


set op mac
set corename DSP48
if {${::AESL::PGuard_autocg_gen} && ${::AESL::PGuard_autocg_ipmgen}} {
if {[info proc ::AESL_LIB_VIRTEX::xil_gen_dsp48] == "::AESL_LIB_VIRTEX::xil_gen_dsp48"} {
eval "::AESL_LIB_VIRTEX::xil_gen_dsp48 { \
    id ${id} \
    name ${name} \
    corename ${corename} \
    op ${op} \
    reset_level 1 \
    sync_rst true \
    true_reset ${TrueReset} \
    stage_num ${stage_num} \
    clk_width ${clk_width} \
    clk_signed ${clk_signed} \
    reset_width ${reset_width} \
    reset_signed ${reset_signed} \
    in0_width ${in0_width} \
    in0_signed ${in0_signed} \
    in1_width ${in1_width} \
    in1_signed ${in1_signed} \
    in2_width ${in2_width} \
    in2_signed ${in2_signed} \
    ce_width ${ce_width} \
    ce_signed ${ce_signed} \
    out_width ${out_width} \
    arg_lists {${arg_lists}} \
}"
} else {
puts "@W \[IMPL-101\] Cannot find ::AESL_LIB_VIRTEX::xil_gen_dsp48, check your platform lib"
}
}


set id 10922
set name awn_forward_mac_muladd_7s_8s_16s_16_4_1
set corename simcore_mac
set op mac
set stage_num 4
set clk_width 1
set clk_signed 0
set reset_width 1
set reset_signed 0
set in0_width 7
set in0_signed 1
set in1_width 8
set in1_signed 1
set in2_width 16
set in2_signed 1
set ce_width 1
set ce_signed 0
set out_width 16
set arg_lists {i0 {7 1 +} i1 {8 1 +} m {15 1 +} i2 {16 1 +} p {16 1 +} c_reg {1} rnd {0} acc {0} }
set TrueReset 0
if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $name BINDTYPE {op} TYPE {all} IMPL {dsp_slice} LATENCY 3 ALLOW_PRAGMA 1
}


set op mac
set corename DSP48
if {${::AESL::PGuard_autocg_gen} && ${::AESL::PGuard_autocg_ipmgen}} {
if {[info proc ::AESL_LIB_VIRTEX::xil_gen_dsp48] == "::AESL_LIB_VIRTEX::xil_gen_dsp48"} {
eval "::AESL_LIB_VIRTEX::xil_gen_dsp48 { \
    id ${id} \
    name ${name} \
    corename ${corename} \
    op ${op} \
    reset_level 1 \
    sync_rst true \
    true_reset ${TrueReset} \
    stage_num ${stage_num} \
    clk_width ${clk_width} \
    clk_signed ${clk_signed} \
    reset_width ${reset_width} \
    reset_signed ${reset_signed} \
    in0_width ${in0_width} \
    in0_signed ${in0_signed} \
    in1_width ${in1_width} \
    in1_signed ${in1_signed} \
    in2_width ${in2_width} \
    in2_signed ${in2_signed} \
    ce_width ${ce_width} \
    ce_signed ${ce_signed} \
    out_width ${out_width} \
    arg_lists {${arg_lists}} \
}"
} else {
puts "@W \[IMPL-101\] Cannot find ::AESL_LIB_VIRTEX::xil_gen_dsp48, check your platform lib"
}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_0_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_1_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_2_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_3_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_4_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_5_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_6_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_7_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_8_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_9_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_10_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_11_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_12_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_13_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_14_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_15_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_16_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_17_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_18_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_19_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_20_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_21_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_22_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_23_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_24_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_25_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_26_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_27_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_28_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_29_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_30_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_31_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_32_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_33_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_34_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_35_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_36_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_37_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_38_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_39_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_40_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_41_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_42_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_43_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_44_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_45_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_46_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_47_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_48_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_49_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_50_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_51_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_52_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_53_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_54_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_55_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_56_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_57_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_58_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_59_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_60_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_61_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_62_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_63_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_64_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_65_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_66_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_67_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_68_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_69_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_70_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_71_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_72_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_73_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_74_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_75_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_76_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_77_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_78_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_79_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_80_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_81_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_82_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_83_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_84_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_85_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_86_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_87_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_88_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_89_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_90_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_91_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_92_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_93_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_94_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_95_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_96_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_97_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_98_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_99_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_100_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_101_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_102_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_103_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_104_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_105_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_106_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_107_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_108_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_109_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_110_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_111_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_112_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_113_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_114_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_115_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_116_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_117_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_118_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_119_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_120_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_121_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_122_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_123_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_124_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_125_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_126_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse0_127_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


# clear list
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_begin
    cg_default_interface_gen_bundle_begin
    AESL_LIB_XILADAPTER::native_axis_begin
}

# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 11237 \
    name out_r \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename out_r \
    op interface \
    ports { out_r_address0 { O 5 vector } out_r_ce0 { O 1 bit } out_r_we0 { O 1 bit } out_r_d0 { O 21 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'out_r'"
}
}


# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11109 \
    name sext_ln186 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln186 \
    op interface \
    ports { sext_ln186 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11110 \
    name x_load_168_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_168_cast \
    op interface \
    ports { x_load_168_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11111 \
    name x_load_253_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_253_cast \
    op interface \
    ports { x_load_253_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11112 \
    name x_load_217_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_217_cast \
    op interface \
    ports { x_load_217_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11113 \
    name x_load_229_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_229_cast \
    op interface \
    ports { x_load_229_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11114 \
    name x_load_252_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_252_cast \
    op interface \
    ports { x_load_252_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11115 \
    name x_load_201_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_201_cast \
    op interface \
    ports { x_load_201_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11116 \
    name x_load_141_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_141_cast \
    op interface \
    ports { x_load_141_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11117 \
    name x_load_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_cast \
    op interface \
    ports { x_load_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11118 \
    name sext_ln190_277 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190_277 \
    op interface \
    ports { sext_ln190_277 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11119 \
    name x_load_172_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_172_cast \
    op interface \
    ports { x_load_172_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11120 \
    name x_load_239_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_239_cast \
    op interface \
    ports { x_load_239_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11121 \
    name x_load_207_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_207_cast \
    op interface \
    ports { x_load_207_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11122 \
    name x_load_178_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_178_cast \
    op interface \
    ports { x_load_178_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11123 \
    name x_load_250_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_250_cast \
    op interface \
    ports { x_load_250_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11124 \
    name x_load_162_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_162_cast \
    op interface \
    ports { x_load_162_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11125 \
    name x_load_183_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_183_cast \
    op interface \
    ports { x_load_183_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11126 \
    name x_load_233_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_233_cast \
    op interface \
    ports { x_load_233_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11127 \
    name x_load_222_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_222_cast \
    op interface \
    ports { x_load_222_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11128 \
    name x_load_197_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_197_cast \
    op interface \
    ports { x_load_197_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11129 \
    name x_load_249_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_249_cast \
    op interface \
    ports { x_load_249_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11130 \
    name x_load_155_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_155_cast \
    op interface \
    ports { x_load_155_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11131 \
    name x_load_225_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_225_cast \
    op interface \
    ports { x_load_225_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11132 \
    name x_load_145_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_145_cast \
    op interface \
    ports { x_load_145_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11133 \
    name x_load_133_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_133_cast \
    op interface \
    ports { x_load_133_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11134 \
    name x_load_140_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_140_cast \
    op interface \
    ports { x_load_140_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11135 \
    name x_load_166_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_166_cast \
    op interface \
    ports { x_load_166_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11136 \
    name x_load_248_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_248_cast \
    op interface \
    ports { x_load_248_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11137 \
    name x_load_148_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_148_cast \
    op interface \
    ports { x_load_148_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11138 \
    name x_load_154_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_154_cast \
    op interface \
    ports { x_load_154_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11139 \
    name x_load_131_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_131_cast \
    op interface \
    ports { x_load_131_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11140 \
    name sext_ln190_276 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190_276 \
    op interface \
    ports { sext_ln190_276 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11141 \
    name x_load_219_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_219_cast \
    op interface \
    ports { x_load_219_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11142 \
    name x_load_182_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_182_cast \
    op interface \
    ports { x_load_182_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11143 \
    name x_load_163_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_163_cast \
    op interface \
    ports { x_load_163_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11144 \
    name x_load_247_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_247_cast \
    op interface \
    ports { x_load_247_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11145 \
    name x_load_174_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_174_cast \
    op interface \
    ports { x_load_174_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11146 \
    name x_load_137_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_137_cast \
    op interface \
    ports { x_load_137_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11147 \
    name sext_ln190_265 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190_265 \
    op interface \
    ports { sext_ln190_265 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11148 \
    name x_load_198_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_198_cast \
    op interface \
    ports { x_load_198_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11149 \
    name x_load_228_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_228_cast \
    op interface \
    ports { x_load_228_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11150 \
    name x_load_205_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_205_cast \
    op interface \
    ports { x_load_205_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11151 \
    name x_load_202_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_202_cast \
    op interface \
    ports { x_load_202_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11152 \
    name x_load_177_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_177_cast \
    op interface \
    ports { x_load_177_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11153 \
    name x_load_246_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_246_cast \
    op interface \
    ports { x_load_246_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11154 \
    name sext_ln190_273 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190_273 \
    op interface \
    ports { sext_ln190_273 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11155 \
    name x_load_212_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_212_cast \
    op interface \
    ports { x_load_212_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11156 \
    name sext_ln190 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190 \
    op interface \
    ports { sext_ln190 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11157 \
    name x_load_146_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_146_cast \
    op interface \
    ports { x_load_146_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11158 \
    name x_load_214_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_214_cast \
    op interface \
    ports { x_load_214_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11159 \
    name x_load_153_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_153_cast \
    op interface \
    ports { x_load_153_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11160 \
    name x_load_237_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_237_cast \
    op interface \
    ports { x_load_237_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11161 \
    name x_load_210_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_210_cast \
    op interface \
    ports { x_load_210_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11162 \
    name x_load_139_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_139_cast \
    op interface \
    ports { x_load_139_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11163 \
    name x_load_245_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_245_cast \
    op interface \
    ports { x_load_245_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11164 \
    name sext_ln190_268 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190_268 \
    op interface \
    ports { sext_ln190_268 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11165 \
    name x_load_132_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_132_cast \
    op interface \
    ports { x_load_132_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11166 \
    name x_load_169_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_169_cast \
    op interface \
    ports { x_load_169_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11167 \
    name x_load_128_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_128_cast \
    op interface \
    ports { x_load_128_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11168 \
    name x_load_171_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_171_cast \
    op interface \
    ports { x_load_171_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11169 \
    name x_load_216_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_216_cast \
    op interface \
    ports { x_load_216_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11170 \
    name x_load_224_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_224_cast \
    op interface \
    ports { x_load_224_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11171 \
    name x_load_129_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_129_cast \
    op interface \
    ports { x_load_129_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11172 \
    name x_load_221_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_221_cast \
    op interface \
    ports { x_load_221_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11173 \
    name x_load_164_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_164_cast \
    op interface \
    ports { x_load_164_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11174 \
    name x_load_244_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_244_cast \
    op interface \
    ports { x_load_244_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11175 \
    name x_load_135_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_135_cast \
    op interface \
    ports { x_load_135_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11176 \
    name x_load_208_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_208_cast \
    op interface \
    ports { x_load_208_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11177 \
    name x_load_167_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_167_cast \
    op interface \
    ports { x_load_167_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11178 \
    name x_load_138_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_138_cast \
    op interface \
    ports { x_load_138_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11179 \
    name sext_ln190_271 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190_271 \
    op interface \
    ports { sext_ln190_271 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11180 \
    name x_load_191_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_191_cast \
    op interface \
    ports { x_load_191_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11181 \
    name sext_ln190_275 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190_275 \
    op interface \
    ports { sext_ln190_275 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11182 \
    name x_load_231_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_231_cast \
    op interface \
    ports { x_load_231_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11183 \
    name x_load_190_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_190_cast \
    op interface \
    ports { x_load_190_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11184 \
    name x_load_227_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_227_cast \
    op interface \
    ports { x_load_227_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11185 \
    name sext_ln190_270 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190_270 \
    op interface \
    ports { sext_ln190_270 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11186 \
    name x_load_243_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_243_cast \
    op interface \
    ports { x_load_243_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11187 \
    name x_load_152_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_152_cast \
    op interface \
    ports { x_load_152_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11188 \
    name x_load_189_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_189_cast \
    op interface \
    ports { x_load_189_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11189 \
    name x_load_180_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_180_cast \
    op interface \
    ports { x_load_180_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11190 \
    name x_load_193_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_193_cast \
    op interface \
    ports { x_load_193_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11191 \
    name x_load_218_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_218_cast \
    op interface \
    ports { x_load_218_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11192 \
    name x_load_203_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_203_cast \
    op interface \
    ports { x_load_203_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11193 \
    name x_load_176_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_176_cast \
    op interface \
    ports { x_load_176_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11194 \
    name x_load_188_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_188_cast \
    op interface \
    ports { x_load_188_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11195 \
    name x_load_134_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_134_cast \
    op interface \
    ports { x_load_134_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11196 \
    name x_load_173_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_173_cast \
    op interface \
    ports { x_load_173_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11197 \
    name x_load_206_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_206_cast \
    op interface \
    ports { x_load_206_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11198 \
    name x_load_149_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_149_cast \
    op interface \
    ports { x_load_149_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11199 \
    name x_load_242_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_242_cast \
    op interface \
    ports { x_load_242_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11200 \
    name x_load_194_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_194_cast \
    op interface \
    ports { x_load_194_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11201 \
    name sext_ln190_269 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190_269 \
    op interface \
    ports { sext_ln190_269 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11202 \
    name x_load_147_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_147_cast \
    op interface \
    ports { x_load_147_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11203 \
    name x_load_235_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_235_cast \
    op interface \
    ports { x_load_235_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11204 \
    name x_load_143_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_143_cast \
    op interface \
    ports { x_load_143_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11205 \
    name x_load_159_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_159_cast \
    op interface \
    ports { x_load_159_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11206 \
    name x_load_200_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_200_cast \
    op interface \
    ports { x_load_200_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11207 \
    name x_load_158_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_158_cast \
    op interface \
    ports { x_load_158_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11208 \
    name x_load_186_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_186_cast \
    op interface \
    ports { x_load_186_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11209 \
    name x_load_223_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_223_cast \
    op interface \
    ports { x_load_223_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11210 \
    name x_load_230_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_230_cast \
    op interface \
    ports { x_load_230_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11211 \
    name x_load_160_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_160_cast \
    op interface \
    ports { x_load_160_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11212 \
    name x_load_195_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_195_cast \
    op interface \
    ports { x_load_195_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11213 \
    name x_load_241_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_241_cast \
    op interface \
    ports { x_load_241_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11214 \
    name sext_ln190_264 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190_264 \
    op interface \
    ports { sext_ln190_264 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11215 \
    name x_load_165_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_165_cast \
    op interface \
    ports { x_load_165_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11216 \
    name x_load_213_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_213_cast \
    op interface \
    ports { x_load_213_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11217 \
    name x_load_220_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_220_cast \
    op interface \
    ports { x_load_220_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11218 \
    name x_load_226_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_226_cast \
    op interface \
    ports { x_load_226_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11219 \
    name x_load_211_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_211_cast \
    op interface \
    ports { x_load_211_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11220 \
    name x_load_179_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_179_cast \
    op interface \
    ports { x_load_179_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11221 \
    name x_load_157_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_157_cast \
    op interface \
    ports { x_load_157_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11222 \
    name x_load_185_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_185_cast \
    op interface \
    ports { x_load_185_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11223 \
    name x_load_144_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_144_cast \
    op interface \
    ports { x_load_144_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11224 \
    name x_load_215_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_215_cast \
    op interface \
    ports { x_load_215_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11225 \
    name sext_ln190_274 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190_274 \
    op interface \
    ports { sext_ln190_274 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11226 \
    name x_load_161_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_161_cast \
    op interface \
    ports { x_load_161_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11227 \
    name x_load_136_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_136_cast \
    op interface \
    ports { x_load_136_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11228 \
    name x_load_240_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_240_cast \
    op interface \
    ports { x_load_240_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11229 \
    name sext_ln190_272 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190_272 \
    op interface \
    ports { sext_ln190_272 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11230 \
    name sext_ln190_267 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190_267 \
    op interface \
    ports { sext_ln190_267 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11231 \
    name x_load_196_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_196_cast \
    op interface \
    ports { x_load_196_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11232 \
    name x_load_204_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_204_cast \
    op interface \
    ports { x_load_204_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11233 \
    name x_load_175_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_175_cast \
    op interface \
    ports { x_load_175_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11234 \
    name x_load_170_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_170_cast \
    op interface \
    ports { x_load_170_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11235 \
    name x_load_184_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_184_cast \
    op interface \
    ports { x_load_184_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11236 \
    name sext_ln190_266 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190_266 \
    op interface \
    ports { sext_ln190_266 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id -1 \
    name ap_ctrl \
    type ap_ctrl \
    reset_level 1 \
    sync_rst true \
    corename ap_ctrl \
    op interface \
    ports { ap_start { I 1 bit } ap_ready { O 1 bit } ap_done { O 1 bit } ap_idle { O 1 bit } } \
} "
}


# Adapter definition:
set PortName ap_clk
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_clock] == "cg_default_interface_gen_clock"} {
eval "cg_default_interface_gen_clock { \
    id -2 \
    name ${PortName} \
    reset_level 1 \
    sync_rst true \
    corename apif_ap_clk \
    data_wd ${DataWd} \
    op interface \
}"
} else {
puts "@W \[IMPL-113\] Cannot find bus interface model in the library. Ignored generation of bus interface for '${PortName}'"
}
}


# Adapter definition:
set PortName ap_rst
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_reset] == "cg_default_interface_gen_reset"} {
eval "cg_default_interface_gen_reset { \
    id -3 \
    name ${PortName} \
    reset_level 1 \
    sync_rst true \
    corename apif_ap_rst \
    data_wd ${DataWd} \
    op interface \
}"
} else {
puts "@W \[IMPL-114\] Cannot find bus interface model in the library. Ignored generation of bus interface for '${PortName}'"
}
}



# merge
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_end
    cg_default_interface_gen_bundle_end
    AESL_LIB_XILADAPTER::native_axis_end
}


# flow_control definition:
set InstName awn_forward_flow_control_loop_pipe_sequential_init_U
set CompName awn_forward_flow_control_loop_pipe_sequential_init
set name flow_control_loop_pipe_sequential_init
if {${::AESL::PGuard_autocg_gen} && ${::AESL::PGuard_autocg_ipmgen}} {
if {[info proc ::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control] == "::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control"} {
eval "::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control { \
    name ${name} \
    prefix awn_forward_ \
}"
} else {
puts "@W \[IMPL-107\] Cannot find ::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control, check your platform lib"
}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $CompName BINDTYPE interface TYPE internal_upc_flow_control INSTNAME $InstName
}


