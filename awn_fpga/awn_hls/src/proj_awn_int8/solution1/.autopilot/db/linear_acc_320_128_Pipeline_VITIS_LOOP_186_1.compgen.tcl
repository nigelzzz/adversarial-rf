# This script segment is generated automatically by AutoPilot

set id 11432
set name awn_forward_mac_muladd_8s_8s_5s_15_4_1
set corename simcore_mac
set op mac
set stage_num 4
set clk_width 1
set clk_signed 0
set reset_width 1
set reset_signed 0
set in0_width 8
set in0_signed 1
set in1_width 8
set in1_signed 1
set in2_width 5
set in2_signed 1
set ce_width 1
set ce_signed 0
set out_width 15
set arg_lists {i0 {8 1 +} i1 {8 1 +} m {15 1 +} i2 {5 1 +} p {15 1 +} c_reg {1} rnd {0} acc {0} }
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
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_bfc0_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_0_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_1_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_2_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_3_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_4_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_5_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_6_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_7_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_8_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_9_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_10_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_11_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_12_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_13_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_14_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_15_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_16_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_17_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_18_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_19_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_20_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_21_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_22_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_23_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_24_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_25_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_26_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_27_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_28_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_29_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_30_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_31_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_32_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_33_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_34_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_35_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_36_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_37_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_38_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_39_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_40_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_41_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_42_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_43_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_44_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_45_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_46_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_47_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_48_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_49_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_50_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_51_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_52_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_53_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_54_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_55_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_56_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_57_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_58_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_59_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_60_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_61_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_62_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_63_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_64_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_65_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_66_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_67_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_68_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_69_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_70_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_71_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_72_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_73_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_74_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_75_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_76_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_77_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_78_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_79_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_80_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_81_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_82_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_83_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_84_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_85_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_86_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_87_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_88_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_89_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_90_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_91_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_92_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_93_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_94_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_95_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_96_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_97_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_98_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_99_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_100_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_101_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_102_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_103_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_104_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_105_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_106_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_107_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_108_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_109_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_110_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_111_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_112_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_113_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_114_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_115_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_116_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_117_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_118_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_119_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_120_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_121_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_122_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_123_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_124_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_125_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_126_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_p_ZL4Wfc0_127_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
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
    id 11743 \
    name out_r \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename out_r \
    op interface \
    ports { out_r_address0 { O 9 vector } out_r_ce0 { O 1 bit } out_r_we0 { O 1 bit } out_r_d0 { O 22 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'out_r'"
}
}


# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11615 \
    name x_load_72_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_72_cast \
    op interface \
    ports { x_load_72_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11616 \
    name x_load_66_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_66_cast \
    op interface \
    ports { x_load_66_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11617 \
    name x_load_126_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_126_cast \
    op interface \
    ports { x_load_126_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11618 \
    name x_load_7_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_7_cast \
    op interface \
    ports { x_load_7_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11619 \
    name x_load_125_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_125_cast \
    op interface \
    ports { x_load_125_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11620 \
    name x_load_59_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_59_cast \
    op interface \
    ports { x_load_59_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11621 \
    name x_load_39_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_39_cast \
    op interface \
    ports { x_load_39_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11622 \
    name x_load_124_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_124_cast \
    op interface \
    ports { x_load_124_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11623 \
    name x_load_81_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_81_cast \
    op interface \
    ports { x_load_81_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11624 \
    name x_load_45_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_45_cast \
    op interface \
    ports { x_load_45_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11625 \
    name x_load_101_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_101_cast \
    op interface \
    ports { x_load_101_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11626 \
    name x_load_123_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_123_cast \
    op interface \
    ports { x_load_123_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11627 \
    name x_load_67_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_67_cast \
    op interface \
    ports { x_load_67_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11628 \
    name sext_ln190_4 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190_4 \
    op interface \
    ports { sext_ln190_4 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11629 \
    name x_load_111_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_111_cast \
    op interface \
    ports { x_load_111_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11630 \
    name x_load_25_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_25_cast \
    op interface \
    ports { x_load_25_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11631 \
    name x_load_122_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_122_cast \
    op interface \
    ports { x_load_122_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11632 \
    name x_load_89_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_89_cast \
    op interface \
    ports { x_load_89_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11633 \
    name x_load_58_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_58_cast \
    op interface \
    ports { x_load_58_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11634 \
    name x_load_76_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_76_cast \
    op interface \
    ports { x_load_76_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11635 \
    name x_load_8_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_8_cast \
    op interface \
    ports { x_load_8_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11636 \
    name x_load_105_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_105_cast \
    op interface \
    ports { x_load_105_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11637 \
    name x_load_121_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_121_cast \
    op interface \
    ports { x_load_121_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11638 \
    name x_load_12_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_12_cast \
    op interface \
    ports { x_load_12_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11639 \
    name sext_ln190_2 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190_2 \
    op interface \
    ports { sext_ln190_2 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11640 \
    name x_load_6_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_6_cast \
    op interface \
    ports { x_load_6_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11641 \
    name x_load_94_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_94_cast \
    op interface \
    ports { x_load_94_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11642 \
    name x_load_51_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_51_cast \
    op interface \
    ports { x_load_51_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11643 \
    name sext_ln190_8 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190_8 \
    op interface \
    ports { sext_ln190_8 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11644 \
    name x_load_120_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_120_cast \
    op interface \
    ports { x_load_120_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11645 \
    name x_load_68_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_68_cast \
    op interface \
    ports { x_load_68_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11646 \
    name x_load_57_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_57_cast \
    op interface \
    ports { x_load_57_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11647 \
    name x_load_79_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_79_cast \
    op interface \
    ports { x_load_79_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11648 \
    name x_load_73_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_73_cast \
    op interface \
    ports { x_load_73_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11649 \
    name x_load_110_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_110_cast \
    op interface \
    ports { x_load_110_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11650 \
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
    id 11651 \
    name x_load_9_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_9_cast \
    op interface \
    ports { x_load_9_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11652 \
    name x_load_119_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_119_cast \
    op interface \
    ports { x_load_119_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11653 \
    name x_load_24_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_24_cast \
    op interface \
    ports { x_load_24_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11654 \
    name x_load_11_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_11_cast \
    op interface \
    ports { x_load_11_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11655 \
    name x_load_5_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_5_cast \
    op interface \
    ports { x_load_5_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11656 \
    name x_load_91_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_91_cast \
    op interface \
    ports { x_load_91_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11657 \
    name x_load_37_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_37_cast \
    op interface \
    ports { x_load_37_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11658 \
    name x_load_56_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_56_cast \
    op interface \
    ports { x_load_56_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11659 \
    name x_load_47_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_47_cast \
    op interface \
    ports { x_load_47_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11660 \
    name x_load_100_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_100_cast \
    op interface \
    ports { x_load_100_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11661 \
    name x_load_118_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_118_cast \
    op interface \
    ports { x_load_118_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11662 \
    name x_load_21_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_21_cast \
    op interface \
    ports { x_load_21_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11663 \
    name x_load_69_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_69_cast \
    op interface \
    ports { x_load_69_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11664 \
    name x_load_104_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_104_cast \
    op interface \
    ports { x_load_104_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11665 \
    name x_load_31_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_31_cast \
    op interface \
    ports { x_load_31_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11666 \
    name x_load_42_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_42_cast \
    op interface \
    ports { x_load_42_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11667 \
    name x_load_32_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_32_cast \
    op interface \
    ports { x_load_32_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11668 \
    name x_load_30_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_30_cast \
    op interface \
    ports { x_load_30_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11669 \
    name sext_ln190_10 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190_10 \
    op interface \
    ports { sext_ln190_10 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11670 \
    name x_load_50_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_50_cast \
    op interface \
    ports { x_load_50_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11671 \
    name x_load_117_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_117_cast \
    op interface \
    ports { x_load_117_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11672 \
    name x_load_55_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_55_cast \
    op interface \
    ports { x_load_55_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11673 \
    name x_load_40_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_40_cast \
    op interface \
    ports { x_load_40_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11674 \
    name x_load_84_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_84_cast \
    op interface \
    ports { x_load_84_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11675 \
    name x_load_44_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_44_cast \
    op interface \
    ports { x_load_44_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11676 \
    name sext_ln190_7 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190_7 \
    op interface \
    ports { sext_ln190_7 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11677 \
    name x_load_77_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_77_cast \
    op interface \
    ports { x_load_77_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11678 \
    name x_load_1_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_1_cast \
    op interface \
    ports { x_load_1_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11679 \
    name x_load_33_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_33_cast \
    op interface \
    ports { x_load_33_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11680 \
    name x_load_82_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_82_cast \
    op interface \
    ports { x_load_82_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11681 \
    name sext_ln190_6 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190_6 \
    op interface \
    ports { sext_ln190_6 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11682 \
    name x_load_74_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_74_cast \
    op interface \
    ports { x_load_74_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11683 \
    name x_load_116_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_116_cast \
    op interface \
    ports { x_load_116_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11684 \
    name x_load_96_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_96_cast \
    op interface \
    ports { x_load_96_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11685 \
    name x_load_29_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_29_cast \
    op interface \
    ports { x_load_29_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11686 \
    name x_load_19_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_19_cast \
    op interface \
    ports { x_load_19_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11687 \
    name x_load_93_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_93_cast \
    op interface \
    ports { x_load_93_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11688 \
    name x_load_88_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_88_cast \
    op interface \
    ports { x_load_88_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11689 \
    name x_load_70_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_70_cast \
    op interface \
    ports { x_load_70_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11690 \
    name x_load_3_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_3_cast \
    op interface \
    ports { x_load_3_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11691 \
    name x_load_108_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_108_cast \
    op interface \
    ports { x_load_108_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11692 \
    name x_load_28_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_28_cast \
    op interface \
    ports { x_load_28_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11693 \
    name x_load_103_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_103_cast \
    op interface \
    ports { x_load_103_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11694 \
    name x_load_54_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_54_cast \
    op interface \
    ports { x_load_54_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11695 \
    name x_load_115_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_115_cast \
    op interface \
    ports { x_load_115_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11696 \
    name x_load_34_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_34_cast \
    op interface \
    ports { x_load_34_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11697 \
    name x_load_99_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_99_cast \
    op interface \
    ports { x_load_99_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11698 \
    name x_load_4_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_4_cast \
    op interface \
    ports { x_load_4_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11699 \
    name x_load_38_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_38_cast \
    op interface \
    ports { x_load_38_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11700 \
    name x_load_80_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_80_cast \
    op interface \
    ports { x_load_80_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11701 \
    name sext_ln190_5 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190_5 \
    op interface \
    ports { sext_ln190_5 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11702 \
    name x_load_23_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_23_cast \
    op interface \
    ports { x_load_23_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11703 \
    name x_load_49_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_49_cast \
    op interface \
    ports { x_load_49_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11704 \
    name x_load_10_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_10_cast \
    op interface \
    ports { x_load_10_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11705 \
    name sext_ln190_3 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190_3 \
    op interface \
    ports { sext_ln190_3 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11706 \
    name x_load_90_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_90_cast \
    op interface \
    ports { x_load_90_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11707 \
    name x_load_15_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_15_cast \
    op interface \
    ports { x_load_15_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11708 \
    name x_load_114_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_114_cast \
    op interface \
    ports { x_load_114_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11709 \
    name x_load_53_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_53_cast \
    op interface \
    ports { x_load_53_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11710 \
    name sext_ln190_1 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190_1 \
    op interface \
    ports { sext_ln190_1 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11711 \
    name x_load_71_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_71_cast \
    op interface \
    ports { x_load_71_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11712 \
    name x_load_35_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_35_cast \
    op interface \
    ports { x_load_35_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11713 \
    name x_load_107_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_107_cast \
    op interface \
    ports { x_load_107_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11714 \
    name x_load_87_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_87_cast \
    op interface \
    ports { x_load_87_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11715 \
    name x_load_14_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_14_cast \
    op interface \
    ports { x_load_14_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11716 \
    name x_load_2_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_2_cast \
    op interface \
    ports { x_load_2_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11717 \
    name x_load_26_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_26_cast \
    op interface \
    ports { x_load_26_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11718 \
    name x_load_75_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_75_cast \
    op interface \
    ports { x_load_75_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11719 \
    name x_load_41_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_41_cast \
    op interface \
    ports { x_load_41_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11720 \
    name x_load_43_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_43_cast \
    op interface \
    ports { x_load_43_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11721 \
    name x_load_78_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_78_cast \
    op interface \
    ports { x_load_78_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11722 \
    name x_load_102_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_102_cast \
    op interface \
    ports { x_load_102_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11723 \
    name x_load_113_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_113_cast \
    op interface \
    ports { x_load_113_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11724 \
    name x_load_17_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_17_cast \
    op interface \
    ports { x_load_17_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11725 \
    name x_load_95_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_95_cast \
    op interface \
    ports { x_load_95_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11726 \
    name x_load_63_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_63_cast \
    op interface \
    ports { x_load_63_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11727 \
    name x_load_64_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_64_cast \
    op interface \
    ports { x_load_64_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11728 \
    name x_load_62_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_62_cast \
    op interface \
    ports { x_load_62_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11729 \
    name x_load_20_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_20_cast \
    op interface \
    ports { x_load_20_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11730 \
    name x_load_65_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_65_cast \
    op interface \
    ports { x_load_65_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11731 \
    name x_load_22_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_22_cast \
    op interface \
    ports { x_load_22_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11732 \
    name x_load_98_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_98_cast \
    op interface \
    ports { x_load_98_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11733 \
    name x_load_61_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_61_cast \
    op interface \
    ports { x_load_61_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11734 \
    name x_load_92_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_92_cast \
    op interface \
    ports { x_load_92_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11735 \
    name x_load_13_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_13_cast \
    op interface \
    ports { x_load_13_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11736 \
    name x_load_52_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_52_cast \
    op interface \
    ports { x_load_52_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11737 \
    name sext_ln190_9 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190_9 \
    op interface \
    ports { sext_ln190_9 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11738 \
    name x_load_85_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_85_cast \
    op interface \
    ports { x_load_85_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11739 \
    name x_load_112_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_112_cast \
    op interface \
    ports { x_load_112_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11740 \
    name sext_ln190_11 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln190_11 \
    op interface \
    ports { sext_ln190_11 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11741 \
    name x_load_60_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_60_cast \
    op interface \
    ports { x_load_60_cast { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11742 \
    name x_load_83_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_x_load_83_cast \
    op interface \
    ports { x_load_83_cast { I 8 vector } } \
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


