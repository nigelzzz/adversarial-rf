# This script segment is generated automatically by AutoPilot

set name awn_forward_mul_8s_7ns_15_1_1
if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $name BINDTYPE {op} TYPE {mul} IMPL {auto} LATENCY 0 ALLOW_PRAGMA 1
}


set id 11260
set name awn_forward_mac_muladd_8s_7ns_15s_16_4_1
set corename simcore_mac
set op mac
set stage_num 4
set clk_width 1
set clk_signed 0
set reset_width 1
set reset_signed 0
set in0_width 8
set in0_signed 1
set in1_width 7
set in1_signed 0
set in2_width 15
set in2_signed 1
set ce_width 1
set ce_signed 0
set out_width 16
set arg_lists {i0 {8 1 +} i1 {7 0 +} m {15 1 +} i2 {15 1 +} p {16 1 +} c_reg {1} rnd {0} acc {0} }
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


set id 11268
set name awn_forward_mac_muladd_7s_7ns_15s_15_4_1
set corename simcore_mac
set op mac
set stage_num 4
set clk_width 1
set clk_signed 0
set reset_width 1
set reset_signed 0
set in0_width 7
set in0_signed 1
set in1_width 7
set in1_signed 0
set in2_width 15
set in2_signed 1
set ce_width 1
set ce_signed 0
set out_width 15
set arg_lists {i0 {7 1 +} i1 {7 0 +} m {14 1 +} i2 {15 1 +} p {15 1 +} c_reg {1} rnd {0} acc {0} }
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
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_0_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_1_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_2_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_3_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_4_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_5_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_6_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_7_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_8_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_9_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_10_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_11_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_12_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_13_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_14_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_15_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_16_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_17_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_18_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_19_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_20_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_21_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_22_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_23_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_24_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_25_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_26_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_27_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_28_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_29_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_30_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_p_ZL4Wse3_31_ROM_AUTO_1R BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
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
    id 11343 \
    name out_r \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename out_r \
    op interface \
    ports { out_r_address0 { O 7 vector } out_r_ce0 { O 1 bit } out_r_we0 { O 1 bit } out_r_d0 { O 20 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'out_r'"
}
}


# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11311 \
    name zext_ln186 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_zext_ln186 \
    op interface \
    ports { zext_ln186 { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11312 \
    name conv12_15_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_15_cast \
    op interface \
    ports { conv12_15_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11313 \
    name conv12_30_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_30_cast \
    op interface \
    ports { conv12_30_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11314 \
    name conv12_17_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_17_cast \
    op interface \
    ports { conv12_17_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11315 \
    name conv12_9_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_9_cast \
    op interface \
    ports { conv12_9_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11316 \
    name conv12_29_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_29_cast \
    op interface \
    ports { conv12_29_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11317 \
    name conv12_24_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_24_cast \
    op interface \
    ports { conv12_24_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11318 \
    name conv12_14_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_14_cast \
    op interface \
    ports { conv12_14_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11319 \
    name conv12_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_cast \
    op interface \
    ports { conv12_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11320 \
    name conv12_28_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_28_cast \
    op interface \
    ports { conv12_28_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11321 \
    name conv12_18_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_18_cast \
    op interface \
    ports { conv12_18_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11322 \
    name conv12_6_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_6_cast \
    op interface \
    ports { conv12_6_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11323 \
    name conv12_13_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_13_cast \
    op interface \
    ports { conv12_13_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11324 \
    name conv12_21_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_21_cast \
    op interface \
    ports { conv12_21_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11325 \
    name conv12_27_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_27_cast \
    op interface \
    ports { conv12_27_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11326 \
    name conv12_4_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_4_cast \
    op interface \
    ports { conv12_4_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11327 \
    name conv12_10_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_10_cast \
    op interface \
    ports { conv12_10_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11328 \
    name conv12_23_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_23_cast \
    op interface \
    ports { conv12_23_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11329 \
    name conv12_3_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_3_cast \
    op interface \
    ports { conv12_3_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11330 \
    name conv12_19_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_19_cast \
    op interface \
    ports { conv12_19_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11331 \
    name conv12_26_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_26_cast \
    op interface \
    ports { conv12_26_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11332 \
    name conv12_12_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_12_cast \
    op interface \
    ports { conv12_12_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11333 \
    name conv12_5_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_5_cast \
    op interface \
    ports { conv12_5_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11334 \
    name conv12_1_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_1_cast \
    op interface \
    ports { conv12_1_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11335 \
    name conv12_2_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_2_cast \
    op interface \
    ports { conv12_2_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11336 \
    name conv12_8_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_8_cast \
    op interface \
    ports { conv12_8_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11337 \
    name conv12_11_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_11_cast \
    op interface \
    ports { conv12_11_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11338 \
    name conv12_25_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_25_cast \
    op interface \
    ports { conv12_25_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11339 \
    name conv12_7_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_7_cast \
    op interface \
    ports { conv12_7_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11340 \
    name conv12_22_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_22_cast \
    op interface \
    ports { conv12_22_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11341 \
    name conv12_20_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_20_cast \
    op interface \
    ports { conv12_20_cast { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11342 \
    name conv12_16_cast \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_conv12_16_cast \
    op interface \
    ports { conv12_16_cast { I 7 vector } } \
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


