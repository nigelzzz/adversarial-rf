# This script segment is generated automatically by AutoPilot

set name awn_forward_mul_8s_32ns_38_1_1
if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $name BINDTYPE {op} TYPE {mul} IMPL {auto} LATENCY 0 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler awn_forward_sparsemux_129_6_8_1_1 BINDTYPE {op} TYPE {sparsemux} IMPL {auto}
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
    id 6487 \
    name d_q \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q \
    op interface \
    ports { d_q_address0 { O 6 vector } d_q_ce0 { O 1 bit } d_q_we0 { O 1 bit } d_q_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6488 \
    name d_q_1 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_1 \
    op interface \
    ports { d_q_1_address0 { O 6 vector } d_q_1_ce0 { O 1 bit } d_q_1_we0 { O 1 bit } d_q_1_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_1'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6489 \
    name d_q_2 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_2 \
    op interface \
    ports { d_q_2_address0 { O 6 vector } d_q_2_ce0 { O 1 bit } d_q_2_we0 { O 1 bit } d_q_2_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_2'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6490 \
    name d_q_3 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_3 \
    op interface \
    ports { d_q_3_address0 { O 6 vector } d_q_3_ce0 { O 1 bit } d_q_3_we0 { O 1 bit } d_q_3_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_3'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6491 \
    name d_q_4 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_4 \
    op interface \
    ports { d_q_4_address0 { O 6 vector } d_q_4_ce0 { O 1 bit } d_q_4_we0 { O 1 bit } d_q_4_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_4'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6492 \
    name d_q_5 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_5 \
    op interface \
    ports { d_q_5_address0 { O 6 vector } d_q_5_ce0 { O 1 bit } d_q_5_we0 { O 1 bit } d_q_5_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_5'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6493 \
    name d_q_6 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_6 \
    op interface \
    ports { d_q_6_address0 { O 6 vector } d_q_6_ce0 { O 1 bit } d_q_6_we0 { O 1 bit } d_q_6_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_6'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6494 \
    name d_q_7 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_7 \
    op interface \
    ports { d_q_7_address0 { O 6 vector } d_q_7_ce0 { O 1 bit } d_q_7_we0 { O 1 bit } d_q_7_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_7'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6495 \
    name d_q_8 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_8 \
    op interface \
    ports { d_q_8_address0 { O 6 vector } d_q_8_ce0 { O 1 bit } d_q_8_we0 { O 1 bit } d_q_8_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_8'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6496 \
    name d_q_9 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_9 \
    op interface \
    ports { d_q_9_address0 { O 6 vector } d_q_9_ce0 { O 1 bit } d_q_9_we0 { O 1 bit } d_q_9_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_9'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6497 \
    name d_q_10 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_10 \
    op interface \
    ports { d_q_10_address0 { O 6 vector } d_q_10_ce0 { O 1 bit } d_q_10_we0 { O 1 bit } d_q_10_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_10'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6498 \
    name d_q_11 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_11 \
    op interface \
    ports { d_q_11_address0 { O 6 vector } d_q_11_ce0 { O 1 bit } d_q_11_we0 { O 1 bit } d_q_11_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_11'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6499 \
    name d_q_12 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_12 \
    op interface \
    ports { d_q_12_address0 { O 6 vector } d_q_12_ce0 { O 1 bit } d_q_12_we0 { O 1 bit } d_q_12_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_12'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6500 \
    name d_q_13 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_13 \
    op interface \
    ports { d_q_13_address0 { O 6 vector } d_q_13_ce0 { O 1 bit } d_q_13_we0 { O 1 bit } d_q_13_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_13'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6501 \
    name d_q_14 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_14 \
    op interface \
    ports { d_q_14_address0 { O 6 vector } d_q_14_ce0 { O 1 bit } d_q_14_we0 { O 1 bit } d_q_14_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_14'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6502 \
    name d_q_15 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_15 \
    op interface \
    ports { d_q_15_address0 { O 6 vector } d_q_15_ce0 { O 1 bit } d_q_15_we0 { O 1 bit } d_q_15_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_15'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6503 \
    name d_q_16 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_16 \
    op interface \
    ports { d_q_16_address0 { O 6 vector } d_q_16_ce0 { O 1 bit } d_q_16_we0 { O 1 bit } d_q_16_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_16'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6504 \
    name d_q_17 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_17 \
    op interface \
    ports { d_q_17_address0 { O 6 vector } d_q_17_ce0 { O 1 bit } d_q_17_we0 { O 1 bit } d_q_17_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_17'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6505 \
    name d_q_18 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_18 \
    op interface \
    ports { d_q_18_address0 { O 6 vector } d_q_18_ce0 { O 1 bit } d_q_18_we0 { O 1 bit } d_q_18_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_18'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6506 \
    name d_q_19 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_19 \
    op interface \
    ports { d_q_19_address0 { O 6 vector } d_q_19_ce0 { O 1 bit } d_q_19_we0 { O 1 bit } d_q_19_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_19'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6507 \
    name d_q_20 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_20 \
    op interface \
    ports { d_q_20_address0 { O 6 vector } d_q_20_ce0 { O 1 bit } d_q_20_we0 { O 1 bit } d_q_20_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_20'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6508 \
    name d_q_21 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_21 \
    op interface \
    ports { d_q_21_address0 { O 6 vector } d_q_21_ce0 { O 1 bit } d_q_21_we0 { O 1 bit } d_q_21_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_21'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6509 \
    name d_q_22 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_22 \
    op interface \
    ports { d_q_22_address0 { O 6 vector } d_q_22_ce0 { O 1 bit } d_q_22_we0 { O 1 bit } d_q_22_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_22'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6510 \
    name d_q_23 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_23 \
    op interface \
    ports { d_q_23_address0 { O 6 vector } d_q_23_ce0 { O 1 bit } d_q_23_we0 { O 1 bit } d_q_23_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_23'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6511 \
    name d_q_24 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_24 \
    op interface \
    ports { d_q_24_address0 { O 6 vector } d_q_24_ce0 { O 1 bit } d_q_24_we0 { O 1 bit } d_q_24_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_24'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6512 \
    name d_q_25 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_25 \
    op interface \
    ports { d_q_25_address0 { O 6 vector } d_q_25_ce0 { O 1 bit } d_q_25_we0 { O 1 bit } d_q_25_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_25'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6513 \
    name d_q_26 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_26 \
    op interface \
    ports { d_q_26_address0 { O 6 vector } d_q_26_ce0 { O 1 bit } d_q_26_we0 { O 1 bit } d_q_26_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_26'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6514 \
    name d_q_27 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_27 \
    op interface \
    ports { d_q_27_address0 { O 6 vector } d_q_27_ce0 { O 1 bit } d_q_27_we0 { O 1 bit } d_q_27_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_27'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6515 \
    name d_q_28 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_28 \
    op interface \
    ports { d_q_28_address0 { O 6 vector } d_q_28_ce0 { O 1 bit } d_q_28_we0 { O 1 bit } d_q_28_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_28'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6516 \
    name d_q_29 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_29 \
    op interface \
    ports { d_q_29_address0 { O 6 vector } d_q_29_ce0 { O 1 bit } d_q_29_we0 { O 1 bit } d_q_29_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_29'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6517 \
    name d_q_30 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_30 \
    op interface \
    ports { d_q_30_address0 { O 6 vector } d_q_30_ce0 { O 1 bit } d_q_30_we0 { O 1 bit } d_q_30_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_30'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6518 \
    name d_q_31 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_31 \
    op interface \
    ports { d_q_31_address0 { O 6 vector } d_q_31_ce0 { O 1 bit } d_q_31_we0 { O 1 bit } d_q_31_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_31'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6519 \
    name d_q_32 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_32 \
    op interface \
    ports { d_q_32_address0 { O 6 vector } d_q_32_ce0 { O 1 bit } d_q_32_we0 { O 1 bit } d_q_32_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_32'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6520 \
    name d_q_33 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_33 \
    op interface \
    ports { d_q_33_address0 { O 6 vector } d_q_33_ce0 { O 1 bit } d_q_33_we0 { O 1 bit } d_q_33_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_33'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6521 \
    name d_q_34 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_34 \
    op interface \
    ports { d_q_34_address0 { O 6 vector } d_q_34_ce0 { O 1 bit } d_q_34_we0 { O 1 bit } d_q_34_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_34'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6522 \
    name d_q_35 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_35 \
    op interface \
    ports { d_q_35_address0 { O 6 vector } d_q_35_ce0 { O 1 bit } d_q_35_we0 { O 1 bit } d_q_35_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_35'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6523 \
    name d_q_36 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_36 \
    op interface \
    ports { d_q_36_address0 { O 6 vector } d_q_36_ce0 { O 1 bit } d_q_36_we0 { O 1 bit } d_q_36_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_36'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6524 \
    name d_q_37 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_37 \
    op interface \
    ports { d_q_37_address0 { O 6 vector } d_q_37_ce0 { O 1 bit } d_q_37_we0 { O 1 bit } d_q_37_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_37'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6525 \
    name d_q_38 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_38 \
    op interface \
    ports { d_q_38_address0 { O 6 vector } d_q_38_ce0 { O 1 bit } d_q_38_we0 { O 1 bit } d_q_38_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_38'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6526 \
    name d_q_39 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_39 \
    op interface \
    ports { d_q_39_address0 { O 6 vector } d_q_39_ce0 { O 1 bit } d_q_39_we0 { O 1 bit } d_q_39_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_39'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6527 \
    name d_q_40 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_40 \
    op interface \
    ports { d_q_40_address0 { O 6 vector } d_q_40_ce0 { O 1 bit } d_q_40_we0 { O 1 bit } d_q_40_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_40'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6528 \
    name d_q_41 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_41 \
    op interface \
    ports { d_q_41_address0 { O 6 vector } d_q_41_ce0 { O 1 bit } d_q_41_we0 { O 1 bit } d_q_41_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_41'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6529 \
    name d_q_42 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_42 \
    op interface \
    ports { d_q_42_address0 { O 6 vector } d_q_42_ce0 { O 1 bit } d_q_42_we0 { O 1 bit } d_q_42_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_42'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6530 \
    name d_q_43 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_43 \
    op interface \
    ports { d_q_43_address0 { O 6 vector } d_q_43_ce0 { O 1 bit } d_q_43_we0 { O 1 bit } d_q_43_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_43'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6531 \
    name d_q_44 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_44 \
    op interface \
    ports { d_q_44_address0 { O 6 vector } d_q_44_ce0 { O 1 bit } d_q_44_we0 { O 1 bit } d_q_44_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_44'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6532 \
    name d_q_45 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_45 \
    op interface \
    ports { d_q_45_address0 { O 6 vector } d_q_45_ce0 { O 1 bit } d_q_45_we0 { O 1 bit } d_q_45_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_45'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6533 \
    name d_q_46 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_46 \
    op interface \
    ports { d_q_46_address0 { O 6 vector } d_q_46_ce0 { O 1 bit } d_q_46_we0 { O 1 bit } d_q_46_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_46'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6534 \
    name d_q_47 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_47 \
    op interface \
    ports { d_q_47_address0 { O 6 vector } d_q_47_ce0 { O 1 bit } d_q_47_we0 { O 1 bit } d_q_47_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_47'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6535 \
    name d_q_48 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_48 \
    op interface \
    ports { d_q_48_address0 { O 6 vector } d_q_48_ce0 { O 1 bit } d_q_48_we0 { O 1 bit } d_q_48_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_48'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6536 \
    name d_q_49 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_49 \
    op interface \
    ports { d_q_49_address0 { O 6 vector } d_q_49_ce0 { O 1 bit } d_q_49_we0 { O 1 bit } d_q_49_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_49'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6537 \
    name d_q_50 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_50 \
    op interface \
    ports { d_q_50_address0 { O 6 vector } d_q_50_ce0 { O 1 bit } d_q_50_we0 { O 1 bit } d_q_50_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_50'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6538 \
    name d_q_51 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_51 \
    op interface \
    ports { d_q_51_address0 { O 6 vector } d_q_51_ce0 { O 1 bit } d_q_51_we0 { O 1 bit } d_q_51_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_51'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6539 \
    name d_q_52 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_52 \
    op interface \
    ports { d_q_52_address0 { O 6 vector } d_q_52_ce0 { O 1 bit } d_q_52_we0 { O 1 bit } d_q_52_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_52'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6540 \
    name d_q_53 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_53 \
    op interface \
    ports { d_q_53_address0 { O 6 vector } d_q_53_ce0 { O 1 bit } d_q_53_we0 { O 1 bit } d_q_53_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_53'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6541 \
    name d_q_54 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_54 \
    op interface \
    ports { d_q_54_address0 { O 6 vector } d_q_54_ce0 { O 1 bit } d_q_54_we0 { O 1 bit } d_q_54_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_54'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6542 \
    name d_q_55 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_55 \
    op interface \
    ports { d_q_55_address0 { O 6 vector } d_q_55_ce0 { O 1 bit } d_q_55_we0 { O 1 bit } d_q_55_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_55'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6543 \
    name d_q_56 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_56 \
    op interface \
    ports { d_q_56_address0 { O 6 vector } d_q_56_ce0 { O 1 bit } d_q_56_we0 { O 1 bit } d_q_56_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_56'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6544 \
    name d_q_57 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_57 \
    op interface \
    ports { d_q_57_address0 { O 6 vector } d_q_57_ce0 { O 1 bit } d_q_57_we0 { O 1 bit } d_q_57_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_57'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6545 \
    name d_q_58 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_58 \
    op interface \
    ports { d_q_58_address0 { O 6 vector } d_q_58_ce0 { O 1 bit } d_q_58_we0 { O 1 bit } d_q_58_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_58'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6546 \
    name d_q_59 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_59 \
    op interface \
    ports { d_q_59_address0 { O 6 vector } d_q_59_ce0 { O 1 bit } d_q_59_we0 { O 1 bit } d_q_59_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_59'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6547 \
    name d_q_60 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_60 \
    op interface \
    ports { d_q_60_address0 { O 6 vector } d_q_60_ce0 { O 1 bit } d_q_60_we0 { O 1 bit } d_q_60_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_60'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6548 \
    name d_q_61 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_61 \
    op interface \
    ports { d_q_61_address0 { O 6 vector } d_q_61_ce0 { O 1 bit } d_q_61_we0 { O 1 bit } d_q_61_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_61'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6549 \
    name d_q_62 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_62 \
    op interface \
    ports { d_q_62_address0 { O 6 vector } d_q_62_ce0 { O 1 bit } d_q_62_we0 { O 1 bit } d_q_62_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_62'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6550 \
    name d_q_63 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename d_q_63 \
    op interface \
    ports { d_q_63_address0 { O 6 vector } d_q_63_ce0 { O 1 bit } d_q_63_we0 { O 1 bit } d_q_63_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'd_q_63'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6551 \
    name p_tanh \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename p_tanh \
    op interface \
    ports { p_tanh_address0 { O 12 vector } p_tanh_ce0 { O 1 bit } p_tanh_q0 { I 1 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'p_tanh'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6552 \
    name odd_q \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q \
    op interface \
    ports { odd_q_address0 { O 6 vector } odd_q_ce0 { O 1 bit } odd_q_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6553 \
    name odd_q_1 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_1 \
    op interface \
    ports { odd_q_1_address0 { O 6 vector } odd_q_1_ce0 { O 1 bit } odd_q_1_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_1'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6554 \
    name odd_q_2 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_2 \
    op interface \
    ports { odd_q_2_address0 { O 6 vector } odd_q_2_ce0 { O 1 bit } odd_q_2_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_2'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6555 \
    name odd_q_3 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_3 \
    op interface \
    ports { odd_q_3_address0 { O 6 vector } odd_q_3_ce0 { O 1 bit } odd_q_3_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_3'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6556 \
    name odd_q_4 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_4 \
    op interface \
    ports { odd_q_4_address0 { O 6 vector } odd_q_4_ce0 { O 1 bit } odd_q_4_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_4'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6557 \
    name odd_q_5 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_5 \
    op interface \
    ports { odd_q_5_address0 { O 6 vector } odd_q_5_ce0 { O 1 bit } odd_q_5_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_5'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6558 \
    name odd_q_6 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_6 \
    op interface \
    ports { odd_q_6_address0 { O 6 vector } odd_q_6_ce0 { O 1 bit } odd_q_6_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_6'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6559 \
    name odd_q_7 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_7 \
    op interface \
    ports { odd_q_7_address0 { O 6 vector } odd_q_7_ce0 { O 1 bit } odd_q_7_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_7'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6560 \
    name odd_q_8 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_8 \
    op interface \
    ports { odd_q_8_address0 { O 6 vector } odd_q_8_ce0 { O 1 bit } odd_q_8_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_8'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6561 \
    name odd_q_9 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_9 \
    op interface \
    ports { odd_q_9_address0 { O 6 vector } odd_q_9_ce0 { O 1 bit } odd_q_9_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_9'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6562 \
    name odd_q_10 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_10 \
    op interface \
    ports { odd_q_10_address0 { O 6 vector } odd_q_10_ce0 { O 1 bit } odd_q_10_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_10'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6563 \
    name odd_q_11 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_11 \
    op interface \
    ports { odd_q_11_address0 { O 6 vector } odd_q_11_ce0 { O 1 bit } odd_q_11_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_11'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6564 \
    name odd_q_12 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_12 \
    op interface \
    ports { odd_q_12_address0 { O 6 vector } odd_q_12_ce0 { O 1 bit } odd_q_12_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_12'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6565 \
    name odd_q_13 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_13 \
    op interface \
    ports { odd_q_13_address0 { O 6 vector } odd_q_13_ce0 { O 1 bit } odd_q_13_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_13'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6566 \
    name odd_q_14 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_14 \
    op interface \
    ports { odd_q_14_address0 { O 6 vector } odd_q_14_ce0 { O 1 bit } odd_q_14_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_14'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6567 \
    name odd_q_15 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_15 \
    op interface \
    ports { odd_q_15_address0 { O 6 vector } odd_q_15_ce0 { O 1 bit } odd_q_15_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_15'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6568 \
    name odd_q_16 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_16 \
    op interface \
    ports { odd_q_16_address0 { O 6 vector } odd_q_16_ce0 { O 1 bit } odd_q_16_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_16'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6569 \
    name odd_q_17 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_17 \
    op interface \
    ports { odd_q_17_address0 { O 6 vector } odd_q_17_ce0 { O 1 bit } odd_q_17_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_17'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6570 \
    name odd_q_18 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_18 \
    op interface \
    ports { odd_q_18_address0 { O 6 vector } odd_q_18_ce0 { O 1 bit } odd_q_18_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_18'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6571 \
    name odd_q_19 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_19 \
    op interface \
    ports { odd_q_19_address0 { O 6 vector } odd_q_19_ce0 { O 1 bit } odd_q_19_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_19'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6572 \
    name odd_q_20 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_20 \
    op interface \
    ports { odd_q_20_address0 { O 6 vector } odd_q_20_ce0 { O 1 bit } odd_q_20_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_20'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6573 \
    name odd_q_21 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_21 \
    op interface \
    ports { odd_q_21_address0 { O 6 vector } odd_q_21_ce0 { O 1 bit } odd_q_21_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_21'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6574 \
    name odd_q_22 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_22 \
    op interface \
    ports { odd_q_22_address0 { O 6 vector } odd_q_22_ce0 { O 1 bit } odd_q_22_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_22'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6575 \
    name odd_q_23 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_23 \
    op interface \
    ports { odd_q_23_address0 { O 6 vector } odd_q_23_ce0 { O 1 bit } odd_q_23_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_23'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6576 \
    name odd_q_24 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_24 \
    op interface \
    ports { odd_q_24_address0 { O 6 vector } odd_q_24_ce0 { O 1 bit } odd_q_24_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_24'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6577 \
    name odd_q_25 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_25 \
    op interface \
    ports { odd_q_25_address0 { O 6 vector } odd_q_25_ce0 { O 1 bit } odd_q_25_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_25'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6578 \
    name odd_q_26 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_26 \
    op interface \
    ports { odd_q_26_address0 { O 6 vector } odd_q_26_ce0 { O 1 bit } odd_q_26_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_26'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6579 \
    name odd_q_27 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_27 \
    op interface \
    ports { odd_q_27_address0 { O 6 vector } odd_q_27_ce0 { O 1 bit } odd_q_27_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_27'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6580 \
    name odd_q_28 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_28 \
    op interface \
    ports { odd_q_28_address0 { O 6 vector } odd_q_28_ce0 { O 1 bit } odd_q_28_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_28'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6581 \
    name odd_q_29 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_29 \
    op interface \
    ports { odd_q_29_address0 { O 6 vector } odd_q_29_ce0 { O 1 bit } odd_q_29_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_29'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6582 \
    name odd_q_30 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_30 \
    op interface \
    ports { odd_q_30_address0 { O 6 vector } odd_q_30_ce0 { O 1 bit } odd_q_30_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_30'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6583 \
    name odd_q_31 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_31 \
    op interface \
    ports { odd_q_31_address0 { O 6 vector } odd_q_31_ce0 { O 1 bit } odd_q_31_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_31'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6584 \
    name odd_q_32 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_32 \
    op interface \
    ports { odd_q_32_address0 { O 6 vector } odd_q_32_ce0 { O 1 bit } odd_q_32_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_32'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6585 \
    name odd_q_33 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_33 \
    op interface \
    ports { odd_q_33_address0 { O 6 vector } odd_q_33_ce0 { O 1 bit } odd_q_33_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_33'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6586 \
    name odd_q_34 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_34 \
    op interface \
    ports { odd_q_34_address0 { O 6 vector } odd_q_34_ce0 { O 1 bit } odd_q_34_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_34'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6587 \
    name odd_q_35 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_35 \
    op interface \
    ports { odd_q_35_address0 { O 6 vector } odd_q_35_ce0 { O 1 bit } odd_q_35_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_35'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6588 \
    name odd_q_36 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_36 \
    op interface \
    ports { odd_q_36_address0 { O 6 vector } odd_q_36_ce0 { O 1 bit } odd_q_36_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_36'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6589 \
    name odd_q_37 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_37 \
    op interface \
    ports { odd_q_37_address0 { O 6 vector } odd_q_37_ce0 { O 1 bit } odd_q_37_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_37'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6590 \
    name odd_q_38 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_38 \
    op interface \
    ports { odd_q_38_address0 { O 6 vector } odd_q_38_ce0 { O 1 bit } odd_q_38_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_38'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6591 \
    name odd_q_39 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_39 \
    op interface \
    ports { odd_q_39_address0 { O 6 vector } odd_q_39_ce0 { O 1 bit } odd_q_39_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_39'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6592 \
    name odd_q_40 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_40 \
    op interface \
    ports { odd_q_40_address0 { O 6 vector } odd_q_40_ce0 { O 1 bit } odd_q_40_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_40'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6593 \
    name odd_q_41 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_41 \
    op interface \
    ports { odd_q_41_address0 { O 6 vector } odd_q_41_ce0 { O 1 bit } odd_q_41_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_41'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6594 \
    name odd_q_42 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_42 \
    op interface \
    ports { odd_q_42_address0 { O 6 vector } odd_q_42_ce0 { O 1 bit } odd_q_42_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_42'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6595 \
    name odd_q_43 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_43 \
    op interface \
    ports { odd_q_43_address0 { O 6 vector } odd_q_43_ce0 { O 1 bit } odd_q_43_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_43'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6596 \
    name odd_q_44 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_44 \
    op interface \
    ports { odd_q_44_address0 { O 6 vector } odd_q_44_ce0 { O 1 bit } odd_q_44_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_44'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6597 \
    name odd_q_45 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_45 \
    op interface \
    ports { odd_q_45_address0 { O 6 vector } odd_q_45_ce0 { O 1 bit } odd_q_45_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_45'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6598 \
    name odd_q_46 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_46 \
    op interface \
    ports { odd_q_46_address0 { O 6 vector } odd_q_46_ce0 { O 1 bit } odd_q_46_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_46'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6599 \
    name odd_q_47 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_47 \
    op interface \
    ports { odd_q_47_address0 { O 6 vector } odd_q_47_ce0 { O 1 bit } odd_q_47_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_47'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6600 \
    name odd_q_48 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_48 \
    op interface \
    ports { odd_q_48_address0 { O 6 vector } odd_q_48_ce0 { O 1 bit } odd_q_48_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_48'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6601 \
    name odd_q_49 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_49 \
    op interface \
    ports { odd_q_49_address0 { O 6 vector } odd_q_49_ce0 { O 1 bit } odd_q_49_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_49'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6602 \
    name odd_q_50 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_50 \
    op interface \
    ports { odd_q_50_address0 { O 6 vector } odd_q_50_ce0 { O 1 bit } odd_q_50_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_50'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6603 \
    name odd_q_51 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_51 \
    op interface \
    ports { odd_q_51_address0 { O 6 vector } odd_q_51_ce0 { O 1 bit } odd_q_51_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_51'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6604 \
    name odd_q_52 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_52 \
    op interface \
    ports { odd_q_52_address0 { O 6 vector } odd_q_52_ce0 { O 1 bit } odd_q_52_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_52'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6605 \
    name odd_q_53 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_53 \
    op interface \
    ports { odd_q_53_address0 { O 6 vector } odd_q_53_ce0 { O 1 bit } odd_q_53_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_53'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6606 \
    name odd_q_54 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_54 \
    op interface \
    ports { odd_q_54_address0 { O 6 vector } odd_q_54_ce0 { O 1 bit } odd_q_54_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_54'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6607 \
    name odd_q_55 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_55 \
    op interface \
    ports { odd_q_55_address0 { O 6 vector } odd_q_55_ce0 { O 1 bit } odd_q_55_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_55'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6608 \
    name odd_q_56 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_56 \
    op interface \
    ports { odd_q_56_address0 { O 6 vector } odd_q_56_ce0 { O 1 bit } odd_q_56_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_56'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6609 \
    name odd_q_57 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_57 \
    op interface \
    ports { odd_q_57_address0 { O 6 vector } odd_q_57_ce0 { O 1 bit } odd_q_57_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_57'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6610 \
    name odd_q_58 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_58 \
    op interface \
    ports { odd_q_58_address0 { O 6 vector } odd_q_58_ce0 { O 1 bit } odd_q_58_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_58'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6611 \
    name odd_q_59 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_59 \
    op interface \
    ports { odd_q_59_address0 { O 6 vector } odd_q_59_ce0 { O 1 bit } odd_q_59_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_59'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6612 \
    name odd_q_60 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_60 \
    op interface \
    ports { odd_q_60_address0 { O 6 vector } odd_q_60_ce0 { O 1 bit } odd_q_60_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_60'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6613 \
    name odd_q_61 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_61 \
    op interface \
    ports { odd_q_61_address0 { O 6 vector } odd_q_61_ce0 { O 1 bit } odd_q_61_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_61'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6614 \
    name odd_q_62 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_62 \
    op interface \
    ports { odd_q_62_address0 { O 6 vector } odd_q_62_ce0 { O 1 bit } odd_q_62_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_62'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 6615 \
    name odd_q_63 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename odd_q_63 \
    op interface \
    ports { odd_q_63_address0 { O 6 vector } odd_q_63_ce0 { O 1 bit } odd_q_63_q0 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_63'"
}
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


