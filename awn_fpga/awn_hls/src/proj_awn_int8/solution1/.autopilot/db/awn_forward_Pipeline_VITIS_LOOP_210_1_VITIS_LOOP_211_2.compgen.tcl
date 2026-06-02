# This script segment is generated automatically by AutoPilot

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
    id 2311 \
    name even_q \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename even_q \
    op interface \
    ports { even_q_address0 { O 12 vector } even_q_ce0 { O 1 bit } even_q_we0 { O 1 bit } even_q_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'even_q'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2312 \
    name odd_q_63 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_63 \
    op interface \
    ports { odd_q_63_address0 { O 6 vector } odd_q_63_ce0 { O 1 bit } odd_q_63_we0 { O 1 bit } odd_q_63_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_63'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2313 \
    name odd_q_62 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_62 \
    op interface \
    ports { odd_q_62_address0 { O 6 vector } odd_q_62_ce0 { O 1 bit } odd_q_62_we0 { O 1 bit } odd_q_62_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_62'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2314 \
    name odd_q_61 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_61 \
    op interface \
    ports { odd_q_61_address0 { O 6 vector } odd_q_61_ce0 { O 1 bit } odd_q_61_we0 { O 1 bit } odd_q_61_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_61'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2315 \
    name odd_q_60 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_60 \
    op interface \
    ports { odd_q_60_address0 { O 6 vector } odd_q_60_ce0 { O 1 bit } odd_q_60_we0 { O 1 bit } odd_q_60_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_60'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2316 \
    name odd_q_59 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_59 \
    op interface \
    ports { odd_q_59_address0 { O 6 vector } odd_q_59_ce0 { O 1 bit } odd_q_59_we0 { O 1 bit } odd_q_59_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_59'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2317 \
    name odd_q_58 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_58 \
    op interface \
    ports { odd_q_58_address0 { O 6 vector } odd_q_58_ce0 { O 1 bit } odd_q_58_we0 { O 1 bit } odd_q_58_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_58'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2318 \
    name odd_q_57 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_57 \
    op interface \
    ports { odd_q_57_address0 { O 6 vector } odd_q_57_ce0 { O 1 bit } odd_q_57_we0 { O 1 bit } odd_q_57_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_57'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2319 \
    name odd_q_56 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_56 \
    op interface \
    ports { odd_q_56_address0 { O 6 vector } odd_q_56_ce0 { O 1 bit } odd_q_56_we0 { O 1 bit } odd_q_56_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_56'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2320 \
    name odd_q_55 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_55 \
    op interface \
    ports { odd_q_55_address0 { O 6 vector } odd_q_55_ce0 { O 1 bit } odd_q_55_we0 { O 1 bit } odd_q_55_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_55'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2321 \
    name odd_q_54 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_54 \
    op interface \
    ports { odd_q_54_address0 { O 6 vector } odd_q_54_ce0 { O 1 bit } odd_q_54_we0 { O 1 bit } odd_q_54_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_54'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2322 \
    name odd_q_53 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_53 \
    op interface \
    ports { odd_q_53_address0 { O 6 vector } odd_q_53_ce0 { O 1 bit } odd_q_53_we0 { O 1 bit } odd_q_53_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_53'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2323 \
    name odd_q_52 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_52 \
    op interface \
    ports { odd_q_52_address0 { O 6 vector } odd_q_52_ce0 { O 1 bit } odd_q_52_we0 { O 1 bit } odd_q_52_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_52'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2324 \
    name odd_q_51 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_51 \
    op interface \
    ports { odd_q_51_address0 { O 6 vector } odd_q_51_ce0 { O 1 bit } odd_q_51_we0 { O 1 bit } odd_q_51_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_51'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2325 \
    name odd_q_50 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_50 \
    op interface \
    ports { odd_q_50_address0 { O 6 vector } odd_q_50_ce0 { O 1 bit } odd_q_50_we0 { O 1 bit } odd_q_50_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_50'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2326 \
    name odd_q_49 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_49 \
    op interface \
    ports { odd_q_49_address0 { O 6 vector } odd_q_49_ce0 { O 1 bit } odd_q_49_we0 { O 1 bit } odd_q_49_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_49'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2327 \
    name odd_q_48 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_48 \
    op interface \
    ports { odd_q_48_address0 { O 6 vector } odd_q_48_ce0 { O 1 bit } odd_q_48_we0 { O 1 bit } odd_q_48_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_48'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2328 \
    name odd_q_47 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_47 \
    op interface \
    ports { odd_q_47_address0 { O 6 vector } odd_q_47_ce0 { O 1 bit } odd_q_47_we0 { O 1 bit } odd_q_47_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_47'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2329 \
    name odd_q_46 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_46 \
    op interface \
    ports { odd_q_46_address0 { O 6 vector } odd_q_46_ce0 { O 1 bit } odd_q_46_we0 { O 1 bit } odd_q_46_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_46'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2330 \
    name odd_q_45 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_45 \
    op interface \
    ports { odd_q_45_address0 { O 6 vector } odd_q_45_ce0 { O 1 bit } odd_q_45_we0 { O 1 bit } odd_q_45_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_45'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2331 \
    name odd_q_44 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_44 \
    op interface \
    ports { odd_q_44_address0 { O 6 vector } odd_q_44_ce0 { O 1 bit } odd_q_44_we0 { O 1 bit } odd_q_44_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_44'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2332 \
    name odd_q_43 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_43 \
    op interface \
    ports { odd_q_43_address0 { O 6 vector } odd_q_43_ce0 { O 1 bit } odd_q_43_we0 { O 1 bit } odd_q_43_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_43'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2333 \
    name odd_q_42 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_42 \
    op interface \
    ports { odd_q_42_address0 { O 6 vector } odd_q_42_ce0 { O 1 bit } odd_q_42_we0 { O 1 bit } odd_q_42_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_42'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2334 \
    name odd_q_41 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_41 \
    op interface \
    ports { odd_q_41_address0 { O 6 vector } odd_q_41_ce0 { O 1 bit } odd_q_41_we0 { O 1 bit } odd_q_41_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_41'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2335 \
    name odd_q_40 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_40 \
    op interface \
    ports { odd_q_40_address0 { O 6 vector } odd_q_40_ce0 { O 1 bit } odd_q_40_we0 { O 1 bit } odd_q_40_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_40'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2336 \
    name odd_q_39 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_39 \
    op interface \
    ports { odd_q_39_address0 { O 6 vector } odd_q_39_ce0 { O 1 bit } odd_q_39_we0 { O 1 bit } odd_q_39_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_39'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2337 \
    name odd_q_38 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_38 \
    op interface \
    ports { odd_q_38_address0 { O 6 vector } odd_q_38_ce0 { O 1 bit } odd_q_38_we0 { O 1 bit } odd_q_38_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_38'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2338 \
    name odd_q_37 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_37 \
    op interface \
    ports { odd_q_37_address0 { O 6 vector } odd_q_37_ce0 { O 1 bit } odd_q_37_we0 { O 1 bit } odd_q_37_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_37'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2339 \
    name odd_q_36 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_36 \
    op interface \
    ports { odd_q_36_address0 { O 6 vector } odd_q_36_ce0 { O 1 bit } odd_q_36_we0 { O 1 bit } odd_q_36_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_36'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2340 \
    name odd_q_35 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_35 \
    op interface \
    ports { odd_q_35_address0 { O 6 vector } odd_q_35_ce0 { O 1 bit } odd_q_35_we0 { O 1 bit } odd_q_35_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_35'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2341 \
    name odd_q_34 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_34 \
    op interface \
    ports { odd_q_34_address0 { O 6 vector } odd_q_34_ce0 { O 1 bit } odd_q_34_we0 { O 1 bit } odd_q_34_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_34'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2342 \
    name odd_q_33 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_33 \
    op interface \
    ports { odd_q_33_address0 { O 6 vector } odd_q_33_ce0 { O 1 bit } odd_q_33_we0 { O 1 bit } odd_q_33_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_33'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2343 \
    name odd_q_32 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_32 \
    op interface \
    ports { odd_q_32_address0 { O 6 vector } odd_q_32_ce0 { O 1 bit } odd_q_32_we0 { O 1 bit } odd_q_32_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_32'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2344 \
    name odd_q_31 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_31 \
    op interface \
    ports { odd_q_31_address0 { O 6 vector } odd_q_31_ce0 { O 1 bit } odd_q_31_we0 { O 1 bit } odd_q_31_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_31'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2345 \
    name odd_q_30 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_30 \
    op interface \
    ports { odd_q_30_address0 { O 6 vector } odd_q_30_ce0 { O 1 bit } odd_q_30_we0 { O 1 bit } odd_q_30_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_30'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2346 \
    name odd_q_29 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_29 \
    op interface \
    ports { odd_q_29_address0 { O 6 vector } odd_q_29_ce0 { O 1 bit } odd_q_29_we0 { O 1 bit } odd_q_29_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_29'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2347 \
    name odd_q_28 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_28 \
    op interface \
    ports { odd_q_28_address0 { O 6 vector } odd_q_28_ce0 { O 1 bit } odd_q_28_we0 { O 1 bit } odd_q_28_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_28'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2348 \
    name odd_q_27 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_27 \
    op interface \
    ports { odd_q_27_address0 { O 6 vector } odd_q_27_ce0 { O 1 bit } odd_q_27_we0 { O 1 bit } odd_q_27_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_27'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2349 \
    name odd_q_26 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_26 \
    op interface \
    ports { odd_q_26_address0 { O 6 vector } odd_q_26_ce0 { O 1 bit } odd_q_26_we0 { O 1 bit } odd_q_26_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_26'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2350 \
    name odd_q_25 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_25 \
    op interface \
    ports { odd_q_25_address0 { O 6 vector } odd_q_25_ce0 { O 1 bit } odd_q_25_we0 { O 1 bit } odd_q_25_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_25'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2351 \
    name odd_q_24 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_24 \
    op interface \
    ports { odd_q_24_address0 { O 6 vector } odd_q_24_ce0 { O 1 bit } odd_q_24_we0 { O 1 bit } odd_q_24_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_24'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2352 \
    name odd_q_23 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_23 \
    op interface \
    ports { odd_q_23_address0 { O 6 vector } odd_q_23_ce0 { O 1 bit } odd_q_23_we0 { O 1 bit } odd_q_23_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_23'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2353 \
    name odd_q_22 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_22 \
    op interface \
    ports { odd_q_22_address0 { O 6 vector } odd_q_22_ce0 { O 1 bit } odd_q_22_we0 { O 1 bit } odd_q_22_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_22'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2354 \
    name odd_q_21 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_21 \
    op interface \
    ports { odd_q_21_address0 { O 6 vector } odd_q_21_ce0 { O 1 bit } odd_q_21_we0 { O 1 bit } odd_q_21_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_21'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2355 \
    name odd_q_20 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_20 \
    op interface \
    ports { odd_q_20_address0 { O 6 vector } odd_q_20_ce0 { O 1 bit } odd_q_20_we0 { O 1 bit } odd_q_20_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_20'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2356 \
    name odd_q_19 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_19 \
    op interface \
    ports { odd_q_19_address0 { O 6 vector } odd_q_19_ce0 { O 1 bit } odd_q_19_we0 { O 1 bit } odd_q_19_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_19'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2357 \
    name odd_q_18 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_18 \
    op interface \
    ports { odd_q_18_address0 { O 6 vector } odd_q_18_ce0 { O 1 bit } odd_q_18_we0 { O 1 bit } odd_q_18_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_18'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2358 \
    name odd_q_17 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_17 \
    op interface \
    ports { odd_q_17_address0 { O 6 vector } odd_q_17_ce0 { O 1 bit } odd_q_17_we0 { O 1 bit } odd_q_17_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_17'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2359 \
    name odd_q_16 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_16 \
    op interface \
    ports { odd_q_16_address0 { O 6 vector } odd_q_16_ce0 { O 1 bit } odd_q_16_we0 { O 1 bit } odd_q_16_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_16'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2360 \
    name odd_q_15 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_15 \
    op interface \
    ports { odd_q_15_address0 { O 6 vector } odd_q_15_ce0 { O 1 bit } odd_q_15_we0 { O 1 bit } odd_q_15_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_15'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2361 \
    name odd_q_14 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_14 \
    op interface \
    ports { odd_q_14_address0 { O 6 vector } odd_q_14_ce0 { O 1 bit } odd_q_14_we0 { O 1 bit } odd_q_14_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_14'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2362 \
    name odd_q_13 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_13 \
    op interface \
    ports { odd_q_13_address0 { O 6 vector } odd_q_13_ce0 { O 1 bit } odd_q_13_we0 { O 1 bit } odd_q_13_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_13'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2363 \
    name odd_q_12 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_12 \
    op interface \
    ports { odd_q_12_address0 { O 6 vector } odd_q_12_ce0 { O 1 bit } odd_q_12_we0 { O 1 bit } odd_q_12_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_12'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2364 \
    name odd_q_11 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_11 \
    op interface \
    ports { odd_q_11_address0 { O 6 vector } odd_q_11_ce0 { O 1 bit } odd_q_11_we0 { O 1 bit } odd_q_11_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_11'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2365 \
    name odd_q_10 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_10 \
    op interface \
    ports { odd_q_10_address0 { O 6 vector } odd_q_10_ce0 { O 1 bit } odd_q_10_we0 { O 1 bit } odd_q_10_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_10'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2366 \
    name odd_q_9 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_9 \
    op interface \
    ports { odd_q_9_address0 { O 6 vector } odd_q_9_ce0 { O 1 bit } odd_q_9_we0 { O 1 bit } odd_q_9_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_9'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2367 \
    name odd_q_8 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_8 \
    op interface \
    ports { odd_q_8_address0 { O 6 vector } odd_q_8_ce0 { O 1 bit } odd_q_8_we0 { O 1 bit } odd_q_8_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_8'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2368 \
    name odd_q_7 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_7 \
    op interface \
    ports { odd_q_7_address0 { O 6 vector } odd_q_7_ce0 { O 1 bit } odd_q_7_we0 { O 1 bit } odd_q_7_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_7'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2369 \
    name odd_q_6 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_6 \
    op interface \
    ports { odd_q_6_address0 { O 6 vector } odd_q_6_ce0 { O 1 bit } odd_q_6_we0 { O 1 bit } odd_q_6_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_6'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2370 \
    name odd_q_5 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_5 \
    op interface \
    ports { odd_q_5_address0 { O 6 vector } odd_q_5_ce0 { O 1 bit } odd_q_5_we0 { O 1 bit } odd_q_5_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_5'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2371 \
    name odd_q_4 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_4 \
    op interface \
    ports { odd_q_4_address0 { O 6 vector } odd_q_4_ce0 { O 1 bit } odd_q_4_we0 { O 1 bit } odd_q_4_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_4'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2372 \
    name odd_q_3 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_3 \
    op interface \
    ports { odd_q_3_address0 { O 6 vector } odd_q_3_ce0 { O 1 bit } odd_q_3_we0 { O 1 bit } odd_q_3_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_3'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2373 \
    name odd_q_2 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_2 \
    op interface \
    ports { odd_q_2_address0 { O 6 vector } odd_q_2_ce0 { O 1 bit } odd_q_2_we0 { O 1 bit } odd_q_2_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_2'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2374 \
    name odd_q_1 \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q_1 \
    op interface \
    ports { odd_q_1_address0 { O 6 vector } odd_q_1_ce0 { O 1 bit } odd_q_1_we0 { O 1 bit } odd_q_1_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q_1'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2375 \
    name odd_q \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename odd_q \
    op interface \
    ports { odd_q_address0 { O 6 vector } odd_q_ce0 { O 1 bit } odd_q_we0 { O 1 bit } odd_q_d0 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'odd_q'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 2376 \
    name y2 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename y2 \
    op interface \
    ports { y2_address0 { O 13 vector } y2_ce0 { O 1 bit } y2_q0 { I 8 vector } y2_address1 { O 13 vector } y2_ce1 { O 1 bit } y2_q1 { I 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'y2'"
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


