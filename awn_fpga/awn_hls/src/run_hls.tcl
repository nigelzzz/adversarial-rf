# Phase 4 — Vitis HLS 2023.2 flow for AWN int8 forward pass.
#
# Stage selection via env var HLS_STAGE (csim | csynth | all). Default csim.

set stage "csim"
if {[info exists ::env(HLS_STAGE)]} {
    set stage $::env(HLS_STAGE)
}
puts "==== HLS_STAGE = $stage ===="

open_project -reset proj_awn_int8
set_top awn_forward

# Include path picks up dims/weights/biases/qparams/golden_io from ../golden.
set incdirs "-I../golden -std=c++14"

add_files awn_int8.cpp -cflags $incdirs
add_files -tb tb_awn_int8.cpp -cflags $incdirs

open_solution -reset -flow_target vivado "solution1"
# Same placeholder ZU7EV part as the hello-world.  Swap to your real board
# (e.g. xczu9eg-ffvb1156-2-e for ZCU102) before final csynth.
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 5 -name default          ;# 200 MHz target

# Phase 4: functional only, no pragmas yet — Phase 5 adds pipelining etc.

if {$stage eq "csim" || $stage eq "all"} {
    puts "==== C SIMULATION ===="
    csim_design
}

if {$stage eq "csynth" || $stage eq "all"} {
    puts "==== C SYNTHESIS ===="
    csynth_design
}

if {$stage eq "all"} {
    puts "==== C/RTL CO-SIMULATION ===="
    cosim_design
}

exit
