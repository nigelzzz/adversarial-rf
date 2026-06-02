# Vitis HLS 2023.2 hello-world flow.
#
# Stage selection via env var HLS_STAGE (csim | csynth | all).
# Default: csim. ($argv handling in vitis_hls is unreliable.)
#
# Usage (from this directory):
#   HLS_STAGE=csim   vitis_hls -f run_hls.tcl
#   HLS_STAGE=csynth vitis_hls -f run_hls.tcl
#   HLS_STAGE=all    vitis_hls -f run_hls.tcl

set stage "csim"
if {[info exists ::env(HLS_STAGE)]} {
    set stage $::env(HLS_STAGE)
}
puts "==== HLS_STAGE = $stage ===="

open_project -reset proj_simple_add
set_top simple_add
add_files simple_add.cpp -cflags "-std=c++14"
add_files -tb tb_simple_add.cpp -cflags "-std=c++14"

open_solution -reset -flow_target vivado "solution1"
# Generic Zynq UltraScale+ part; change if you have a specific board.
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 5 -name default          ;# 200 MHz target

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
