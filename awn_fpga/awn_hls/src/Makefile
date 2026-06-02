# Wrapper around vitis_hls for the AWN int8 forward pass.
# Requires `source ~/vitas/Vitis/2023.2/settings64.sh` first.

.PHONY: csim csynth all report clean

csim:
	HLS_STAGE=csim vitis_hls -f run_hls.tcl

csynth:
	HLS_STAGE=csynth vitis_hls -f run_hls.tcl

all:
	HLS_STAGE=all vitis_hls -f run_hls.tcl

report:
	@find proj_awn_int8 -name "*_csynth.rpt" -exec cat {} \;

clean:
	rm -rf proj_awn_int8 vitis_hls.log *.log
