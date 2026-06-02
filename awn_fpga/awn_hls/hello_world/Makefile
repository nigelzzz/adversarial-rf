# Wrapper around vitis_hls. Requires `source <Vitis>/settings64.sh` first.

.PHONY: csim csynth all clean report

csim:
	HLS_STAGE=csim vitis_hls -f run_hls.tcl

csynth:
	HLS_STAGE=csynth vitis_hls -f run_hls.tcl

all:
	HLS_STAGE=all vitis_hls -f run_hls.tcl

report:
	@find proj_simple_add -name "*_csynth.rpt" -exec cat {} \;

clean:
	rm -rf proj_simple_add vitis_hls.log *.log
