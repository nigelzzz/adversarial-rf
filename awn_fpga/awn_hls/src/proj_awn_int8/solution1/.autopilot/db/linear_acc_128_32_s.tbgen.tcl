set moduleName linear_acc_128_32_s
set isTopModule 0
set isCombinational 0
set isDatapathOnly 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set DLRegFirstOffset 0
set DLRegItemOffset 0
set C_modelName {linear_acc<128, 32>}
set C_modelType { void 0 }
set C_modelArgList {
	{ x int 7 regular {array 32 { 1 1 } 1 1 }  }
	{ out_r int 20 regular {array 128 { 0 3 } 0 1 }  }
}
set hasAXIMCache 0
set AXIMCacheInstList { }
set C_modelArgMapList {[ 
	{ "Name" : "x", "interface" : "memory", "bitwidth" : 7, "direction" : "READONLY"} , 
 	{ "Name" : "out_r", "interface" : "memory", "bitwidth" : 20, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 16
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ x_address0 sc_out sc_lv 5 signal 0 } 
	{ x_ce0 sc_out sc_logic 1 signal 0 } 
	{ x_q0 sc_in sc_lv 7 signal 0 } 
	{ x_address1 sc_out sc_lv 5 signal 0 } 
	{ x_ce1 sc_out sc_logic 1 signal 0 } 
	{ x_q1 sc_in sc_lv 7 signal 0 } 
	{ out_r_address0 sc_out sc_lv 7 signal 1 } 
	{ out_r_ce0 sc_out sc_logic 1 signal 1 } 
	{ out_r_we0 sc_out sc_logic 1 signal 1 } 
	{ out_r_d0 sc_out sc_lv 20 signal 1 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "x_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "x", "role": "address0" }} , 
 	{ "name": "x_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "x", "role": "ce0" }} , 
 	{ "name": "x_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "x", "role": "q0" }} , 
 	{ "name": "x_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "x", "role": "address1" }} , 
 	{ "name": "x_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "x", "role": "ce1" }} , 
 	{ "name": "x_q1", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "x", "role": "q1" }} , 
 	{ "name": "out_r_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "out_r", "role": "address0" }} , 
 	{ "name": "out_r_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "out_r", "role": "ce0" }} , 
 	{ "name": "out_r_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "out_r", "role": "we0" }} , 
 	{ "name": "out_r_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":20, "type": "signal", "bundle":{"name": "out_r", "role": "d0" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1"],
		"CDFG" : "linear_acc_128_32_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "151", "EstimateLatencyMax" : "151",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "x", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "out_r", "Type" : "Memory", "Direction" : "O",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "out_r", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_0", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_1", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_2", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_3", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_4", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_5", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_5", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_6", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_6", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_7", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_7", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_8", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_8", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_9", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_9", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_10", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_10", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_11", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_11", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_12", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_12", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_13", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_13", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_14", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_14", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_15", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_15", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_16", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_16", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_17", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_17", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_18", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_18", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_19", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_19", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_20", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_20", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_21", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_21", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_22", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_22", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_23", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_23", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_24", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_24", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_25", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_25", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_26", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_26", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_27", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_27", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_28", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_28", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_29", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_29", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_30", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_30", "Inst_start_state" : "17", "Inst_end_state" : "18"}]},
			{"Name" : "p_ZL4Wse3_31", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Port" : "p_ZL4Wse3_31", "Inst_start_state" : "17", "Inst_end_state" : "18"}]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431", "Parent" : "0", "Child" : ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66"],
		"CDFG" : "linear_acc_128_32_Pipeline_VITIS_LOOP_186_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "134", "EstimateLatencyMax" : "134",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "zext_ln186", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_15_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_30_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_17_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_9_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_29_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_24_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_14_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_28_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_18_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_6_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_13_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_21_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_27_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_4_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_10_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_23_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_3_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_19_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_26_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_12_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_5_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_1_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_2_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_8_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_11_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_25_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_7_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_22_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_20_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv12_16_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "out_r", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "p_ZL4Wse3_0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_5", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_6", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_7", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_8", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_9", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_10", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_11", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_12", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_13", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_14", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_15", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_16", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_17", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_18", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_19", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_20", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_21", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_22", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_23", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_24", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_25", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_26", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_27", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_28", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_29", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_30", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_ZL4Wse3_31", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_186_1", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter5", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter5", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "2", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_0_U", "Parent" : "1"},
	{"ID" : "3", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_1_U", "Parent" : "1"},
	{"ID" : "4", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_2_U", "Parent" : "1"},
	{"ID" : "5", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_3_U", "Parent" : "1"},
	{"ID" : "6", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_4_U", "Parent" : "1"},
	{"ID" : "7", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_5_U", "Parent" : "1"},
	{"ID" : "8", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_6_U", "Parent" : "1"},
	{"ID" : "9", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_7_U", "Parent" : "1"},
	{"ID" : "10", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_8_U", "Parent" : "1"},
	{"ID" : "11", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_9_U", "Parent" : "1"},
	{"ID" : "12", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_10_U", "Parent" : "1"},
	{"ID" : "13", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_11_U", "Parent" : "1"},
	{"ID" : "14", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_12_U", "Parent" : "1"},
	{"ID" : "15", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_13_U", "Parent" : "1"},
	{"ID" : "16", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_14_U", "Parent" : "1"},
	{"ID" : "17", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_15_U", "Parent" : "1"},
	{"ID" : "18", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_16_U", "Parent" : "1"},
	{"ID" : "19", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_17_U", "Parent" : "1"},
	{"ID" : "20", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_18_U", "Parent" : "1"},
	{"ID" : "21", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_19_U", "Parent" : "1"},
	{"ID" : "22", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_20_U", "Parent" : "1"},
	{"ID" : "23", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_21_U", "Parent" : "1"},
	{"ID" : "24", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_22_U", "Parent" : "1"},
	{"ID" : "25", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_23_U", "Parent" : "1"},
	{"ID" : "26", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_24_U", "Parent" : "1"},
	{"ID" : "27", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_25_U", "Parent" : "1"},
	{"ID" : "28", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_26_U", "Parent" : "1"},
	{"ID" : "29", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_27_U", "Parent" : "1"},
	{"ID" : "30", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_28_U", "Parent" : "1"},
	{"ID" : "31", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_29_U", "Parent" : "1"},
	{"ID" : "32", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_30_U", "Parent" : "1"},
	{"ID" : "33", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.p_ZL4Wse3_31_U", "Parent" : "1"},
	{"ID" : "34", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mul_8s_7ns_15_1_1_U11244", "Parent" : "1"},
	{"ID" : "35", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mul_8s_7ns_15_1_1_U11245", "Parent" : "1"},
	{"ID" : "36", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mul_8s_7ns_15_1_1_U11246", "Parent" : "1"},
	{"ID" : "37", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mul_8s_7ns_15_1_1_U11247", "Parent" : "1"},
	{"ID" : "38", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mul_8s_7ns_15_1_1_U11248", "Parent" : "1"},
	{"ID" : "39", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mul_8s_7ns_15_1_1_U11249", "Parent" : "1"},
	{"ID" : "40", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mul_8s_7ns_15_1_1_U11250", "Parent" : "1"},
	{"ID" : "41", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mul_8s_7ns_15_1_1_U11251", "Parent" : "1"},
	{"ID" : "42", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mul_8s_7ns_15_1_1_U11252", "Parent" : "1"},
	{"ID" : "43", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mul_8s_7ns_15_1_1_U11253", "Parent" : "1"},
	{"ID" : "44", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mul_8s_7ns_15_1_1_U11254", "Parent" : "1"},
	{"ID" : "45", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mul_8s_7ns_15_1_1_U11255", "Parent" : "1"},
	{"ID" : "46", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mul_8s_7ns_15_1_1_U11256", "Parent" : "1"},
	{"ID" : "47", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mul_8s_7ns_15_1_1_U11257", "Parent" : "1"},
	{"ID" : "48", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mul_8s_7ns_15_1_1_U11258", "Parent" : "1"},
	{"ID" : "49", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mul_8s_7ns_15_1_1_U11259", "Parent" : "1"},
	{"ID" : "50", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mac_muladd_8s_7ns_15s_16_4_1_U11260", "Parent" : "1"},
	{"ID" : "51", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mac_muladd_8s_7ns_15s_16_4_1_U11261", "Parent" : "1"},
	{"ID" : "52", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mac_muladd_8s_7ns_15s_16_4_1_U11262", "Parent" : "1"},
	{"ID" : "53", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mac_muladd_8s_7ns_15s_16_4_1_U11263", "Parent" : "1"},
	{"ID" : "54", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mac_muladd_8s_7ns_15s_16_4_1_U11264", "Parent" : "1"},
	{"ID" : "55", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mac_muladd_8s_7ns_15s_16_4_1_U11265", "Parent" : "1"},
	{"ID" : "56", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mac_muladd_8s_7ns_15s_16_4_1_U11266", "Parent" : "1"},
	{"ID" : "57", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mac_muladd_8s_7ns_15s_16_4_1_U11267", "Parent" : "1"},
	{"ID" : "58", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mac_muladd_7s_7ns_15s_15_4_1_U11268", "Parent" : "1"},
	{"ID" : "59", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mac_muladd_8s_7ns_15s_16_4_1_U11269", "Parent" : "1"},
	{"ID" : "60", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mac_muladd_8s_7ns_15s_16_4_1_U11270", "Parent" : "1"},
	{"ID" : "61", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mac_muladd_8s_7ns_15s_16_4_1_U11271", "Parent" : "1"},
	{"ID" : "62", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mac_muladd_8s_7ns_15s_16_4_1_U11272", "Parent" : "1"},
	{"ID" : "63", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mac_muladd_8s_7ns_15s_16_4_1_U11273", "Parent" : "1"},
	{"ID" : "64", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mac_muladd_8s_7ns_15s_16_4_1_U11274", "Parent" : "1"},
	{"ID" : "65", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.mac_muladd_7s_7ns_15s_15_4_1_U11275", "Parent" : "1"},
	{"ID" : "66", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431.flow_control_loop_pipe_sequential_init_U", "Parent" : "1"}]}


set ArgLastReadFirstWriteLatency {
	linear_acc_128_32_s {
		x {Type I LastRead 16 FirstWrite -1}
		out_r {Type O LastRead -1 FirstWrite 5}
		p_ZL4Wse3_0 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_1 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_2 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_3 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_4 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_5 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_6 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_7 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_8 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_9 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_10 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_11 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_12 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_13 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_14 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_15 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_16 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_17 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_18 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_19 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_20 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_21 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_22 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_23 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_24 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_25 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_26 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_27 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_28 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_29 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_30 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_31 {Type I LastRead -1 FirstWrite -1}}
	linear_acc_128_32_Pipeline_VITIS_LOOP_186_1 {
		zext_ln186 {Type I LastRead 0 FirstWrite -1}
		conv12_15_cast {Type I LastRead 0 FirstWrite -1}
		conv12_30_cast {Type I LastRead 0 FirstWrite -1}
		conv12_17_cast {Type I LastRead 0 FirstWrite -1}
		conv12_9_cast {Type I LastRead 0 FirstWrite -1}
		conv12_29_cast {Type I LastRead 0 FirstWrite -1}
		conv12_24_cast {Type I LastRead 0 FirstWrite -1}
		conv12_14_cast {Type I LastRead 0 FirstWrite -1}
		conv12_cast {Type I LastRead 0 FirstWrite -1}
		conv12_28_cast {Type I LastRead 0 FirstWrite -1}
		conv12_18_cast {Type I LastRead 0 FirstWrite -1}
		conv12_6_cast {Type I LastRead 0 FirstWrite -1}
		conv12_13_cast {Type I LastRead 0 FirstWrite -1}
		conv12_21_cast {Type I LastRead 0 FirstWrite -1}
		conv12_27_cast {Type I LastRead 0 FirstWrite -1}
		conv12_4_cast {Type I LastRead 0 FirstWrite -1}
		conv12_10_cast {Type I LastRead 0 FirstWrite -1}
		conv12_23_cast {Type I LastRead 0 FirstWrite -1}
		conv12_3_cast {Type I LastRead 0 FirstWrite -1}
		conv12_19_cast {Type I LastRead 0 FirstWrite -1}
		conv12_26_cast {Type I LastRead 0 FirstWrite -1}
		conv12_12_cast {Type I LastRead 0 FirstWrite -1}
		conv12_5_cast {Type I LastRead 0 FirstWrite -1}
		conv12_1_cast {Type I LastRead 0 FirstWrite -1}
		conv12_2_cast {Type I LastRead 0 FirstWrite -1}
		conv12_8_cast {Type I LastRead 0 FirstWrite -1}
		conv12_11_cast {Type I LastRead 0 FirstWrite -1}
		conv12_25_cast {Type I LastRead 0 FirstWrite -1}
		conv12_7_cast {Type I LastRead 0 FirstWrite -1}
		conv12_22_cast {Type I LastRead 0 FirstWrite -1}
		conv12_20_cast {Type I LastRead 0 FirstWrite -1}
		conv12_16_cast {Type I LastRead 0 FirstWrite -1}
		out_r {Type O LastRead -1 FirstWrite 5}
		p_ZL4Wse3_0 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_1 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_2 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_3 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_4 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_5 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_6 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_7 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_8 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_9 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_10 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_11 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_12 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_13 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_14 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_15 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_16 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_17 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_18 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_19 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_20 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_21 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_22 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_23 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_24 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_25 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_26 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_27 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_28 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_29 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_30 {Type I LastRead -1 FirstWrite -1}
		p_ZL4Wse3_31 {Type I LastRead -1 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "151", "Max" : "151"}
	, {"Name" : "Interval", "Min" : "151", "Max" : "151"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	x { ap_memory {  { x_address0 mem_address 1 5 }  { x_ce0 mem_ce 1 1 }  { x_q0 mem_dout 0 7 }  { x_address1 MemPortADDR2 1 5 }  { x_ce1 MemPortCE2 1 1 }  { x_q1 MemPortDOUT2 0 7 } } }
	out_r { ap_memory {  { out_r_address0 mem_address 1 7 }  { out_r_ce0 mem_ce 1 1 }  { out_r_we0 mem_we 1 1 }  { out_r_d0 mem_din 1 20 } } }
}
