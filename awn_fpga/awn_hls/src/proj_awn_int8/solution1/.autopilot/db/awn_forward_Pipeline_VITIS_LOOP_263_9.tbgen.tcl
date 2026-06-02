set moduleName awn_forward_Pipeline_VITIS_LOOP_263_9
set isTopModule 0
set isCombinational 0
set isDatapathOnly 0
set isPipelined 1
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set DLRegFirstOffset 0
set DLRegItemOffset 0
set C_modelName {awn_forward_Pipeline_VITIS_LOOP_263_9}
set C_modelType { void 0 }
set C_modelArgList {
	{ se3_acc int 20 regular {array 128 { 1 3 } 1 1 }  }
	{ sig_q int 7 regular {array 128 { 0 3 } 0 1 }  }
}
set hasAXIMCache 0
set AXIMCacheInstList { }
set C_modelArgMapList {[ 
	{ "Name" : "se3_acc", "interface" : "memory", "bitwidth" : 20, "direction" : "READONLY"} , 
 	{ "Name" : "sig_q", "interface" : "memory", "bitwidth" : 7, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 13
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ se3_acc_address0 sc_out sc_lv 7 signal 0 } 
	{ se3_acc_ce0 sc_out sc_logic 1 signal 0 } 
	{ se3_acc_q0 sc_in sc_lv 20 signal 0 } 
	{ sig_q_address0 sc_out sc_lv 7 signal 1 } 
	{ sig_q_ce0 sc_out sc_logic 1 signal 1 } 
	{ sig_q_we0 sc_out sc_logic 1 signal 1 } 
	{ sig_q_d0 sc_out sc_lv 7 signal 1 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "se3_acc_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "se3_acc", "role": "address0" }} , 
 	{ "name": "se3_acc_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "se3_acc", "role": "ce0" }} , 
 	{ "name": "se3_acc_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":20, "type": "signal", "bundle":{"name": "se3_acc", "role": "q0" }} , 
 	{ "name": "sig_q_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "sig_q", "role": "address0" }} , 
 	{ "name": "sig_q_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "sig_q", "role": "ce0" }} , 
 	{ "name": "sig_q_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "sig_q", "role": "we0" }} , 
 	{ "name": "sig_q_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "sig_q", "role": "d0" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3"],
		"CDFG" : "awn_forward_Pipeline_VITIS_LOOP_263_9",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "133", "EstimateLatencyMax" : "133",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "se3_acc", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "sig_q", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "SIGMOID_LUT", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_263_9", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter4", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter4", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.SIGMOID_LUT_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_20s_32ns_51_1_1_U11346", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.flow_control_loop_pipe_sequential_init_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	awn_forward_Pipeline_VITIS_LOOP_263_9 {
		se3_acc {Type I LastRead 0 FirstWrite -1}
		sig_q {Type O LastRead -1 FirstWrite 4}
		SIGMOID_LUT {Type I LastRead -1 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "133", "Max" : "133"}
	, {"Name" : "Interval", "Min" : "133", "Max" : "133"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	se3_acc { ap_memory {  { se3_acc_address0 mem_address 1 7 }  { se3_acc_ce0 mem_ce 1 1 }  { se3_acc_q0 mem_dout 0 20 } } }
	sig_q { ap_memory {  { sig_q_address0 mem_address 1 7 }  { sig_q_ce0 mem_ce 1 1 }  { sig_q_we0 mem_we 1 1 }  { sig_q_d0 mem_din 1 7 } } }
}
