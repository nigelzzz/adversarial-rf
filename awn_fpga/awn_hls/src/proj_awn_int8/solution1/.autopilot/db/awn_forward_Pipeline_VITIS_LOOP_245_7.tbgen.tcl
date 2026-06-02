set moduleName awn_forward_Pipeline_VITIS_LOOP_245_7
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
set C_modelName {awn_forward_Pipeline_VITIS_LOOP_245_7}
set C_modelType { void 0 }
set C_modelArgList {
	{ avg_d_q int 8 regular {array 64 { 1 3 } 1 1 }  }
	{ cat_q int 8 regular {array 128 { 0 0 } 0 1 }  }
	{ avg_c_q int 8 regular {array 64 { 1 3 } 1 1 }  }
}
set hasAXIMCache 0
set AXIMCacheInstList { }
set C_modelArgMapList {[ 
	{ "Name" : "avg_d_q", "interface" : "memory", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "cat_q", "interface" : "memory", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "avg_c_q", "interface" : "memory", "bitwidth" : 8, "direction" : "READONLY"} ]}
# RTL Port declarations: 
set portNum 20
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ avg_d_q_address0 sc_out sc_lv 6 signal 0 } 
	{ avg_d_q_ce0 sc_out sc_logic 1 signal 0 } 
	{ avg_d_q_q0 sc_in sc_lv 8 signal 0 } 
	{ cat_q_address0 sc_out sc_lv 7 signal 1 } 
	{ cat_q_ce0 sc_out sc_logic 1 signal 1 } 
	{ cat_q_we0 sc_out sc_logic 1 signal 1 } 
	{ cat_q_d0 sc_out sc_lv 8 signal 1 } 
	{ cat_q_address1 sc_out sc_lv 7 signal 1 } 
	{ cat_q_ce1 sc_out sc_logic 1 signal 1 } 
	{ cat_q_we1 sc_out sc_logic 1 signal 1 } 
	{ cat_q_d1 sc_out sc_lv 8 signal 1 } 
	{ avg_c_q_address0 sc_out sc_lv 6 signal 2 } 
	{ avg_c_q_ce0 sc_out sc_logic 1 signal 2 } 
	{ avg_c_q_q0 sc_in sc_lv 8 signal 2 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "avg_d_q_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "avg_d_q", "role": "address0" }} , 
 	{ "name": "avg_d_q_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "avg_d_q", "role": "ce0" }} , 
 	{ "name": "avg_d_q_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "avg_d_q", "role": "q0" }} , 
 	{ "name": "cat_q_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "cat_q", "role": "address0" }} , 
 	{ "name": "cat_q_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "cat_q", "role": "ce0" }} , 
 	{ "name": "cat_q_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "cat_q", "role": "we0" }} , 
 	{ "name": "cat_q_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "cat_q", "role": "d0" }} , 
 	{ "name": "cat_q_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "cat_q", "role": "address1" }} , 
 	{ "name": "cat_q_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "cat_q", "role": "ce1" }} , 
 	{ "name": "cat_q_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "cat_q", "role": "we1" }} , 
 	{ "name": "cat_q_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "cat_q", "role": "d1" }} , 
 	{ "name": "avg_c_q_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "avg_c_q", "role": "address0" }} , 
 	{ "name": "avg_c_q_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "avg_c_q", "role": "ce0" }} , 
 	{ "name": "avg_c_q_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "avg_c_q", "role": "q0" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2"],
		"CDFG" : "awn_forward_Pipeline_VITIS_LOOP_245_7",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "68", "EstimateLatencyMax" : "68",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "avg_d_q", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "cat_q", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "avg_c_q", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_245_7", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter3", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter3", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_8s_32ns_39_1_1_U10845", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.flow_control_loop_pipe_sequential_init_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	awn_forward_Pipeline_VITIS_LOOP_245_7 {
		avg_d_q {Type I LastRead 0 FirstWrite -1}
		cat_q {Type O LastRead -1 FirstWrite 1}
		avg_c_q {Type I LastRead 0 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "68", "Max" : "68"}
	, {"Name" : "Interval", "Min" : "68", "Max" : "68"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	avg_d_q { ap_memory {  { avg_d_q_address0 mem_address 1 6 }  { avg_d_q_ce0 mem_ce 1 1 }  { avg_d_q_q0 in_data 0 8 } } }
	cat_q { ap_memory {  { cat_q_address0 mem_address 1 7 }  { cat_q_ce0 mem_ce 1 1 }  { cat_q_we0 mem_we 1 1 }  { cat_q_d0 mem_din 1 8 }  { cat_q_address1 MemPortADDR2 1 7 }  { cat_q_ce1 MemPortCE2 1 1 }  { cat_q_we1 MemPortWE2 1 1 }  { cat_q_d1 MemPortDIN2 1 8 } } }
	avg_c_q { ap_memory {  { avg_c_q_address0 mem_address 1 6 }  { avg_c_q_ce0 mem_ce 1 1 }  { avg_c_q_q0 mem_dout 0 8 } } }
}
