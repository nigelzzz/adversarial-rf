set ModuleHierarchy {[{
"Name" : "awn_forward","ID" : "0","Type" : "sequential",
"SubInsts" : [
	{"Name" : "grp_conv1_block_fu_20192","ID" : "1","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_53_1_VITIS_LOOP_54_2","ID" : "2","Type" : "pipeline"},]},
	{"Name" : "grp_u_branch_fu_20870","ID" : "3","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_100_1","ID" : "4","Type" : "no"},
		{"Name" : "VITIS_LOOP_116_5","ID" : "5","Type" : "no",
		"SubInsts" : [
		{"Name" : "grp_u_branch_Pipeline_VITIS_LOOP_117_6_fu_34","ID" : "6","Type" : "sequential",
				"SubLoops" : [
				{"Name" : "VITIS_LOOP_117_6","ID" : "7","Type" : "pipeline"},]},]},]},
	{"Name" : "grp_p_branch_fu_20876","ID" : "8","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_137_1","ID" : "9","Type" : "no"},
		{"Name" : "VITIS_LOOP_153_5","ID" : "10","Type" : "no",
		"SubInsts" : [
		{"Name" : "grp_p_branch_Pipeline_VITIS_LOOP_154_6_fu_34","ID" : "11","Type" : "sequential",
				"SubLoops" : [
				{"Name" : "VITIS_LOOP_154_6","ID" : "12","Type" : "pipeline"},]},]},]},
	{"Name" : "grp_conv2_block_fu_20882","ID" : "13","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_76_1","ID" : "14","Type" : "no",
		"SubInsts" : [
		{"Name" : "grp_conv2_block_Pipeline_VITIS_LOOP_77_2_fu_5483","ID" : "15","Type" : "sequential",
				"SubLoops" : [
				{"Name" : "VITIS_LOOP_77_2","ID" : "16","Type" : "pipeline"},]},]},]},
	{"Name" : "grp_awn_forward_Pipeline_VITIS_LOOP_210_1_VITIS_LOOP_211_2_fu_21849","ID" : "17","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_210_1_VITIS_LOOP_211_2","ID" : "18","Type" : "pipeline"},]},
	{"Name" : "grp_awn_forward_Pipeline_VITIS_LOOP_221_3_VITIS_LOOP_222_4_fu_21919","ID" : "19","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_221_3_VITIS_LOOP_222_4","ID" : "20","Type" : "pipeline"},]},
	{"Name" : "grp_awn_forward_Pipeline_VITIS_LOOP_232_5_VITIS_LOOP_233_6_fu_26021","ID" : "21","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_232_5_VITIS_LOOP_233_6","ID" : "22","Type" : "pipeline"},]},
	{"Name" : "grp_avgpool_64_fu_26154","ID" : "23","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_172_1","ID" : "24","Type" : "pipeline"},]},
	{"Name" : "grp_avgpool_64_1_fu_26223","ID" : "25","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_172_1","ID" : "26","Type" : "pipeline"},]},
	{"Name" : "grp_awn_forward_Pipeline_VITIS_LOOP_245_7_fu_30324","ID" : "27","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_245_7","ID" : "28","Type" : "pipeline"},]},
	{"Name" : "grp_linear_acc_32_128_s_fu_30331","ID" : "29","Type" : "sequential",
		"SubInsts" : [
		{"Name" : "grp_linear_acc_32_128_Pipeline_VITIS_LOOP_186_1_fu_1679","ID" : "30","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "VITIS_LOOP_186_1","ID" : "31","Type" : "pipeline"},]},]},
	{"Name" : "grp_awn_forward_Pipeline_VITIS_LOOP_255_8_fu_30593","ID" : "32","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_255_8","ID" : "33","Type" : "pipeline"},]},
	{"Name" : "grp_linear_acc_128_32_s_fu_30599","ID" : "34","Type" : "sequential",
		"SubInsts" : [
		{"Name" : "grp_linear_acc_128_32_Pipeline_VITIS_LOOP_186_1_fu_431","ID" : "35","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "VITIS_LOOP_186_1","ID" : "36","Type" : "pipeline"},]},]},
	{"Name" : "grp_awn_forward_Pipeline_VITIS_LOOP_263_9_fu_30669","ID" : "37","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_263_9","ID" : "38","Type" : "pipeline"},]},
	{"Name" : "grp_awn_forward_Pipeline_VITIS_LOOP_269_10_fu_30677","ID" : "39","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_269_10","ID" : "40","Type" : "pipeline"},]},
	{"Name" : "grp_linear_acc_320_128_s_fu_30684","ID" : "41","Type" : "sequential",
		"SubInsts" : [
		{"Name" : "grp_linear_acc_320_128_Pipeline_VITIS_LOOP_186_1_fu_1681","ID" : "42","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "VITIS_LOOP_186_1","ID" : "43","Type" : "pipeline"},]},]},
	{"Name" : "grp_awn_forward_Pipeline_VITIS_LOOP_277_11_fu_30948","ID" : "44","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_277_11","ID" : "45","Type" : "pipeline"},]},
	{"Name" : "grp_linear_acc_11_320_s_fu_30954","ID" : "46","Type" : "sequential",
		"SubInsts" : [
		{"Name" : "grp_linear_acc_11_320_Pipeline_VITIS_LOOP_186_1_fu_4177","ID" : "47","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "VITIS_LOOP_186_1","ID" : "48","Type" : "pipeline"},]},]},
	{"Name" : "grp_awn_forward_Pipeline_VITIS_LOOP_284_12_fu_31602","ID" : "49","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_284_12","ID" : "50","Type" : "pipeline"},]},]
}]}