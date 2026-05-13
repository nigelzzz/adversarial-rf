`timescale 1ns/10ps

module tb_pe_s8;

localparam CYCLE = 20;

reg              clk, rst_n, en, acc_clear;
reg  signed [7:0]  a_in, b_in;
wire signed [7:0]  a_out, b_out;
wire signed [31:0] acc;

pe_s8 DUT (
    .clk(clk), .rst_n(rst_n), .en(en), .acc_clear(acc_clear),
    .a_in(a_in), .b_in(b_in),
    .a_out(a_out), .b_out(b_out), .acc(acc)
);

initial clk = 0;
always #(CYCLE/2.0) clk = ~clk;

integer i;
reg signed [31:0] exp_acc;
reg signed [7:0]  exp_a, exp_b;

task check_fail;
    input [255:0] tag;
    input signed [31:0] got, exp;
    begin
        if (got !== exp) begin
            $display("FAIL %0s: got %0d, expected %0d", tag, got, exp);
            $finish;
        end
    end
endtask

initial begin
    rst_n = 1; en = 0; acc_clear = 0;
    a_in = 0; b_in = 0;

    // Reset: drive low for >= 2 cycles, then release
    #(CYCLE); rst_n = 0; #(2*CYCLE); rst_n = 1;
    @(negedge clk);

    // Test 1: Reset values
    check_fail("T1 acc",   acc,   32'sd0);
    check_fail("T1 a_out", {{24{a_out[7]}}, a_out}, 32'sd0);
    check_fail("T1 b_out", {{24{b_out[7]}}, b_out}, 32'sd0);
    $display("PASS test 1: reset");

    // Test 2: Single MAC (acc_clear=1 => acc = product = 3*7 = 21)
    en = 1; acc_clear = 1;
    a_in = 8'sd3; b_in = 8'sd7;
    @(negedge clk);
    check_fail("T2 acc", acc, 32'sd21);
    $display("PASS test 2: single MAC => 21");

    // Test 3: Accumulate (acc = 21 + 5*(-3) = 6)
    acc_clear = 0;
    a_in = 8'sd5; b_in = -8'sd3;
    @(negedge clk);
    check_fail("T3 acc", acc, 32'sd6);
    $display("PASS test 3: accumulate => 6");

    // Test 4: Continue (acc = 6 + (-10)*4 = -34)
    a_in = -8'sd10; b_in = 8'sd4;
    @(negedge clk);
    check_fail("T4 acc", acc, -32'sd34);
    $display("PASS test 4: continue => -34");

    // Test 5: Passthrough latency — a_out/b_out reflect inputs from PREVIOUS cycle
    // After test 4's posedge, a_out should be -10 and b_out should be 4
    // (those were the inputs that just got latched)
    check_fail("T5a a_out", {{24{a_out[7]}}, a_out}, -32'sd10);
    check_fail("T5a b_out", {{24{b_out[7]}}, b_out}, 32'sd4);
    // Now feed new values and verify they appear after one clock
    acc_clear = 1;
    a_in = 8'sd42; b_in = -8'sd100;
    @(negedge clk);
    check_fail("T5 a_out", {{24{a_out[7]}}, a_out}, 32'sd42);
    check_fail("T5 b_out", {{24{b_out[7]}}, b_out}, -32'sd100);
    $display("PASS test 5: passthrough a=42, b=-100");

    // Test 6: Enable gate (en=0 freezes all state)
    exp_acc = acc;
    exp_a = a_out;
    exp_b = b_out;
    en = 0;
    a_in = 8'sd99; b_in = 8'sd99;
    @(negedge clk);
    check_fail("T6 acc",   acc, exp_acc);
    check_fail("T6 a_out", {{24{a_out[7]}}, a_out}, {{24{exp_a[7]}}, exp_a});
    check_fail("T6 b_out", {{24{b_out[7]}}, b_out}, {{24{exp_b[7]}}, exp_b});
    // Wait another cycle to be sure
    @(negedge clk);
    check_fail("T6b acc",  acc, exp_acc);
    $display("PASS test 6: en=0 freezes state");

    // Test 7: Signed min x min (-128 * -128 = 16384)
    en = 1; acc_clear = 1;
    a_in = -8'sd128; b_in = -8'sd128;
    @(negedge clk);
    check_fail("T7 acc", acc, 32'sd16384);
    $display("PASS test 7: -128 * -128 => 16384");

    // Test 8: Signed cross (acc = 16384 + (-128)*127 = 16384 - 16256 = 128)
    acc_clear = 0;
    a_in = -8'sd128; b_in = 8'sd127;
    @(negedge clk);
    check_fail("T8 acc", acc, 32'sd128);
    $display("PASS test 8: acc => 128");

    // Test 9: Stress x256 (1x acc_clear=1 + 255x acc_clear=0, all -128*-128)
    acc_clear = 1;
    a_in = -8'sd128; b_in = -8'sd128;
    @(negedge clk); // first MAC: acc = 16384
    acc_clear = 0;
    for (i = 0; i < 255; i = i + 1) begin
        @(negedge clk);
    end
    check_fail("T9 acc", acc, 32'sd4194304); // 256 * 16384
    $display("PASS test 9: 256x stress => 4194304");

    $display("ALL PE TESTS PASSED");
    $finish;
end

endmodule
