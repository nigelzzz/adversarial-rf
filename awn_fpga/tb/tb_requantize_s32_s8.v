`timescale 1ns/10ps
`include "requantize_s32_s8.v"

module tb_requantize_s32_s8;

localparam CYCLE = 20;
localparam LEN = 16384, ADDR_W = 16;

reg                clk, rst_n, start;
reg  [ADDR_W-1:0]  length;
reg  signed [31:0] mul;
reg         [5:0]  shift;
reg  signed [7:0]  out_zp;
reg         [1:0]  act_mode;
wire               done;

requantize_s32_s8 #(.LEN(LEN), .ADDR_W(ADDR_W)) DUT (
    .clk(clk), .rst_n(rst_n), .start(start), .length(length),
    .mul(mul), .shift(shift), .out_zp(out_zp), .act_mode(act_mode),
    .done(done));

initial clk = 0; always #(CYCLE/2.0) clk = ~clk;

integer arglen, fout, k, mul_arg, shift_arg, zp_arg, act_arg;
reg [255:0] in_path, out_path;
reg [31:0]  cycles;

initial begin
    if (!$value$plusargs("len=%d", arglen))     arglen = 64;
    if (!$value$plusargs("mul=%d", mul_arg))    mul_arg = 1073741824; // ~0.5 in Q31
    if (!$value$plusargs("shift=%d", shift_arg)) shift_arg = 31;
    if (!$value$plusargs("zp=%d", zp_arg))      zp_arg = 0;
    if (!$value$plusargs("act=%d", act_arg))    act_arg = 0;
    if (!$value$plusargs("in=%s",  in_path))    in_path  = "vectors/rq_in.hex";
    if (!$value$plusargs("out=%s", out_path))   out_path = "vectors/rq_out.hex";

    length = arglen[ADDR_W-1:0];
    mul = mul_arg;
    shift = shift_arg[5:0];
    out_zp = zp_arg[7:0];
    act_mode = act_arg[1:0];

    rst_n = 1; start = 0;
    $readmemh(in_path, DUT.in_buf);
    #(2*CYCLE); rst_n = 0; #(2*CYCLE); rst_n = 1;
    @(negedge clk); start = 1; @(negedge clk); start = 0;

    cycles = 0;
    while (!done) begin
        cycles = cycles + 1;
        if (cycles > 5*LEN + 1000) begin $display("TIMEOUT"); $finish; end
        @(negedge clk);
    end

    fout = $fopen(out_path, "w");
    for (k = 0; k < arglen; k = k + 1)
        $fwrite(fout, "%02x\n", DUT.out_buf[k] & 8'hff);
    $fclose(fout);
    $display("requantize_s32_s8 OK len=%0d cycles=%0d", arglen, cycles);
    $finish;
end

endmodule
