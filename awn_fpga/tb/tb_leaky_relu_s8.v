`timescale 1ns/10ps
`include "leaky_relu_s8.v"

module tb_leaky_relu_s8;

localparam CYCLE = 20;
localparam LEN = 8192, ADDR_W = 16;

reg                clk, rst_n, start;
reg  [ADDR_W-1:0]  length;
reg  [15:0]        alpha_mul;
reg  [4:0]         alpha_shift;
wire               done;

leaky_relu_s8 #(.LEN(LEN), .ADDR_W(ADDR_W)) DUT (
    .clk(clk), .rst_n(rst_n), .start(start),
    .length(length), .alpha_mul(alpha_mul), .alpha_shift(alpha_shift),
    .done(done));

initial clk = 0; always #(CYCLE/2.0) clk = ~clk;

integer arglen, fout, k, am, ash;
reg [255:0] in_path, out_path;
reg [31:0]  cycles;

initial begin
    if (!$value$plusargs("len=%d", arglen))   arglen = 64;
    if (!$value$plusargs("amul=%d", am))      am = 328;
    if (!$value$plusargs("ashift=%d", ash))   ash = 15;
    if (!$value$plusargs("in=%s",  in_path))  in_path  = "vectors/lrelu_in.hex";
    if (!$value$plusargs("out=%s", out_path)) out_path = "vectors/lrelu_out.hex";

    length = arglen[ADDR_W-1:0];
    alpha_mul = am[15:0];
    alpha_shift = ash[4:0];

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
    $display("leaky_relu_s8 OK len=%0d cycles=%0d", arglen, cycles);
    $finish;
end

endmodule
