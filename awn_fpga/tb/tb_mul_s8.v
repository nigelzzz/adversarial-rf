`timescale 1ns/10ps
`include "mul_s8.v"

module tb_mul_s8;

localparam CYCLE = 20;
localparam LEN = 8192, ADDR_W = 16;

reg               clk, rst_n, start;
reg  [ADDR_W-1:0] length;
reg  signed [31:0] mul_q;
reg         [5:0]  shift;
wire              done;

mul_s8 #(.LEN(LEN), .ADDR_W(ADDR_W)) DUT (
    .clk(clk), .rst_n(rst_n), .start(start),
    .length(length), .mul_q(mul_q), .shift(shift), .done(done));

initial clk = 0; always #(CYCLE/2.0) clk = ~clk;

integer arglen, fout, k, mul_arg, shift_arg;
reg [255:0] a_path, b_path, out_path;
reg [31:0]  cycles;

initial begin
    if (!$value$plusargs("len=%d", arglen))    arglen = 64;
    if (!$value$plusargs("mul=%d", mul_arg))   mul_arg = 1073741824;
    if (!$value$plusargs("shift=%d", shift_arg)) shift_arg = 31;
    if (!$value$plusargs("a=%s",   a_path))    a_path = "vectors/mul_a.hex";
    if (!$value$plusargs("b=%s",   b_path))    b_path = "vectors/mul_b.hex";
    if (!$value$plusargs("out=%s", out_path))  out_path = "vectors/mul_out.hex";

    length = arglen[ADDR_W-1:0];
    mul_q = mul_arg;
    shift = shift_arg[5:0];

    rst_n = 1; start = 0;
    $readmemh(a_path, DUT.a_buf);
    $readmemh(b_path, DUT.b_buf);
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
    $display("mul_s8 OK len=%0d cycles=%0d", arglen, cycles);
    $finish;
end

endmodule
