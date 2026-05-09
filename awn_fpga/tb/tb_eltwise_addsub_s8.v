`timescale 1ns/10ps
`include "eltwise_addsub_s8.v"

module tb_eltwise_addsub_s8;

localparam CYCLE = 20;
localparam LEN = 8192, ADDR_W = 16;

reg               clk, rst_n, start;
reg  [ADDR_W-1:0] length;
reg               op_sel;
wire              done;

eltwise_addsub_s8 #(.LEN(LEN), .ADDR_W(ADDR_W)) DUT (
    .clk(clk), .rst_n(rst_n), .start(start),
    .length(length), .op_sel(op_sel), .done(done));

initial clk = 0; always #(CYCLE/2.0) clk = ~clk;

integer arglen, fout, k, op_arg;
reg [255:0] a_path, b_path, out_path;
reg [31:0]  cycles;

initial begin
    if (!$value$plusargs("len=%d", arglen))    arglen = 64;
    if (!$value$plusargs("op=%d",  op_arg))    op_arg = 0;
    if (!$value$plusargs("a=%s",   a_path))    a_path = "vectors/elt_a.hex";
    if (!$value$plusargs("b=%s",   b_path))    b_path = "vectors/elt_b.hex";
    if (!$value$plusargs("out=%s", out_path))  out_path = "vectors/elt_out.hex";

    length = arglen[ADDR_W-1:0];
    op_sel = op_arg[0];

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
    $display("eltwise_addsub_s8 OK len=%0d op=%0d cycles=%0d", arglen, op_arg, cycles);
    $finish;
end

endmodule
