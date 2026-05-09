`timescale 1ns/10ps
`include "lut_s8.v"

module tb_lut_s8;

localparam CYCLE = 20;
localparam LEN = 8192, ADDR_W = 16;

reg               clk, rst_n, start;
reg [ADDR_W-1:0]  length;
wire              done;

lut_s8 #(.LEN(LEN), .ADDR_W(ADDR_W)) DUT (
    .clk(clk), .rst_n(rst_n), .start(start),
    .length(length), .done(done));

initial clk = 0; always #(CYCLE/2.0) clk = ~clk;

integer arglen, fout, k;
reg [255:0] in_path, out_path, lut_path;
reg [31:0]  cycles;

initial begin
    if (!$value$plusargs("len=%d", arglen))    arglen = 64;
    if (!$value$plusargs("lut=%s", lut_path))  lut_path = "vectors/tanh_lut.hex";
    if (!$value$plusargs("in=%s",  in_path))   in_path  = "vectors/lut_in.hex";
    if (!$value$plusargs("out=%s", out_path))  out_path = "vectors/lut_out.hex";

    length = arglen[ADDR_W-1:0];

    rst_n = 1; start = 0;
    $readmemh(lut_path, DUT.lut);
    $readmemh(in_path,  DUT.in_buf);
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
    $display("lut_s8 OK len=%0d cycles=%0d", arglen, cycles);
    $finish;
end

endmodule
