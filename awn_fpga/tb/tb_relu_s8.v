`timescale 1ns/10ps
`include "relu_s8.v"

module tb_relu_s8;

localparam CYCLE = 20;
localparam LEN = 8192;
localparam ADDR_W = 16;

reg                 clk;
reg                 rst_n;
reg                 start;
reg  [ADDR_W-1:0]   length;
wire                done;

relu_s8 #(.LEN(LEN), .ADDR_W(ADDR_W)) DUT (
    .clk(clk), .rst_n(rst_n),
    .start(start), .length(length), .done(done)
);

initial clk = 0;
always #(CYCLE/2.0) clk = ~clk;

integer arglen, code, fout, k;
reg [255:0] in_path, out_path;
reg [31:0]  cycles;

initial begin
    $dumpfile("build/wave_relu_s8.vcd");
    $dumpvars(0, tb_relu_s8);
end

initial begin
    rst_n = 1; start = 0; length = 0;

    // --- args via $value$plusargs (e.g. +len=128 +in=... +out=...)
    if (!$value$plusargs("len=%d", arglen)) arglen = 64;
    if (!$value$plusargs("in=%s",  in_path))  in_path  = "vectors/relu_in.hex";
    if (!$value$plusargs("out=%s", out_path)) out_path = "vectors/relu_out.hex";

    length = arglen[ADDR_W-1:0];

    // load input
    $readmemh(in_path, DUT.in_buf);

    // reset
    #(2*CYCLE); rst_n = 0;
    #(2*CYCLE); rst_n = 1;
    @(negedge clk);

    // start pulse
    start = 1; @(negedge clk);
    start = 0;

    // wait for done with timeout
    cycles = 0;
    while (!done) begin
        cycles = cycles + 1;
        if (cycles > 5*LEN + 1000) begin
            $display("TIMEOUT after %0d cycles", cycles);
            $finish;
        end
        @(negedge clk);
    end

    // dump output
    fout = $fopen(out_path, "w");
    if (fout == 0) begin $display("cannot open %0s", out_path); $finish; end
    for (k = 0; k < arglen; k = k + 1) begin
        $fwrite(fout, "%02x\n", DUT.out_buf[k] & 8'hff);
    end
    $fclose(fout);

    $display("relu_s8 OK len=%0d cycles=%0d", arglen, cycles);
    $finish;
end

endmodule
