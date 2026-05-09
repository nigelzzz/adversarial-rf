`timescale 1ns/10ps
`include "avgpool1d_s8.v"

module tb_avgpool1d_s8;

localparam CYCLE = 20;
localparam LEN = 16384, ADDR_W = 16, CHAN_W = 12, POOL_W = 12;

reg                 clk, rst_n, start;
reg [CHAN_W-1:0]    channels;
reg [POOL_W-1:0]    pool_len;
reg signed [31:0]   mul_q;
reg [5:0]           shift;
wire                done;

avgpool1d_s8 #(.LEN(LEN), .ADDR_W(ADDR_W), .CHAN_W(CHAN_W), .POOL_W(POOL_W)) DUT (
    .clk(clk), .rst_n(rst_n), .start(start),
    .channels(channels), .pool_len(pool_len),
    .mul_q(mul_q), .shift(shift), .done(done));

initial clk = 0; always #(CYCLE/2.0) clk = ~clk;

integer chans_arg, plen_arg, mul_arg, shift_arg, fout, k;
reg [255:0] in_path, out_path;
reg [31:0]  cycles;

initial begin
    if (!$value$plusargs("chans=%d", chans_arg)) chans_arg = 64;
    if (!$value$plusargs("plen=%d",  plen_arg))  plen_arg = 64;
    if (!$value$plusargs("mul=%d",   mul_arg))   mul_arg = 1073741824;
    if (!$value$plusargs("shift=%d", shift_arg)) shift_arg = 31;
    if (!$value$plusargs("in=%s",  in_path))  in_path  = "vectors/ap_in.hex";
    if (!$value$plusargs("out=%s", out_path)) out_path = "vectors/ap_out.hex";

    channels = chans_arg[CHAN_W-1:0];
    pool_len = plen_arg[POOL_W-1:0];
    mul_q = mul_arg;
    shift = shift_arg[5:0];

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
    for (k = 0; k < chans_arg; k = k + 1)
        $fwrite(fout, "%02x\n", DUT.out_buf[k] & 8'hff);
    $fclose(fout);
    $display("avgpool1d_s8 OK chans=%0d plen=%0d cycles=%0d",
             chans_arg, plen_arg, cycles);
    $finish;
end

endmodule
