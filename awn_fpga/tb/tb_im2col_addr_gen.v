`timescale 1ns/10ps

module tb_im2col_addr_gen;

localparam CYCLE = 20;
localparam DIM_W = 16;
localparam AW    = 14;
localparam MAX_FMAP = 16384;
localparam MAX_OUT  = 65536;

reg                clk, rst_n, start;
reg  [DIM_W-1:0]  cfg_cin, cfg_hin, cfg_win, cfg_kh, cfg_kw, cfg_pad_left, cfg_nout;
reg  [1:0]        cfg_pad_mode;
wire [AW-1:0]     o_addr;
wire               o_zero, o_valid, o_done;

im2col_addr_gen #(.AW(AW), .DIM_W(DIM_W)) DUT (
    .clk(clk), .rst_n(rst_n), .start(start),
    .cfg_cin(cfg_cin), .cfg_hin(cfg_hin), .cfg_win(cfg_win),
    .cfg_kh(cfg_kh),   .cfg_kw(cfg_kw),
    .cfg_pad_left(cfg_pad_left), .cfg_pad_mode(cfg_pad_mode),
    .cfg_nout(cfg_nout),
    .addr(o_addr), .zero_flag(o_zero), .valid(o_valid), .done(o_done)
);

initial clk = 0; always #(CYCLE/2.0) clk = ~clk;

reg [7:0] fmap [0:MAX_FMAP-1];
reg [7:0] out_data [0:MAX_OUT-1];

integer cin_a, hin_a, win_a, kh_a, kw_a, padl_a, padm_a, nout_a;
reg [511:0] fmap_path, out_path;
integer fout, i, out_cnt;
reg [31:0] cycles;

initial begin
    if (!$value$plusargs("cin=%d",  cin_a))  cin_a  = 64;
    if (!$value$plusargs("hin=%d",  hin_a))  hin_a  = 1;
    if (!$value$plusargs("win=%d",  win_a))  win_a  = 66;
    if (!$value$plusargs("kh=%d",   kh_a))   kh_a   = 1;
    if (!$value$plusargs("kw=%d",   kw_a))   kw_a   = 3;
    if (!$value$plusargs("padl=%d", padl_a)) padl_a = 0;
    if (!$value$plusargs("padm=%d", padm_a)) padm_a = 0;
    if (!$value$plusargs("nout=%d", nout_a)) nout_a = 64;
    if (!$value$plusargs("fmap=%s", fmap_path)) fmap_path = "vectors/im2col_fmap.hex";
    if (!$value$plusargs("out=%s",  out_path))  out_path  = "vectors/im2col_out.hex";

    cfg_cin      = cin_a[DIM_W-1:0];
    cfg_hin      = hin_a[DIM_W-1:0];
    cfg_win      = win_a[DIM_W-1:0];
    cfg_kh       = kh_a[DIM_W-1:0];
    cfg_kw       = kw_a[DIM_W-1:0];
    cfg_pad_left = padl_a[DIM_W-1:0];
    cfg_pad_mode = padm_a[1:0];
    cfg_nout     = nout_a[DIM_W-1:0];

    rst_n = 1; start = 0;
    $readmemh(fmap_path, fmap);

    #(2*CYCLE); rst_n = 0; #(2*CYCLE); rst_n = 1;
    @(negedge clk); start = 1; @(negedge clk); start = 0;

    out_cnt = 0;
    cycles  = 0;
    while (!o_done) begin
        @(negedge clk);
        cycles = cycles + 1;
        if (o_valid) begin
            out_data[out_cnt] = o_zero ? 8'h00 : fmap[o_addr];
            out_cnt = out_cnt + 1;
        end
        if (cycles > 500000) begin
            $display("TIMEOUT after %0d cycles", cycles); $finish;
        end
    end

    fout = $fopen(out_path, "w");
    for (i = 0; i < out_cnt; i = i + 1)
        $fwrite(fout, "%02x\n", out_data[i]);
    $fclose(fout);
    $display("im2col OK cin=%0d hin=%0d win=%0d kh=%0d kw=%0d padl=%0d padm=%0d nout=%0d elems=%0d cycles=%0d",
             cin_a, hin_a, win_a, kh_a, kw_a, padl_a, padm_a, nout_a, out_cnt, cycles);
    $finish;
end

endmodule
