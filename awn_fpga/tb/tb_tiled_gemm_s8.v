`timescale 1ns/10ps

module tb_tiled_gemm_s8;

localparam CYCLE = 20;
localparam DIM_W = 16;

reg                 clk, rst_n, start;
reg  [DIM_W-1:0]    M_in, K_in, N_in;
reg                 use_bias;
wire                done;

tiled_gemm_s8 #(.DIM_W(DIM_W)) DUT (
    .clk(clk), .rst_n(rst_n), .start(start),
    .M_in(M_in), .K_in(K_in), .N_in(N_in),
    .use_bias(use_bias), .done(done));

initial clk = 0; always #(CYCLE/2.0) clk = ~clk;

integer M_arg, K_arg, N_arg, bias_arg, fout, k;
reg [255:0] a_path, b_path, bias_path, out_path;
reg [31:0]  cycles;
integer mn;

initial begin
    if (!$value$plusargs("M=%d", M_arg))      M_arg = 4;
    if (!$value$plusargs("K=%d", K_arg))      K_arg = 4;
    if (!$value$plusargs("N=%d", N_arg))      N_arg = 4;
    if (!$value$plusargs("bias=%d", bias_arg)) bias_arg = 0;
    if (!$value$plusargs("a=%s",   a_path))    a_path = "vectors/a.hex";
    if (!$value$plusargs("b=%s",   b_path))    b_path = "vectors/b.hex";
    if (!$value$plusargs("bi=%s",  bias_path)) bias_path = "vectors/bi.hex";
    if (!$value$plusargs("out=%s", out_path))  out_path = "vectors/out.hex";

    M_in = M_arg[DIM_W-1:0];
    K_in = K_arg[DIM_W-1:0];
    N_in = N_arg[DIM_W-1:0];
    use_bias = bias_arg[0];

    rst_n = 1; start = 0;
    $readmemh(a_path, DUT.a_buf);
    $readmemh(b_path, DUT.b_buf);
    if (bias_arg) $readmemh(bias_path, DUT.bias_buf);

    #(2*CYCLE); rst_n = 0; #(2*CYCLE); rst_n = 1;
    @(negedge clk); start = 1; @(negedge clk); start = 0;

    cycles = 0;
    while (!done) begin
        cycles = cycles + 1;
        if (cycles > 200_000_000) begin
            $display("TIMEOUT after %0d cycles", cycles); $finish;
        end
        @(negedge clk);
    end

    mn = M_arg * N_arg;
    fout = $fopen(out_path, "w");
    for (k = 0; k < mn; k = k + 1)
        $fwrite(fout, "%08x\n", DUT.c_buf[k] & 32'hffffffff);
    $fclose(fout);
    $display("tiled_gemm_s8 OK M=%0d K=%0d N=%0d bias=%0d cycles=%0d",
             M_arg, K_arg, N_arg, bias_arg, cycles);
    $finish;
end

endmodule
