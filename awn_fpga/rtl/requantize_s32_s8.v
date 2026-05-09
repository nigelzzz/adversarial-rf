// requantize_s32_s8: int32 acc -> int8 with TFLite-style fixed-point requant.
//   out = sat_int8( round((acc * mul) >> shift) + out_zp )
// mul is signed 32-bit (typically positive Q-format mantissa), shift unsigned.
// act_mode: 0 = identity, 1 = ReLU clamp at out_zp, 2 = ReLU6 (out_zp..out_zp+max6)
//
// Used as the post-GEMM stage and anywhere int32 accumulators need to be
// projected back into int8 activation space.

module requantize_s32_s8 #(
    parameter LEN = 8192,
    parameter ADDR_W = 16
)(
    input                   clk,
    input                   rst_n,
    input                   start,
    input  [ADDR_W-1:0]     length,
    input  signed [31:0]    mul,
    input         [5:0]     shift,
    input  signed [7:0]     out_zp,
    input         [1:0]     act_mode,   // 0 none, 1 relu
    output reg              done
);

reg signed [31:0] in_buf  [0:LEN-1];
reg signed [7:0]  out_buf [0:LEN-1];

reg [ADDR_W:0] i;
reg busy;

wire signed [31:0] acc = in_buf[i[ADDR_W-1:0]];
wire signed [63:0] prod = acc * mul;
wire signed [63:0] half = (shift == 0) ? 64'sd0 : ($signed(64'd1) <<< (shift - 1));
wire signed [63:0] shifted = (prod + half) >>> shift;
wire signed [63:0] biased  = shifted + $signed({{56{out_zp[7]}}, out_zp});
wire signed [7:0]  raw     = (biased >  127) ? 8'sd127
                          : (biased < -128) ? -8'sd128
                                            : biased[7:0];
wire signed [7:0]  acted   = (act_mode == 2'd1 && raw < out_zp) ? out_zp : raw;

integer k;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        busy <= 0; done <= 0; i <= 0;
        for (k = 0; k < LEN; k = k + 1) out_buf[k] <= 0;
    end else begin
        done <= 0;
        if (start && !busy) begin busy <= 1; i <= 0; end
        else if (busy) begin
            if (i < {1'b0, length}) begin
                out_buf[i[ADDR_W-1:0]] <= acted;
                i <= i + 1;
            end else begin
                busy <= 0; done <= 1;
            end
        end
    end
end

endmodule
