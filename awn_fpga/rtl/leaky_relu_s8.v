// leaky_relu_s8: signed int8 LeakyReLU with parameterized negative slope.
//   if x >= 0: out = x
//   if x <  0: out = sat_int8( ((x * alpha_mul) + (1 << (alpha_shift-1))) >>> alpha_shift )
// alpha_mul is a 16-bit unsigned Q-format multiplier; for AWN's alpha=0.01 with
// alpha_shift=15: alpha_mul = round(0.01 * 32768) = 328.

module leaky_relu_s8 #(
    parameter LEN = 8192,
    parameter ADDR_W = 16
)(
    input                   clk,
    input                   rst_n,
    input                   start,
    input  [ADDR_W-1:0]     length,
    input  [15:0]           alpha_mul,
    input  [4:0]            alpha_shift,
    output reg              done
);

reg signed [7:0] in_buf  [0:LEN-1];
reg signed [7:0] out_buf [0:LEN-1];

reg [ADDR_W:0] i;
reg busy;

wire signed [7:0]  x       = in_buf[i[ADDR_W-1:0]];
wire signed [23:0] x_ext   = {{16{x[7]}}, x};
wire signed [39:0] prod    = x_ext * $signed({1'b0, alpha_mul});  // signed*unsigned
wire signed [39:0] half    = (alpha_shift == 0) ? 40'sd0
                                                : ($signed(1) <<< (alpha_shift - 1));
wire signed [39:0] biased  = prod + half;
wire signed [39:0] shifted = biased >>> alpha_shift;
wire signed [7:0]  neg_out = (shifted >  127) ? 8'sd127
                           : (shifted < -128) ? -8'sd128
                                              : shifted[7:0];

integer k;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        busy <= 1'b0;
        done <= 1'b0;
        i    <= 0;
        for (k = 0; k < LEN; k = k + 1) out_buf[k] <= 8'sd0;
    end else begin
        done <= 1'b0;
        if (start && !busy) begin
            busy <= 1'b1; i <= 0;
        end else if (busy) begin
            if (i < {1'b0, length}) begin
                out_buf[i[ADDR_W-1:0]] <= (x >= 0) ? x : neg_out;
                i <= i + 1;
            end else begin
                busy <= 1'b0;
                done <= 1'b1;
            end
        end
    end
end

endmodule
