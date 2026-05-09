// mul_s8: int8 * int8 -> int8, with TFLite-style requant.
//   prod = a * b                        (int16, signed)
//   out  = sat_int8( ((prod*mul) + half) >>> shift )
// Symmetric quantization assumed (zp = 0).

module mul_s8 #(
    parameter LEN = 8192,
    parameter ADDR_W = 16
)(
    input                   clk,
    input                   rst_n,
    input                   start,
    input  [ADDR_W-1:0]     length,
    input  signed [31:0]    mul_q,
    input         [5:0]     shift,
    output reg              done
);

reg signed [7:0] a_buf   [0:LEN-1];
reg signed [7:0] b_buf   [0:LEN-1];
reg signed [7:0] out_buf [0:LEN-1];

reg [ADDR_W:0] i;
reg busy;

wire signed [15:0] prod16 = a_buf[i[ADDR_W-1:0]] * b_buf[i[ADDR_W-1:0]];
wire signed [63:0] prod   = $signed({{48{prod16[15]}}, prod16}) * mul_q;
wire signed [63:0] half   = (shift == 0) ? 64'sd0
                                         : ($signed(64'd1) <<< (shift - 1));
wire signed [63:0] sh     = (prod + half) >>> shift;
wire signed [7:0]  sat    = (sh >  127) ? 8'sd127
                         : (sh < -128) ? -8'sd128
                                       : sh[7:0];

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
                out_buf[i[ADDR_W-1:0]] <= sat;
                i <= i + 1;
            end else begin busy <= 0; done <= 1; end
        end
    end
end

endmodule
