// eltwise_addsub_s8: int8 + int8 -> int8 OR int8 - int8 -> int8 with saturation.
// Assumes both inputs share the same scale and zero-point=0 (symmetric).
// op_sel: 0 -> add, 1 -> sub.

module eltwise_addsub_s8 #(
    parameter LEN = 8192,
    parameter ADDR_W = 16
)(
    input                   clk,
    input                   rst_n,
    input                   start,
    input  [ADDR_W-1:0]     length,
    input                   op_sel,
    output reg              done
);

reg signed [7:0] a_buf   [0:LEN-1];
reg signed [7:0] b_buf   [0:LEN-1];
reg signed [7:0] out_buf [0:LEN-1];

reg [ADDR_W:0] i;
reg busy;

wire signed [8:0] a_ext = {{1{a_buf[i[ADDR_W-1:0]][7]}}, a_buf[i[ADDR_W-1:0]]};
wire signed [8:0] b_ext = {{1{b_buf[i[ADDR_W-1:0]][7]}}, b_buf[i[ADDR_W-1:0]]};
wire signed [9:0] sum   = (op_sel == 1'b0) ? (a_ext + b_ext) : (a_ext - b_ext);
wire signed [7:0] sat   = (sum >  127) ? 8'sd127
                       : (sum < -128) ? -8'sd128
                                      : sum[7:0];

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
