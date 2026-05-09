// lut_s8: 256-entry int8-in / int8-out lookup table. Used for tanh, sigmoid,
// and any other element-wise nonlinearity. Index = in[i] + 128 (so signed -128
// maps to entry 0, 127 maps to entry 255).
// LUT is loaded by the testbench via $readmemh into DUT.lut.

module lut_s8 #(
    parameter LEN = 8192,
    parameter ADDR_W = 16
)(
    input                clk,
    input                rst_n,
    input                start,
    input  [ADDR_W-1:0]  length,
    output reg           done
);

reg signed [7:0] in_buf  [0:LEN-1];
reg signed [7:0] out_buf [0:LEN-1];
reg signed [7:0] lut     [0:255];

reg [ADDR_W:0] i;
reg busy;

wire [7:0] idx = in_buf[i[ADDR_W-1:0]] + 8'sd128;  // -128..127 -> 0..255

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
                out_buf[i[ADDR_W-1:0]] <= lut[idx];
                i <= i + 1;
            end else begin busy <= 0; done <= 1; end
        end
    end
end

endmodule
