// pe_s8: single processing element for int8 output-stationary systolic array.
// int8x int8 MAC with int32 accumulator; registered passthrough (1-cycle latency).
module pe_s8 (
    input               clk,
    input               rst_n,
    input               en,
    input               acc_clear,
    input  signed [7:0] a_in,
    input  signed [7:0] b_in,
    output reg signed [7:0]  a_out,
    output reg signed [7:0]  b_out,
    output reg signed [31:0] acc
);

wire signed [15:0] prod = a_in * b_in;
wire signed [31:0] prod32 = {{16{prod[15]}}, prod};

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        a_out <= 8'sd0;
        b_out <= 8'sd0;
        acc   <= 32'sd0;
    end else if (en) begin
        a_out <= a_in;
        b_out <= b_in;
        acc   <= acc_clear ? prod32 : (acc + prod32);
    end
end

endmodule
