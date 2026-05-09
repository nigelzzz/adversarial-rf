// relu_s8: signed int8 ReLU. Assumes symmetric quantization (zero-point = 0).
//   out[i] = (in[i] > 0) ? in[i] : 0
//
// Buffers (in_buf, out_buf) are internal regs; the testbench $readmemh's the
// input directly into hierarchical name DUT.in_buf and $writememh's out_buf.
//
// Handshake: pulse `start` for one cycle while length is valid; wait for `done`.

module relu_s8 #(
    parameter LEN = 8192,
    parameter ADDR_W = 16
)(
    input                  clk,
    input                  rst_n,
    input                  start,
    input  [ADDR_W-1:0]    length,
    output reg             done
);

reg signed [7:0] in_buf  [0:LEN-1];
reg signed [7:0] out_buf [0:LEN-1];

reg [ADDR_W:0] i;
reg busy;

integer k;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        busy <= 1'b0;
        done <= 1'b0;
        i    <= {(ADDR_W+1){1'b0}};
        for (k = 0; k < LEN; k = k + 1) out_buf[k] <= 8'sd0;
    end else begin
        done <= 1'b0;
        if (start && !busy) begin
            busy <= 1'b1;
            i    <= {(ADDR_W+1){1'b0}};
        end else if (busy) begin
            if (i < {1'b0, length}) begin
                out_buf[i[ADDR_W-1:0]] <= (in_buf[i[ADDR_W-1:0]] > 0)
                                          ? in_buf[i[ADDR_W-1:0]]
                                          : 8'sd0;
                i <= i + 1;
            end else begin
                busy <= 1'b0;
                done <= 1'b1;
            end
        end
    end
end

endmodule
