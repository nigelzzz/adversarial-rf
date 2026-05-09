// gemm_s8: int8 matrix multiply with optional int32 bias.
//   A is row-major M-by-K  (a_buf[m*K + k] = A[m,k])
//   B is row-major K-by-N  (b_buf[k*N + n] = B[k,n])
//   bias is length M       (broadcast across N)
//   C is row-major M-by-N  (c_buf[m*N + n] = sum_k A[m,k]*B[k,n] + bias[m])
//
// Behavioral one-MAC-per-cycle implementation. For AWN's largest layer
// (conv2 im2col: M=64, K=320, N=128) this is ~2.6M cycles per call;
// iverilog runs it in well under a minute.

module gemm_s8 #(
    parameter A_LEN    = 65536,
    parameter B_LEN    = 65536,
    parameter C_LEN    = 16384,
    parameter BIAS_LEN = 1024,
    parameter DIM_W    = 16
)(
    input                   clk,
    input                   rst_n,
    input                   start,
    input  [DIM_W-1:0]      M_in,
    input  [DIM_W-1:0]      K_in,
    input  [DIM_W-1:0]      N_in,
    input                   use_bias,
    output reg              done
);

reg signed [7:0]  a_buf    [0:A_LEN-1];
reg signed [7:0]  b_buf    [0:B_LEN-1];
reg signed [31:0] bias_buf [0:BIAS_LEN-1];
reg signed [31:0] c_buf    [0:C_LEN-1];

reg [DIM_W-1:0]   m, n;
reg [DIM_W:0]     k;       // one extra bit so k can equal K_in to terminate
reg signed [31:0] acc;

reg [2:0] state;
localparam S_IDLE  = 3'd0;
localparam S_INIT  = 3'd1;
localparam S_MAC   = 3'd2;
localparam S_WRITE = 3'd3;
localparam S_NEXT  = 3'd4;
localparam S_DONE  = 3'd5;

wire [31:0] a_idx = m * K_in + k[DIM_W-1:0];
wire [31:0] b_idx = k[DIM_W-1:0] * N_in + n;
wire [31:0] c_idx = m * N_in + n;

wire signed [15:0] prod16 = $signed(a_buf[a_idx[15:0]])
                          * $signed(b_buf[b_idx[15:0]]);
wire signed [31:0] prod32 = {{16{prod16[15]}}, prod16};

integer kk;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= S_IDLE;
        done  <= 1'b0;
        m <= 0; n <= 0; k <= 0;
        acc <= 0;
        for (kk = 0; kk < C_LEN; kk = kk + 1) c_buf[kk] <= 0;
    end else begin
        done <= 1'b0;
        case (state)
            S_IDLE: begin
                if (start) begin
                    m <= 0; n <= 0; k <= 0;
                    state <= S_INIT;
                end
            end
            S_INIT: begin
                acc   <= use_bias ? bias_buf[m] : 32'sd0;
                k     <= 0;
                state <= S_MAC;
            end
            S_MAC: begin
                if (k < {1'b0, K_in}) begin
                    acc <= acc + prod32;
                    k   <= k + 1;
                end else begin
                    state <= S_WRITE;
                end
            end
            S_WRITE: begin
                c_buf[c_idx[DIM_W:0]] <= acc;
                state <= S_NEXT;
            end
            S_NEXT: begin
                if (n + 1 < N_in) begin
                    n <= n + 1;
                    state <= S_INIT;
                end else begin
                    n <= 0;
                    if (m + 1 < M_in) begin
                        m <= m + 1;
                        state <= S_INIT;
                    end else begin
                        state <= S_DONE;
                    end
                end
            end
            S_DONE: begin
                done  <= 1'b1;
                state <= S_IDLE;
            end
            default: state <= S_IDLE;
        endcase
    end
end

endmodule
