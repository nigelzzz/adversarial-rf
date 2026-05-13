// systolic_mesh_s8: 8x16 output-stationary systolic array for int8 GEMM.
// Interface matches gemm_s8.v so existing testbenches work unchanged.
// Single-tile only (M<=8, N<=16, arbitrary K). Full tiling in S02.
module systolic_mesh_s8 #(
    parameter PM       = 8,
    parameter PN       = 16,
    parameter A_LEN    = 65536,
    parameter B_LEN    = 65536,
    parameter C_LEN    = 16384,
    parameter BIAS_LEN = 1024,
    parameter DIM_W    = 16
)(
    input                clk,
    input                rst_n,
    input                start,
    input  [DIM_W-1:0]   M_in,
    input  [DIM_W-1:0]   K_in,
    input  [DIM_W-1:0]   N_in,
    input                use_bias,
    output reg           done
);

    // Testbench accesses these via DUT.a_buf etc. — names must match gemm_s8.v
    reg signed [7:0]  a_buf    [0:A_LEN-1];
    reg signed [7:0]  b_buf    [0:B_LEN-1];
    reg signed [31:0] bias_buf [0:BIAS_LEN-1];
    reg signed [31:0] c_buf    [0:C_LEN-1];

    // FSM states
    localparam S_IDLE    = 2'd0;
    localparam S_COMPUTE = 2'd1;
    localparam S_DRAIN   = 2'd2;
    localparam S_DONE    = 2'd3;

    reg [1:0]        state;
    reg [DIM_W-1:0]  cycle_cnt;
    reg [DIM_W-1:0]  K_reg, M_reg, N_reg;
    reg              use_bias_reg;
    reg [DIM_W-1:0]  drain_m, drain_n;

    // Total compute cycles needed: K + PM + PN - 2 (wavefront fill + drain)
    wire [DIM_W-1:0] total_cycles = K_reg + PM[DIM_W-1:0] + PN[DIM_W-1:0] - 16'd2;

    // PE control signals
    reg pe_en;
    reg pe_acc_clear;

    // Flat 1D accumulator output array for reliable variable-index reads
    wire signed [31:0] pe_acc [0:PM*PN-1];

    // Horizontal (activation) wires: a_wire[row][0] = left-edge input
    wire signed [7:0] a_wire [0:PM-1][0:PN];
    // Vertical (weight) wires: b_wire[0][col] = top-edge input
    wire signed [7:0] b_wire [0:PM][0:PN-1];

    // --- PE grid ---
    genvar gi, gj;
    generate
        for (gi = 0; gi < PM; gi = gi + 1) begin : row
            for (gj = 0; gj < PN; gj = gj + 1) begin : col
                pe_s8 pe_inst (
                    .clk      (clk),
                    .rst_n    (rst_n),
                    .en       (pe_en),
                    .acc_clear(pe_acc_clear),
                    .a_in     (a_wire[gi][gj]),
                    .b_in     (b_wire[gi][gj]),
                    .a_out    (a_wire[gi][gj+1]),
                    .b_out    (b_wire[gi+1][gj]),
                    .acc      (pe_acc[gi*PN+gj])
                );
            end
        end
    endgenerate

    // --- Left-edge feeding (skewed by row index) ---
    // Row m feeds A[m, t-m] at compute cycle t, valid when t >= m and t-m < K
    generate
        for (gi = 0; gi < PM; gi = gi + 1) begin : a_feed
            wire [DIM_W-1:0] a_offset = cycle_cnt - gi[DIM_W-1:0];
            wire [31:0]      a_addr   = gi[DIM_W-1:0] * K_reg + a_offset;
            wire             a_valid  = (state == S_COMPUTE) &&
                                        (cycle_cnt >= gi[DIM_W-1:0]) &&
                                        (a_offset < K_reg);
            assign a_wire[gi][0] = a_valid ? a_buf[a_addr[15:0]] : 8'sd0;
        end
    endgenerate

    // --- Top-edge feeding (skewed by column index) ---
    // Col n feeds B[t-n, n] at compute cycle t, valid when t >= n and t-n < K
    generate
        for (gj = 0; gj < PN; gj = gj + 1) begin : b_feed
            wire [DIM_W-1:0] b_offset = cycle_cnt - gj[DIM_W-1:0];
            wire [31:0]      b_addr   = b_offset * N_reg + gj[DIM_W-1:0];
            wire             b_valid  = (state == S_COMPUTE) &&
                                        (cycle_cnt >= gj[DIM_W-1:0]) &&
                                        (b_offset < K_reg);
            assign b_wire[0][gj] = b_valid ? b_buf[b_addr[15:0]] : 8'sd0;
        end
    endgenerate

    // --- FSM ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            done         <= 1'b0;
            pe_en        <= 1'b0;
            pe_acc_clear <= 1'b0;
            cycle_cnt    <= {DIM_W{1'b0}};
            K_reg        <= {DIM_W{1'b0}};
            M_reg        <= {DIM_W{1'b0}};
            N_reg        <= {DIM_W{1'b0}};
            use_bias_reg <= 1'b0;
            drain_m      <= {DIM_W{1'b0}};
            drain_n      <= {DIM_W{1'b0}};
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        M_reg        <= M_in;
                        K_reg        <= K_in;
                        N_reg        <= N_in;
                        use_bias_reg <= use_bias;
                        cycle_cnt    <= {DIM_W{1'b0}};
                        pe_en        <= 1'b1;
                        pe_acc_clear <= 1'b1;
                        state        <= S_COMPUTE;
                    end
                end

                S_COMPUTE: begin
                    if (cycle_cnt == {DIM_W{1'b0}})
                        pe_acc_clear <= 1'b0;

                    if (cycle_cnt == total_cycles) begin
                        pe_en   <= 1'b0;
                        drain_m <= {DIM_W{1'b0}};
                        drain_n <= {DIM_W{1'b0}};
                        state   <= S_DRAIN;
                    end else begin
                        cycle_cnt <= cycle_cnt + 1'b1;
                    end
                end

                S_DRAIN: begin
                    c_buf[drain_m * N_reg + drain_n] <=
                        pe_acc[drain_m[3:0]*PN + drain_n[4:0]] +
                        (use_bias_reg ? bias_buf[drain_m] : 32'sd0);

                    if (drain_n == N_reg - 1'b1) begin
                        drain_n <= {DIM_W{1'b0}};
                        if (drain_m == M_reg - 1'b1)
                            state <= S_DONE;
                        else
                            drain_m <= drain_m + 1'b1;
                    end else begin
                        drain_n <= drain_n + 1'b1;
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
