// tiled_gemm_s8: tile-sequencing wrapper around an 8x16 PE grid.
// Decomposes arbitrary M×K×N GEMMs into ceil(M/8)×ceil(N/16) tiles,
// processing each sequentially with boundary zero-padding.
// Weight double-buffering: two bram_feeder_b banks hide B-tile load latency.
module tiled_gemm_s8 #(
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

    reg signed [7:0]  a_buf    [0:A_LEN-1];
    reg signed [7:0]  b_buf    [0:B_LEN-1];
    reg signed [31:0] bias_buf [0:BIAS_LEN-1];
    reg signed [31:0] c_buf    [0:C_LEN-1];

    // FSM states
    localparam S_IDLE       = 3'd0;
    localparam S_LOAD_B0    = 3'd1;
    localparam S_COMPUTE    = 3'd2;
    localparam S_DRAIN      = 3'd3;
    localparam S_NEXT       = 3'd4;
    localparam S_DONE       = 3'd5;

    reg [2:0] state;

    // Dimension registers
    reg [DIM_W-1:0] M_reg, K_reg, N_reg;
    reg             use_bias_reg;

    // Tiling registers
    reg [DIM_W-1:0] mt, nt;
    reg [DIM_W-1:0] mt_count, nt_count;
    reg [DIM_W-1:0] m_base, n_base;
    reg [DIM_W-1:0] M_tile, N_tile;

    // Compute/drain control
    reg [DIM_W-1:0] cycle_cnt;
    reg [DIM_W-1:0] drain_m, drain_n;
    reg pe_en, pe_acc_clear;

    // Double-buffering registers
    reg              active_bank;
    reg [DIM_W-1:0]  load_k;
    reg              loading;
    reg [DIM_W-1:0]  load_n_base;

    wire [DIM_W-1:0] total_cycles = K_reg + PM[DIM_W-1:0] + PN[DIM_W-1:0] - 16'd2;

    // --- BRAM banks for B-matrix double-buffering ---
    localparam AW = 9;

    // Write data assembly (combinational from b_buf)
    wire [DIM_W-1:0] cur_load_nb = (state == S_LOAD_B0) ? n_base : load_n_base;
    reg [PN*8-1:0] wr_data_flat_r;
    integer ci;
    always @(*) begin
        for (ci = 0; ci < PN; ci = ci + 1) begin
            if (cur_load_nb + ci[DIM_W-1:0] < N_reg)
                wr_data_flat_r[ci*8 +: 8] =
                    b_buf[(load_k * N_reg + cur_load_nb + ci[DIM_W-1:0]) & 16'hFFFF];
            else
                wr_data_flat_r[ci*8 +: 8] = 8'd0;
        end
    end
    wire [PN*8-1:0] wr_data_flat = wr_data_flat_r;

    // Write enable logic
    wire load_active   = (state == S_LOAD_B0) || (state == S_COMPUTE && loading);
    wire load_to_bank1 = (state == S_LOAD_B0) ? 1'b0 : ~active_bank;
    wire wr_en0 = load_active && !load_to_bank1;
    wire wr_en1 = load_active &&  load_to_bank1;
    wire [AW-1:0] wr_k_wire = load_k[AW-1:0];

    // Per-column read addresses (skewed by column index for systolic timing)
    wire [PN*AW-1:0] b_rd_rows;
    genvar gk;
    generate
        for (gk = 0; gk < PN; gk = gk + 1) begin : b_addr_gen
            wire [DIM_W-1:0] b_off_k = cycle_cnt - gk[DIM_W-1:0];
            assign b_rd_rows[gk*AW +: AW] = b_off_k[AW-1:0];
        end
    endgenerate

    // Two BRAM banks
    wire [PN*8-1:0] b_rd_data0, b_rd_data1;

    bram_feeder_b #(.COLS(PN), .DEPTH(512), .AW(AW)) bank0 (
        .clk(clk), .wr_en(wr_en0), .wr_k(wr_k_wire),
        .wr_data_flat(wr_data_flat),
        .rd_rows_flat(b_rd_rows), .rd_datas_flat(b_rd_data0)
    );
    bram_feeder_b #(.COLS(PN), .DEPTH(512), .AW(AW)) bank1 (
        .clk(clk), .wr_en(wr_en1), .wr_k(wr_k_wire),
        .wr_data_flat(wr_data_flat),
        .rd_rows_flat(b_rd_rows), .rd_datas_flat(b_rd_data1)
    );

    // --- PE grid ---
    wire signed [31:0] pe_acc [0:PM*PN-1];
    wire signed [7:0] a_wire [0:PM-1][0:PN];
    wire signed [7:0] b_wire [0:PM][0:PN-1];

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

    // --- Left-edge feeding with tile offsets and boundary check ---
    generate
        for (gi = 0; gi < PM; gi = gi + 1) begin : a_feed
            wire [DIM_W-1:0] a_offset = cycle_cnt - gi[DIM_W-1:0];
            wire [31:0]      a_addr   = (m_base + gi[DIM_W-1:0]) * K_reg + a_offset;
            wire             a_valid  = (state == S_COMPUTE) &&
                                        (cycle_cnt >= gi[DIM_W-1:0]) &&
                                        (a_offset < K_reg) &&
                                        (gi[DIM_W-1:0] < M_tile);
            assign a_wire[gi][0] = a_valid ? a_buf[a_addr[15:0]] : 8'sd0;
        end
    endgenerate

    // --- Top-edge feeding from BRAM banks ---
    generate
        for (gj = 0; gj < PN; gj = gj + 1) begin : b_feed
            wire [DIM_W-1:0] b_offset = cycle_cnt - gj[DIM_W-1:0];
            wire             b_valid  = (state == S_COMPUTE) &&
                                        (cycle_cnt >= gj[DIM_W-1:0]) &&
                                        (b_offset < K_reg) &&
                                        (gj[DIM_W-1:0] < N_tile);
            wire signed [7:0] b_val = active_bank
                ? $signed(b_rd_data1[gj*8 +: 8])
                : $signed(b_rd_data0[gj*8 +: 8]);
            assign b_wire[0][gj] = b_valid ? b_val : 8'sd0;
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
            M_reg        <= {DIM_W{1'b0}};
            K_reg        <= {DIM_W{1'b0}};
            N_reg        <= {DIM_W{1'b0}};
            use_bias_reg <= 1'b0;
            mt           <= {DIM_W{1'b0}};
            nt           <= {DIM_W{1'b0}};
            mt_count     <= {DIM_W{1'b0}};
            nt_count     <= {DIM_W{1'b0}};
            m_base       <= {DIM_W{1'b0}};
            n_base       <= {DIM_W{1'b0}};
            M_tile       <= {DIM_W{1'b0}};
            N_tile       <= {DIM_W{1'b0}};
            drain_m      <= {DIM_W{1'b0}};
            drain_n      <= {DIM_W{1'b0}};
            active_bank  <= 1'b0;
            load_k       <= {DIM_W{1'b0}};
            loading      <= 1'b0;
            load_n_base  <= {DIM_W{1'b0}};
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        M_reg        <= M_in;
                        K_reg        <= K_in;
                        N_reg        <= N_in;
                        use_bias_reg <= use_bias;
                        mt_count     <= (M_in + 16'd7) >> 3;
                        nt_count     <= (N_in + 16'd15) >> 4;
                        mt           <= {DIM_W{1'b0}};
                        nt           <= {DIM_W{1'b0}};
                        m_base       <= {DIM_W{1'b0}};
                        n_base       <= {DIM_W{1'b0}};
                        M_tile       <= (M_in >= PM[DIM_W-1:0]) ? PM[DIM_W-1:0] : M_in;
                        N_tile       <= (N_in >= PN[DIM_W-1:0]) ? PN[DIM_W-1:0] : N_in;
                        load_k       <= {DIM_W{1'b0}};
                        active_bank  <= 1'b0;
                        loading      <= 1'b0;
                        state        <= S_LOAD_B0;
                    end
                end

                S_LOAD_B0: begin
                    if (load_k == K_reg - 1'b1) begin
                        active_bank  <= 1'b0;
                        cycle_cnt    <= {DIM_W{1'b0}};
                        pe_en        <= 1'b1;
                        pe_acc_clear <= 1'b1;
                        if (nt_count > 16'd1) begin
                            loading     <= 1'b1;
                            load_k      <= {DIM_W{1'b0}};
                            load_n_base <= n_base + PN[DIM_W-1:0];
                        end else begin
                            loading <= 1'b0;
                        end
                        state <= S_COMPUTE;
                    end else begin
                        load_k <= load_k + 1'b1;
                    end
                end

                S_COMPUTE: begin
                    if (cycle_cnt == {DIM_W{1'b0}})
                        pe_acc_clear <= 1'b0;

                    if (loading) begin
                        if (load_k == K_reg - 1'b1)
                            loading <= 1'b0;
                        else
                            load_k <= load_k + 1'b1;
                    end

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
                    c_buf[(m_base + drain_m) * N_reg + (n_base + drain_n)] <=
                        pe_acc[drain_m[3:0]*PN + drain_n[4:0]] +
                        (use_bias_reg ? bias_buf[m_base + drain_m] : 32'sd0);

                    if (drain_n == N_tile - 1'b1) begin
                        drain_n <= {DIM_W{1'b0}};
                        if (drain_m == M_tile - 1'b1)
                            state <= S_NEXT;
                        else
                            drain_m <= drain_m + 1'b1;
                    end else begin
                        drain_n <= drain_n + 1'b1;
                    end
                end

                S_NEXT: begin
                    if (nt + 1'b1 < nt_count) begin
                        nt           <= nt + 1'b1;
                        n_base       <= n_base + PN[DIM_W-1:0];
                        N_tile       <= ((N_reg - n_base - PN[DIM_W-1:0]) >= PN[DIM_W-1:0])
                                        ? PN[DIM_W-1:0]
                                        : (N_reg - n_base - PN[DIM_W-1:0]);
                        active_bank  <= ~active_bank;
                        cycle_cnt    <= {DIM_W{1'b0}};
                        pe_en        <= 1'b1;
                        pe_acc_clear <= 1'b1;
                        if (nt + 16'd2 < nt_count) begin
                            loading     <= 1'b1;
                            load_k      <= {DIM_W{1'b0}};
                            load_n_base <= n_base + PN[DIM_W-1:0] + PN[DIM_W-1:0];
                        end else begin
                            loading <= 1'b0;
                        end
                        state <= S_COMPUTE;
                    end else if (mt + 1'b1 < mt_count) begin
                        mt           <= mt + 1'b1;
                        nt           <= {DIM_W{1'b0}};
                        m_base       <= m_base + PM[DIM_W-1:0];
                        n_base       <= {DIM_W{1'b0}};
                        M_tile       <= ((M_reg - m_base - PM[DIM_W-1:0]) >= PM[DIM_W-1:0])
                                        ? PM[DIM_W-1:0]
                                        : (M_reg - m_base - PM[DIM_W-1:0]);
                        N_tile       <= (N_reg >= PN[DIM_W-1:0]) ? PN[DIM_W-1:0] : N_reg;
                        load_k       <= {DIM_W{1'b0}};
                        loading      <= 1'b0;
                        state        <= S_LOAD_B0;
                    end else begin
                        state <= S_DONE;
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
