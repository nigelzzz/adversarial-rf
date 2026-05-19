// im2col_addr_gen: parameterized address generator for im2col transformation.
// Supports 1D and 2D convolutions with no-padding, zero-padding, and
// reflection-padding modes. Outputs one (addr, zero_flag) per cycle.
//
// For 2D, output columns iterate h_out (slow) then w_out (fast), matching
// numpy's im2col_2d loop order. Padding applies to W dimension only.
module im2col_addr_gen #(
    parameter AW    = 14,
    parameter DIM_W = 16
)(
    input                clk,
    input                rst_n,
    input                start,
    input  [DIM_W-1:0]   cfg_cin,
    input  [DIM_W-1:0]   cfg_hin,
    input  [DIM_W-1:0]   cfg_win,
    input  [DIM_W-1:0]   cfg_kh,
    input  [DIM_W-1:0]   cfg_kw,
    input  [DIM_W-1:0]   cfg_pad_left,
    input  [1:0]          cfg_pad_mode, // 00=none, 01=zero, 10=reflect
    input  [DIM_W-1:0]   cfg_nout,     // total output positions (H_out * W_out)
    output reg [AW-1:0]   addr,
    output reg            zero_flag,
    output reg            valid,
    output reg            done
);

    localparam S_IDLE = 2'd0;
    localparam S_RUN  = 2'd1;
    localparam S_DONE = 2'd2;
    reg [1:0] state;

    reg [DIM_W-1:0] C_in, W_in, kH, kW, pad_left;
    reg [1:0]       pad_mode;
    reg [DIM_W-1:0] stride_cin;
    reg [DIM_W-1:0] two_win_m2;
    reg [DIM_W-1:0] H_out, W_out;

    reg [DIM_W-1:0] h_out, w_out, cin, kh_cnt, kw_cnt;
    reg [31:0]      cin_base, kh_base, h_base;

    // W-dimension position (signed for negative padding offsets)
    wire signed [DIM_W:0] pos_w =
        $signed({1'b0, w_out}) + $signed({1'b0, kw_cnt}) - $signed({1'b0, pad_left});
    wire signed [DIM_W:0] win_s = $signed({1'b0, W_in});
    wire                  pos_neg = pos_w[DIM_W];
    wire                  pos_ge_win = (pos_w >= win_s);
    wire [DIM_W-1:0]      neg_pos_w = (~pos_w[DIM_W-1:0]) + 16'd1;

    reg [DIM_W-1:0] pos_final;
    reg             is_zero;

    always @(*) begin
        pos_final = pos_w[DIM_W-1:0];
        is_zero   = 1'b0;
        case (pad_mode)
            2'b00: pos_final = pos_w[DIM_W-1:0];
            2'b01: begin
                if (pos_neg || pos_ge_win)
                    is_zero = 1'b1;
                else
                    pos_final = pos_w[DIM_W-1:0];
            end
            2'b10: begin
                if (pos_neg)
                    pos_final = neg_pos_w;
                else if (pos_ge_win)
                    pos_final = two_win_m2 - pos_w[DIM_W-1:0];
                else
                    pos_final = pos_w[DIM_W-1:0];
            end
            default: pos_final = pos_w[DIM_W-1:0];
        endcase
    end

    // BRAM address: cin*stride + (h_out+kh)*W_in + pos_final
    wire [31:0] bram_addr = cin_base + h_base + kh_base + {16'd0, pos_final};

    wire last_kw   = (kw_cnt == kW - 16'd1);
    wire last_kh   = (kh_cnt == kH - 16'd1);
    wire last_cin  = (cin    == C_in - 16'd1);
    wire last_wout = (w_out  == W_out - 16'd1);
    wire last_hout = (h_out  == H_out - 16'd1);
    wire last_elem = last_kw & last_kh & last_cin & last_wout & last_hout;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            valid <= 1'b0;
            done  <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    valid <= 1'b0;
                    done  <= 1'b0;
                    if (start) begin
                        C_in       <= cfg_cin;
                        W_in       <= cfg_win;
                        kH         <= cfg_kh;
                        kW         <= cfg_kw;
                        pad_left   <= cfg_pad_left;
                        pad_mode   <= cfg_pad_mode;
                        stride_cin <= cfg_hin * cfg_win;
                        two_win_m2 <= {cfg_win[DIM_W-2:0], 1'b0} - 16'd2;
                        H_out      <= cfg_hin - cfg_kh + 16'd1;
                        W_out      <= (cfg_pad_mode != 2'b00)
                                        ? cfg_win + {cfg_pad_left[DIM_W-2:0], 1'b0}
                                          - cfg_kw + 16'd1
                                        : cfg_win - cfg_kw + 16'd1;
                        h_out      <= 16'd0;
                        w_out      <= 16'd0;
                        cin        <= 16'd0;
                        kh_cnt     <= 16'd0;
                        kw_cnt     <= 16'd0;
                        cin_base   <= 32'd0;
                        kh_base    <= 32'd0;
                        h_base     <= 32'd0;
                        state      <= S_RUN;
                    end
                end

                S_RUN: begin
                    valid     <= 1'b1;
                    addr      <= bram_addr[AW-1:0];
                    zero_flag <= is_zero;
                    if (last_elem) begin
                        state <= S_DONE;
                    end else if (!last_kw) begin
                        kw_cnt <= kw_cnt + 16'd1;
                    end else begin
                        kw_cnt <= 16'd0;
                        if (!last_kh) begin
                            kh_cnt  <= kh_cnt + 16'd1;
                            kh_base <= kh_base + {16'd0, W_in};
                        end else begin
                            kh_cnt  <= 16'd0;
                            kh_base <= 32'd0;
                            if (!last_cin) begin
                                cin      <= cin + 16'd1;
                                cin_base <= cin_base + {16'd0, stride_cin};
                            end else begin
                                cin      <= 16'd0;
                                cin_base <= 32'd0;
                                if (!last_wout) begin
                                    w_out <= w_out + 16'd1;
                                end else begin
                                    w_out  <= 16'd0;
                                    h_out  <= h_out + 16'd1;
                                    h_base <= h_base + {16'd0, W_in};
                                end
                            end
                        end
                    end
                end

                S_DONE: begin
                    valid <= 1'b0;
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
