// avgpool1d_s8: per-channel reduction over `pool_len` consecutive int8 elements
// of an int8 input arranged as `channels` channels of `pool_len` samples each
// (NCT layout, N=1). Output is `channels` int8 values.
//   out[c] = sat_int8( ((sum[c] * mul) + half) >>> shift )
// where sum is signed accumulation over pool_len samples.
//
// in_buf layout:  in_buf[c * pool_len + t]
// out_buf layout: out_buf[c]

module avgpool1d_s8 #(
    parameter LEN = 16384,
    parameter ADDR_W = 16,
    parameter CHAN_W = 12,
    parameter POOL_W = 12
)(
    input                   clk,
    input                   rst_n,
    input                   start,
    input  [CHAN_W-1:0]     channels,
    input  [POOL_W-1:0]     pool_len,
    input  signed [31:0]    mul_q,
    input         [5:0]     shift,
    output reg              done
);

reg signed [7:0] in_buf  [0:LEN-1];
reg signed [7:0] out_buf [0:LEN-1];

reg [CHAN_W-1:0] c;
reg [POOL_W:0]   t;
reg signed [31:0] acc;
reg busy;

reg [ADDR_W-1:0] addr;
always @* addr = c * pool_len + t[POOL_W-1:0];

wire signed [63:0] prod  = $signed({{32{acc[31]}}, acc}) * mul_q;
wire signed [63:0] half  = (shift == 0) ? 64'sd0
                                        : ($signed(64'd1) <<< (shift - 1));
wire signed [63:0] sh    = (prod + half) >>> shift;
wire signed [7:0]  sat   = (sh >  127) ? 8'sd127
                        : (sh < -128) ? -8'sd128
                                      : sh[7:0];

integer k;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        busy <= 0; done <= 0; c <= 0; t <= 0; acc <= 0;
        for (k = 0; k < LEN; k = k + 1) out_buf[k] <= 0;
    end else begin
        done <= 0;
        if (start && !busy) begin
            busy <= 1; c <= 0; t <= 0; acc <= 0;
        end else if (busy) begin
            if (t < {1'b0, pool_len}) begin
                acc <= acc + in_buf[addr];
                t   <= t + 1;
            end else begin
                // commit channel result, advance c
                out_buf[c] <= sat;
                acc <= 0;
                t   <= 0;
                if (c == channels - 1) begin
                    busy <= 0; done <= 1;
                end else begin
                    c <= c + 1;
                end
            end
        end
    end
end

endmodule
