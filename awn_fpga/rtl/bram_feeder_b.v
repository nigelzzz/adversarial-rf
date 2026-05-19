// bram_feeder_b: dual-port BRAM for B-matrix tile storage.
// Wide write port (all COLS simultaneously), per-column parallel read ports.
module bram_feeder_b #(
    parameter COLS  = 16,
    parameter DEPTH = 512,
    parameter AW    = 9
)(
    input                    clk,
    // Wide write port: all COLS values at row wr_k
    input                    wr_en,
    input  [AW-1:0]          wr_k,
    input  [COLS*8-1:0]      wr_data_flat,
    // Per-column parallel read: each column has its own row address
    input  [COLS*AW-1:0]     rd_rows_flat,
    output [COLS*8-1:0]      rd_datas_flat
);

    reg signed [7:0] mem [0:COLS-1][0:DEPTH-1];

    genvar wi;
    generate
        for (wi = 0; wi < COLS; wi = wi + 1) begin : wr
            always @(posedge clk)
                if (wr_en) mem[wi][wr_k] <= wr_data_flat[wi*8 +: 8];
        end
    endgenerate

    genvar ri;
    generate
        for (ri = 0; ri < COLS; ri = ri + 1) begin : rd
            wire [AW-1:0] row_addr = rd_rows_flat[ri*AW +: AW];
            assign rd_datas_flat[ri*8 +: 8] = mem[ri][row_addr];
        end
    endgenerate

endmodule
