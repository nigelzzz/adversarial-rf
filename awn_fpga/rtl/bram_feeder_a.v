// bram_feeder_a: dual-port BRAM for A-matrix row storage.
// Clocked write, combinational read. One row-bank per PE row.
module bram_feeder_a #(
    parameter ROWS  = 8,
    parameter DEPTH = 512,
    parameter AW    = 9
)(
    input               clk,
    // Write port
    input               wr_en,
    input  [2:0]        wr_row,
    input  [AW-1:0]     wr_col,
    input  signed [7:0] wr_data,
    // Read port (combinational)
    input  [2:0]        rd_row,
    input  [AW-1:0]     rd_col,
    output signed [7:0] rd_data
);

    reg signed [7:0] mem [0:ROWS-1][0:DEPTH-1];

    always @(posedge clk)
        if (wr_en) mem[wr_row][wr_col] <= wr_data;

    assign rd_data = mem[rd_row][rd_col];

endmodule
