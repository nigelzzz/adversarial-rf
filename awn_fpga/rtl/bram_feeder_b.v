// bram_feeder_b: dual-port BRAM for B-matrix column storage.
// Clocked write, combinational read. One col-bank per PE column.
module bram_feeder_b #(
    parameter COLS  = 16,
    parameter DEPTH = 512,
    parameter AW    = 9
)(
    input               clk,
    // Write port
    input               wr_en,
    input  [3:0]        wr_col,
    input  [AW-1:0]     wr_row,
    input  signed [7:0] wr_data,
    // Read port (combinational)
    input  [3:0]        rd_col,
    input  [AW-1:0]     rd_row,
    output signed [7:0] rd_data
);

    reg signed [7:0] mem [0:COLS-1][0:DEPTH-1];

    always @(posedge clk)
        if (wr_en) mem[wr_col][wr_row] <= wr_data;

    assign rd_data = mem[rd_col][rd_row];

endmodule
