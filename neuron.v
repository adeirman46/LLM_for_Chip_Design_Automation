`include "mac_module.v"

module neuron #(
    parameter width = 16
)
(
    output reg [2*width+1-1:0] y,
    input  [width-1:0] x1,
    input  [width-1:0] x2,
    input   clk
);

wire [2*width+1-1:0] y1;
wire [2*width+-1:0] y2;

mac_module #(
    .width(width)
) mac1 (
    .a(16'b1), 
    .x(x1), 
    .b(16'b1), 
    .clk(clk), 
    .y(y1) 
);

mac_module #(
    .width(width)
) mac2 (
    .a(16'b1), 
    .x(x2), 
    .b(16'b1), 
    .clk(clk), 
    .y(y2) 
);

always @(posedge clk) begin
    y <= y1 + y2;
end

endmodule