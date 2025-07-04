module mac_module #(
    parameter width = 16
)
(
    input  [width-1:0] a,
    input   clk,
    input  [width-1:0] x,
    input  [width-1:0] b,
    output reg [2*width+1-1:0] y
);

always @(posedge clk) begin
    y <= a * x + b;
end

endmodule