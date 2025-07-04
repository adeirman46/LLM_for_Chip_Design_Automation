module adder #(
    parameter width = 32
)
(
    input  [width-1:0] A,
    input  [width-1:0] B,
    input   clk,
    output reg [width+1-1:0] Y
);

always @(posedge clk) begin
    Y <= A + B;
end

endmodule