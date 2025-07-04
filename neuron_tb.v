`timescale 1ns/100ps

module neuron_tb;

parameter WIDTH = 16;

reg clk;
reg [WIDTH-1:0] x1;
reg [WIDTH-1:0] x2;

wire [2*WIDTH+1-1:0] y;

neuron #(
    .width(WIDTH)
) DUT (
    .y(y),
    .x1(x1),
    .x2(x2),
    .clk(clk)
);

initial begin
    clk = 0;
    x1 = 0;
    x2 = 0;
    #10 x1 = 1;
    #10 x2 = 1;
    #10 $finish;
end

always #5 clk = ~clk;

initial begin
    $dumpfile("neuron.vcd");
    $dumpvars(0, neuron_tb);
end

endmodule