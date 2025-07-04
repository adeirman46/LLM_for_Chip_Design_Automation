`timescale 1ns/100ps

module mac_module_tb;

parameter WIDTH = 32;

reg [WIDTH-1:0] a;
reg clk;
reg [WIDTH-1:0] x;
reg [WIDTH-1:0] b;
wire [2*WIDTH+1-1:0] y;

mac_module #(
    .width(WIDTH)
) uut (
    .a(a),
    .clk(clk),
    .x(x),
    .b(b),
    .y(y)
);

initial begin
    $dumpfile("mac_module.vcd");
    $dumpvars(0, mac_module_tb);
end

initial begin
    clk = 0;
    forever #5 clk = ~clk;
end

initial begin
    a = 0;
    x = 0;
    b = 0;
    #10;
    a = 1;
    x = 2;
    b = 3;
    #10;
    $finish;
end

endmodule