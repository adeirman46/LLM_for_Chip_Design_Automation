`timescale 1ns/100ps

module adder_tb;

    localparam width = 32;

    reg [width-1:0] A;
    reg [width-1:0] B;
    reg  clk;
    wire [width+1-1:0] Y;

    adder #(
        .width(width)
    ) uut (
        .A(A),
        .B(B),
        .clk(clk),
        .Y(Y)
    );

    initial begin
        $dumpfile("adder.vcd");
        $dumpvars(0, adder_tb);
    end

    initial begin
        clk = 0;
        forever begin
            #5 clk = ~clk;
        end
    end

    initial begin
        // Apply test vectors
        A = 0; B = 0; #10;
        A = 1; B = 1; #10;
        A = 2; B = 3; #10;
        A = 4; B = 5; #10;
        $finish;
    end

endmodule