#include "Halide.h"

namespace {

using namespace Halide;

class UnitTestBitwise : public Halide::Generator<UnitTestBitwise> {
public:
    Input<Buffer<uint8_t>>  input{"input", 2};
    Output<Buffer<uint8_t>> output{"output", 2};

    void generate() {
        /* THE ALGORITHM */

        Var x("x"), y("y");

        Func hw_input("hw_input");
        Func hw_input_copy("hw_input_copy");
        hw_input(x, y) = cast<uint16_t>(input(x, y));
        hw_input_copy(x, y) = hw_input(x, y);

        Func bitwise_and, bitwise_or, bitwise_inv, bitwise_xor;
        bitwise_and(x,y) = hw_input(x,y) & 235;
        bitwise_inv(x,y) = ~( bitwise_and(x,y) );
        bitwise_or(x,y)  = hw_input(x,y) | 63;
        bitwise_xor(x,y) = bitwise_inv(x,y) ^ bitwise_or(x,y);

        Func hw_output("hw_output");
        hw_output(x, y) = cast<uint8_t>(bitwise_xor(x, y));
        output(x, y) = hw_output(x,y);

        /* THE SCHEDULE */
        if (get_target().has_feature(Target::CoreIR)) {
          Var xi,yi, xo,yo;
          
          hw_input.compute_root();
          hw_output.compute_root();

          output.bound(x, 0, 64);
          output.bound(y, 0, 64);
          
          hw_output.tile(x,y, xo,yo, xi,yi, 64, 64)
            .hw_accelerate(xi, xo);

          hw_input.compute_at(hw_output, xi).store_at(hw_output, xo);
          hw_input.stream_to_accelerator();
          
        } else if (get_target().has_feature(Target::Clockwork)) {
          Var xi,yi, xo,yo;

          output.bound(x, 0, 64);
          output.bound(y, 0, 64);
          
          hw_output.compute_root();

          hw_output.tile(x,y, xo,yo, xi,yi, 64, 64)
            .hw_accelerate(xi, xo);

          hw_input_copy.compute_at(hw_output, xo);
          hw_input.compute_root();
          hw_input.stream_to_accelerator();


        } else {  // schedule to CPU
          output.compute_root();
        }
        
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(UnitTestBitwise, bitwise)
