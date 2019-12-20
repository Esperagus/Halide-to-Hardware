#include "Halide.h"

namespace {

using namespace Halide;

class UnitTestArith : public Halide::Generator<UnitTestArith> {
public:
    Input<Buffer<uint8_t>>  input{"input", 2};
    Output<Buffer<uint8_t>> output{"output", 2};

    int tilesize = 30;

    void generate() {
        /* THE ALGORITHM */

        Var x("x"), y("y");

        Func hw_input("hw_input");
        hw_input(x, y) = cast<int16_t>(input(x, y));

        Func mult, div, add, sub, mod;
        mult(x,y) = hw_input(x,y) * 13;
        div(x,y) = hw_input(x,y) / 4;
        mod(x,y) = hw_input(x,y) % 16;
        add(x,y) = div(x,y) + mod(x,y);
        sub(x,y) = mult(x,y) - add(x,y);

        Func hw_output("hw_output");
        hw_output(x, y) = cast<uint8_t>(sub(x, y));
        output(x, y) = hw_output(x,y);

        /* THE SCHEDULE */
        if (get_target().has_feature(Target::CoreIR)) {
          Var xi,yi, xo,yo;
          
          hw_input.compute_root();
          hw_output.compute_root();

          output.bound(x, 0, 64);
          output.bound(y, 0, 64);
          
          // hw_output.tile(x,y, xo,yo, xi,yi, tilesize, tilesize)
          //   .hw_accelerate(xi, xo);
          hw_output.hw_accelerate(x);

          hw_input.stream_to_accelerator();
          
        } else {  // schedule to CPU
          output.compute_root();
        }
        
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(UnitTestArith, arith)
