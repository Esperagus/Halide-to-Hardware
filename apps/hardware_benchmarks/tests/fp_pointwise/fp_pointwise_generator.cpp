#include "Halide.h"

namespace {

using namespace Halide;

class PointwiseKernel : public Halide::Generator<PointwiseKernel> {
public:
    Input<Buffer<uint8_t>>  input{"input", 2};
    Output<Buffer<uint8_t>> output{"output", 2};

    void generate() {
        /* THE ALGORITHM */

        Var x("x"), y("y");

        Func product("product");

        Func hw_input("hw_input");
        hw_input(x, y) = cast<bfloat16_t>(input(x, y));
        Expr pi = bfloat16_t(3.1415926535f);
        product(x, y)  = hw_input(x, y) * pi;
        //Expr const_val = bfloat16_t(7.f);
        //product(x, y)  = select(hw_input(x, y) != const_val, 255, 0);

        Func hw_output("hw_output");
        hw_output(x, y) = product(x, y);
        output(x, y) = cast<uint8_t>(hw_output(x,y));

        /* THE SCHEDULE */
        if (get_target().has_feature(Target::CoreIR)) {
          Var xi,yi, xo,yo;
          
          hw_input.compute_root();
          hw_output.compute_root();
          
          hw_output.tile(x,y, xo,yo, xi,yi, 64, 64)
            .hw_accelerate(xi, xo);

          hw_input.stream_to_accelerator();
          
        } else {  // schedule to CPU
          output.compute_root();
        }
        
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(PointwiseKernel, fp_pointwise)