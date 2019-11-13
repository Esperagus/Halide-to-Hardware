#include "Halide.h"

namespace {

using namespace Halide;

class MaxPoolKernel : public Halide::Generator<MaxPoolKernel> {
public:
    Input<Buffer<uint8_t>>  input{"input", 2};
    Output<Buffer<uint8_t>> output{"output", 2};

    void generate() {
        /* THE ALGORITHM */
        int stride = 1;

        Var x("x"), y("y");

        Func max_pool("max_pool");
        RDom r(0, stride,
               0, stride);

        Func hw_input("hw_input");
        hw_input(x, y) = cast<uint16_t>(input(x, y));

        max_pool(x, y) = maximum(hw_input(x * stride + r.x, y * stride + r.y));

        Func hw_output("hw_output");
        hw_output(x, y) = cast<uint8_t>(max_pool(x, y));
        output(x ,y) = hw_output(x, y);

        /* THE SCHEDULE */
        if (get_target().has_feature(Target::CoreIR)) {
            Var xi, yi, xo, yo;
            hw_input.compute_root();
            hw_output.compute_root();

            hw_output.tile(x, y, xo, yo, xi, yi, 32, 32)
                .hw_accelerate(xi, xo);

            max_pool.unroll(x, stride)
                    .unroll(y, stride);

            max_pool.linebuffer();

            hw_input.stream_to_accelerator();
        } else { // schedule to CPU
            max_pool.compute_root();
            max_pool.unroll(x, stride)
                    .unroll(y, stride);
        }
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(MaxPoolKernel, down_sample)