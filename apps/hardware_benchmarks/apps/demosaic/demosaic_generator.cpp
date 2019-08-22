/*
 * An application for taking a raw image and demosaicking the image
 * to an RGB image.
 */

#include "Halide.h"

namespace {

using namespace Halide;

class Demosaic : public Halide::Generator<Demosaic> {
public:
    Input<Buffer<uint8_t>>  input{"input", 2};
    Output<Buffer<uint8_t>> output{"output", 3};

    // assumption that uint8_t phase = 0;

    void generate() {
        /* THE ALGORITHM */
        Var c("c"), x("x"), y("y");
        Var xo("xo"), yo("yo"), xi("xi"), yi("yi");

        Func hw_input;
        hw_input(x,y) = cast<uint16_t>(input(x+1,y+1));

        // common patterns: average of four surrounding pixels
        Func neswNeighbors, diagNeighbors;
        neswNeighbors(x, y) = (hw_input(x-1, y)   + hw_input(x+1, y) +
                               hw_input(x,   y-1) + hw_input(x,   y+1)/4);
        diagNeighbors(x, y) = (hw_input(x-1, y-1) + hw_input(x+1, y-1) +
                               hw_input(x-1, y+1) + hw_input(x+1, y+1)/4);

        // common patterns: average of two adjacent pixels
        Func vNeighbors, hNeighbors;
        vNeighbors(x, y) = (hw_input(x, y-1) + hw_input(x, y+1))/2;
        hNeighbors(x, y) = (hw_input(x-1, y) + hw_input(x+1, y))/2;

        // output pixels depending on image layout.
        // Generally, image looks like
        //    R G R G R G R G
        //    G B G B G B G B
        //    R G R G R G R G
        //    G B G B G B G B
        Func green, red, blue;
        green(x, y) = select((y % 2) == (0),
                             select((x % 2) == (0), neswNeighbors(x, y), hw_input(x, y)), // First row, RG
                             select((x % 2) == (0), hw_input(x, y),      neswNeighbors(x, y))); // Second row, GB

        red(x, y) = select((y % 2) == (0),
                           select((x % 2) == (0), hw_input(x, y),   hNeighbors(x, y)), // First row, RG
                           select((x % 2) == (0), vNeighbors(x, y), diagNeighbors(x, y))); // Second row, GB

        blue(x, y) = select((y % 2) == (0),
                            select((x % 2) == (0), diagNeighbors(x, y), vNeighbors(x, y)), // First row, RG
                            select((x % 2) == (0), hNeighbors(x, y),    hw_input(x, y))); // Second row, GB

        // output all channels
        Func demosaic, hw_output;
        demosaic(x,y,c) = cast<uint8_t>(select(c == 0, red(x, y),
                                               c == 1, green(x, y),
                                               blue(x, y)));

        hw_output(x,y,c) = demosaic(x,y,c);
        output(x,y,c) = hw_output(x,y,c);

        output.bound(c, 0, 3);
        output.bound(x, 0, 62);
        output.bound(y, 0, 62);
            
        /* THE SCHEDULE */
        if (get_target().has_feature(Target::CoreIR)) {
          hw_input.compute_root();
          hw_output.compute_root();
          
          hw_output.tile(x, y, xo, yo, xi, yi, 62,62)
            .reorder(c, xi, yi, xo, yo);

          hw_input.stream_to_accelerator();
          hw_output.hw_accelerate(xi, xo);
          hw_output.unroll(c);


        } else {    // schedule to CPU
          output.compute_root();
        }
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(Demosaic, demosaic)

