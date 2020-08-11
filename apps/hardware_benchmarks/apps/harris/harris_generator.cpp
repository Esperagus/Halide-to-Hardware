/*
 * An application for detecting corners in images. It computes
 * gradients in the x and y direction, and then uses Harris's
 * method to calculate cornerness efficiently.
 */

#include "Halide.h"

namespace {

using namespace Halide;

// Size of blur for gradients.
int blockSize = 3;

// k is a sensitivity parameter for detecting corners.
// k should vary from 0.04 to 0.15 according to literature.
int shiftk = 4; // equiv to k = 0.0625

// Threshold for cornerness measure.
int threshold = 1;

class HarrisCornerDetector : public Halide::Generator<HarrisCornerDetector> {
public:
    Input<Buffer<uint8_t>>  input{"input", 2};
    Output<Buffer<uint8_t>> output{"output", 2};

    GeneratorParam<uint8_t> schedule{"schedule", 0};    // default: 0

    void generate() {
        /* THE ALGORITHM */

        Var x("x"), y("y");
        Var xo("xo"), yo("yo"), xi("xi"), yi("yi");

        Func padded16, padded, hw_input_copy;
        padded(x, y) = input(x+3, y+3);
        padded16(x, y) = cast<int16_t>(padded(x, y));
        //hw_input_copy(x, y) = cast<int16_t>(input(x+3,y+3));
        //padded16(x, y) = hw_input_copy(x, y);

        // sobel filter
        Func grad_x_unclamp, grad_y_unclamp, grad_x, grad_y;
        grad_x_unclamp(x, y) = cast<int16_t>(  -padded16(x-1,y-1) +   padded16(x+1,y-1)
                                             -2*padded16(x-1,y)   + 2*padded16(x+1,y)
                                               -padded16(x-1,y+1) +   padded16(x+1,y+1));
        grad_y_unclamp(x, y) = cast<int16_t>(   padded16(x-1,y+1) -   padded16(x-1,y-1) +
                                              2*padded16(x,  y+1) - 2*padded16(x,  y-1) +
                                                padded16(x+1,y+1) -   padded16(x+1,y-1));
        //RDom r(-1, 3, -1, 3);
        //grad_x_unclamp(x, y) = 0;
        //grad_x_unclamp(x, y) += cast<int16_t>(padded16(x+r.x, y+r.y));
        //grad_y_unclamp(x, y) = 0;
        //grad_y_unclamp(x, y) += cast<int16_t>(padded16(x+r.x, y+r.y));

        grad_x(x, y) = clamp(grad_x_unclamp(x,y), -255, 255);
        grad_y(x, y) = clamp(grad_y_unclamp(x,y), -255, 255);
        
        // product of gradients
        Func grad_xx, grad_yy, grad_xy;
        grad_xx(x, y) = cast<int16_t>(grad_x(x,y)) * cast<int16_t>(grad_x(x,y));
        grad_yy(x, y) = cast<int16_t>(grad_y(x,y)) * cast<int16_t>(grad_y(x,y));
        grad_xy(x, y) = cast<int16_t>(grad_x(x,y)) * cast<int16_t>(grad_y(x,y));

        // shift gradients
        Func lxx, lyy, lxy;
        //lxx(x, y) = grad_xx(x, y) >> 7;
        //lyy(x, y) = grad_yy(x, y) >> 7;
        //lxy(x, y) = grad_xy(x, y) >> 7;
        lxx(x, y) = cast<int16_t>(grad_x(x,y)) * cast<int16_t>(grad_x(x,y)) >> 7;
        lyy(x, y) = cast<int16_t>(grad_y(x,y)) * cast<int16_t>(grad_y(x,y)) >> 7;
        lxy(x, y) = cast<int16_t>(grad_x(x,y)) * cast<int16_t>(grad_y(x,y)) >> 7;


        // box filter (i.e. windowed sum)
        Func lgxx, lgyy, lgxy;
        RDom box(-blockSize/2, blockSize, -blockSize/2, blockSize);
        //RDom box(0, blockSize, 0, blockSize);
        lgxx(x, y) += lxx(x+box.x, y+box.y);
        lgyy(x, y) += lyy(x+box.x, y+box.y);
        lgxy(x, y) += lxy(x+box.x, y+box.y);

        Expr lgxx8 = lgxx(x,y) >> 6;
        Expr lgyy8 = lgyy(x,y) >> 6;
        Expr lgxy8 = lgxy(x,y) >> 6;

        // calculate Cim
        //        int scale = (1 << (Ksize-1)) * blockSize;
        //        Expr lgx = cast<float>(grad_gx(x, y) / scale / scale);
        //        Expr lgy = cast<float>(grad_gy(x, y) / scale / scale);
        //        Expr lgxy = cast<float>(grad_gxy(x, y) / scale / scale);

        // scale==12, so dividing by 144
        // approx~ 1>>7==divide by 128
        Func cim;
        Expr det = lgxx8*lgyy8 - lgxy8*lgxy8;
        Expr trace = lgxx8 + lgyy8;
        //cim(x, y) = det - (trace*trace >> shiftk);
        cim(x, y) = det - ((lgxx8+lgyy8)*(lgxx8+lgyy8) >> shiftk);

        // Perform non-maximal suppression
        Func hw_output;
        Expr is_max = cim(x, y) > cim(x-1, y-1) && cim(x, y) > cim(x, y-1) &&
            cim(x, y) > cim(x+1, y-1) && cim(x, y) > cim(x-1, y) &&
            cim(x, y) > cim(x+1, y) && cim(x, y) > cim(x-1, y+1) &&
            cim(x, y) > cim(x, y+1) && cim(x, y) > cim(x+1, y+1);
        Func cim_output;
        cim_output(x,y) = cast<int16_t>(select( is_max && (cim(x, y) >= threshold), 255, 0));
        hw_output(x, y) = cim_output(x,y);
        //hw_output(x, y) = cast<uint8_t>(cim(x,y));
        //hw_output(x, y) = cast<uint8_t>(lgxx(x,y));

        output(x, y) = cast<uint8_t>(hw_output(x, y));


        /* THE SCHEDULE */
        if (get_target().has_feature(Target::CoreIR) || get_target().has_feature(Target::HLS)) {

          grad_x.bound(x, -2, 62);
          grad_x.bound(y, -2, 62);
          grad_y.bound(x, -2, 62);
          grad_y.bound(y, -2, 62);

          //grad_xx.bound(x, -2, 62);
          //grad_xx.bound(y, -2, 62);
          //grad_xy.bound(x, -2, 62);
          //grad_xy.bound(y, -2, 62);
          //grad_yy.bound(x, -2, 62);
          //grad_yy.bound(y, -2, 62);

          lxx.bound(x, -2, 62);
          lxx.bound(y, -2, 62);
          lxy.bound(x, -2, 62);
          lxy.bound(y, -2, 62);
          lyy.bound(x, -2, 62);
          lyy.bound(y, -2, 62);

          lgxx.bound(x, -1, 60);
          lgxx.bound(y, -1, 60);
          lgxy.bound(x, -1, 60);
          lgxy.bound(y, -1, 60);
          lgyy.bound(x, -1, 60);
          lgyy.bound(y, -1, 60);

          hw_output.bound(x, 0, 58);
          hw_output.bound(y, 0, 58);
          output.bound(x, 0, 58);
          output.bound(y, 0, 58);
          cim_output.bound(x, 0, 58);
          cim_output.bound(y, 0, 58);

          //output.tile(x, y, xo, yo, xi, yi, 64, 64);
          //padded16.compute_root();
          //hw_output.compute_at(output, xo);
          hw_output.compute_root();

          //int tileSize = 8;
          int tileSize = 58;
          hw_output
            .tile(x, y, xo, yo, xi, yi, tileSize, tileSize)
            .accelerate({padded16}, xi, xo);
            //.hw_accelerate(xi, xo);
          //padded16.stream_to_accelerator();

          grad_x.linebuffer();
          grad_y.linebuffer();
          lxx.linebuffer();
          lyy.linebuffer();
          lxy.linebuffer();
          lgxx.linebuffer();
          lgyy.linebuffer();
          lgxy.linebuffer();
          //cim.linebuffer();
          cim_output.linebuffer();

          lgxx.update().unroll(box.x).unroll(box.y);
          lgyy.update().unroll(box.x).unroll(box.y);
          lgxy.update().unroll(box.x).unroll(box.y);

          padded16.store_at(hw_output, xo).compute_at(hw_output, xi);
          padded16.stream_to_accelerator();

        } else if (get_target().has_feature(Target::Clockwork)) {
          output.bound(x, 0, 58);
          output.bound(y, 0, 58);

          hw_output.compute_root();
          int tileSize = 58;

          if (schedule == 1) { // few buffers
            hw_output
              .tile(x, y, xo, yo, xi, yi, tileSize, tileSize)
              .hw_accelerate(xi, xo);

            lgxx.compute_at(hw_output, xo);
            lgyy.compute_at(hw_output, xo);
            lgxy.compute_at(hw_output, xo);
            
            padded16.stream_to_accelerator();
            
          } else if (schedule == 2) { // more buffers
            hw_output
              .tile(x, y, xo, yo, xi, yi, tileSize, tileSize)
              .hw_accelerate(xi, xo);

            grad_x.compute_at(hw_output, xo);
            grad_y.compute_at(hw_output, xo);
            lxx.compute_at(hw_output, xo);
            lyy.compute_at(hw_output, xo);
            lxy.compute_at(hw_output, xo);
            lgxx.compute_at(hw_output, xo);
            lgyy.compute_at(hw_output, xo);
            lgxy.compute_at(hw_output, xo);
            cim.compute_at(hw_output, xo);
            cim_output.compute_at(hw_output, xo);

            padded16.stream_to_accelerator();

          } else if (schedule == 3) { // unroll some
            hw_output
              .tile(x, y, xo, yo, xi, yi, tileSize, tileSize)
              .hw_accelerate(xi, xo);

            grad_x.compute_at(hw_output, xo);
            grad_y.compute_at(hw_output, xo);
            lxx.compute_at(hw_output, xo);
            lyy.compute_at(hw_output, xo);
            lxy.compute_at(hw_output, xo);
            lgxx.compute_at(hw_output, xo);
            lgyy.compute_at(hw_output, xo);
            lgxy.compute_at(hw_output, xo);
            cim.compute_at(hw_output, xo);
            cim_output.compute_at(hw_output, xo);

            lgxx.update().unroll(box.x);
            lgyy.update().unroll(box.x);
            lgxy.update().unroll(box.x);
            
            padded16.stream_to_accelerator();

          } else if (schedule == 4) { // unroll all
            hw_output
              .tile(x, y, xo, yo, xi, yi, tileSize, tileSize)
              .hw_accelerate(xi, xo);

            grad_x.compute_at(hw_output, xo);
            grad_y.compute_at(hw_output, xo);
            lxx.compute_at(hw_output, xo);
            lyy.compute_at(hw_output, xo);
            lxy.compute_at(hw_output, xo);
            lgxx.compute_at(hw_output, xo);
            lgyy.compute_at(hw_output, xo);
            lgxy.compute_at(hw_output, xo);
            cim.compute_at(hw_output, xo);
            cim_output.compute_at(hw_output, xo);

            lgxx.update().unroll(box.x).unroll(box.y);
            lgyy.update().unroll(box.x).unroll(box.y);
            lgxy.update().unroll(box.x).unroll(box.y);
            
            padded16.stream_to_accelerator();

          } else if (schedule == 5) { // end at cim
            //cim.compute_root();
            hw_output
              .tile(x, y, xo, yo, xi, yi, tileSize, tileSize)
              .hw_accelerate(xi, xo);

            grad_x.compute_at(hw_output, xo);
            grad_y.compute_at(hw_output, xo);
            lxx.compute_at(hw_output, xo);
            lyy.compute_at(hw_output, xo);
            lxy.compute_at(hw_output, xo);
            lgxx.compute_at(hw_output, xo);
            lgyy.compute_at(hw_output, xo);
            lgxy.compute_at(hw_output, xo);

            lgxx.update().unroll(box.x).unroll(box.y);
            lgyy.update().unroll(box.x).unroll(box.y);
            lgxy.update().unroll(box.x).unroll(box.y);
            
            padded16.stream_to_accelerator();
            
          } else {
            hw_output
              .tile(x, y, xo, yo, xi, yi, tileSize, tileSize)
              .hw_accelerate(xi, xo);
            //padded16.stream_to_accelerator();

            grad_x.compute_at(hw_output, xo);
            grad_y.compute_at(hw_output, xo);
            lxx.compute_at(hw_output, xo);
            lyy.compute_at(hw_output, xo);
            lxy.compute_at(hw_output, xo);
            lgxx.compute_at(hw_output, xo);
            lgyy.compute_at(hw_output, xo);
            lgxy.compute_at(hw_output, xo);
            ////cim.linebuffer();
            cim.compute_at(hw_output, xo);
            cim_output.compute_at(hw_output, xo);

            lgxx.update().unroll(box.x).unroll(box.y);
            lgyy.update().unroll(box.x).unroll(box.y);
            lgxy.update().unroll(box.x).unroll(box.y);

            //padded16.compute_at(hw_output, xo);
            padded16.stream_to_accelerator();
            //hw_input_copy.compute_root();
          }
        } else {    // schedule to CPU
          output.tile(x, y, xo, yo, xi, yi, 58, 58);
        
          grad_x.compute_at(output, xo).vectorize(x, 8);
          grad_y.compute_at(output, xo).vectorize(x, 8);
          grad_xx.compute_at(output, xo).vectorize(x, 4);
          grad_yy.compute_at(output, xo).vectorize(x, 4);
          grad_xy.compute_at(output, xo).vectorize(x, 4);

          //grad_xx.compute_with(grad_yy, x);
          //grad_xy.compute_with(grad_yy, x);
          
          lgxx.compute_at(output, xo).vectorize(x, 4);
          lgyy.compute_at(output, xo).vectorize(x, 4);
          lgxy.compute_at(output, xo).vectorize(x, 4);
          cim.compute_at(output, xo).vectorize(x, 4);

          lgxx.update(0).unroll(box.x).unroll(box.y);
          lgyy.update(0).unroll(box.x).unroll(box.y);
          lgxy.update(0).unroll(box.x).unroll(box.y);

          output.fuse(xo, yo, xo).parallel(xo).vectorize(xi, 4);
        }
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(HarrisCornerDetector, harris)

