#include <iostream>
#include <math.h>
#include <cstdio>
#include "hardware_process_helper.h"
#include "halide_image_io.h"

#if defined(WITH_CPU)
   #include "resnet_one_input.h"
#endif

#if defined(WITH_COREIR)
    #include "coreir_interpret.h"
#endif

#if defined(WITH_CLOCKWORK)
    #include "rdai_api.h"
    #include "clockwork_sim_platform.h"
    #include "resnet_one_input_clockwork.h"
#endif

using namespace Halide::Tools;
using namespace Halide::Runtime;

int main( int argc, char **argv ) {
  std::map<std::string, std::function<void()>> functions;
  ManyInOneOut_ProcessController<uint16_t> processor("resnet_one_input", {"input.mat"});

  #if defined(WITH_CPU)
      auto cpu_process = [&]( auto &proc ) {
        resnet_one_input(proc.inputs["input.mat"], proc.output);
      };
      functions["cpu"] = [&](){ cpu_process( processor ); } ;
  #endif
  
  #if defined(WITH_COREIR)
      auto coreir_process = [&]( auto &proc ) {
          run_coreir_on_interpreter<>( "bin/design_top.json",
                                       proc.inputs["input.mat"], proc.output,
                                       "self.in_arg_0_0_0", "self.out_0_0" );
      };
      functions["coreir"] = [&](){ coreir_process( processor ); };
  #endif
  
  #if defined(WITH_CLOCKWORK)
      auto clockwork_process = [&]( auto &proc ) {
        RDAI_Platform *rdai_platform = RDAI_register_platform( &rdai_clockwork_sim_ops );
        if ( rdai_platform ) {
          printf( "[RUN_INFO] found an RDAI platform\n" );
          resnet_one_input_clockwork(proc.inputs["input.mat"], proc.output);
          RDAI_unregister_platform( rdai_platform );
        } else {
          printf("[RUN_INFO] failed to register RDAI platform!\n");
        }
      };
      functions["clockwork"] = [&](){ clockwork_process( processor ); };
  #endif

    // Add all defined functions
    processor.run_calls = functions;

    // Set enviroment variable to set these:
    //  HALIDE_GEN_ARGS="in_img=28 pad=1 ksize=3 stride=1 k_ic=8 k_oc=6"

    auto OX = getenv("in_img");
    auto P = getenv("pad");
    auto K = getenv("ksize");
    auto S = getenv("stride");
    auto IC = getenv("k_ic");
    auto OC = getenv("k_oc");

    auto in_img = OX ? atoi(OX) : 28;
    auto pad = 0;
    auto ksize = K ? atoi(K) : 3;
    auto stride = S ? atoi(S) : 1;
    auto k_ic = IC ? atoi(IC) : 8;
    //auto k_ic = IC ? atoi(IC) : 4;
    //auto k_ic = IC ? atoi(IC) : 1;
    //auto k_oc = OC ? atoi(OC) : 6;
    auto k_oc = OC ? atoi(OC) : 3;
    //auto k_oc = OC ? atoi(OC) : 1;

    int X = in_img;
    int Y = X;
    int K_X = ksize;
    int K_Y = K_X;
    int P_X = pad;
    int P_Y = P_X;
    int Z = k_ic; // input channel 
    int W = k_oc; // output channel

    std::cout << "using inputs set within process.cpp" << std::endl;
    processor.inputs_preset = true;

    
    ///// INPUT IMAGE /////
    processor.inputs["input.mat"] = Buffer<uint16_t>(Z, X, Y);
    auto input_copy_stencil = processor.inputs["input.mat"];
    int i=1;
    int max_rand = pow(2,8) - 1;
    // srand(1);
    for (int y = 0; y < input_copy_stencil.dim(2).extent(); y++) {
      for (int x = 0; x < input_copy_stencil.dim(1).extent(); x++) {
        for (int z = 0; z < input_copy_stencil.dim(0).extent(); z++) {
          //input_copy_stencil(z, x, y) = z + x + y;      // diagonal
          input_copy_stencil(z, x, y) = 1;              // all ones
          // input_copy_stencil(z, x, y) = i;    i = i+1;  // increasing
          // if (rand() % 100 < 60) { // 60% zero, else rand
          //   input_copy_stencil(z, x, y) = 0;
          // } else {
          //   input_copy_stencil(z, x, y) = (rand() % (max_rand));
          // }
    } } }

    std::cout << "input has dims: " << processor.inputs["input.mat"].dim(0).extent() << "x"
              << processor.inputs["input.mat"].dim(1).extent() << "x"
              << processor.inputs["input.mat"].dim(2).extent() << "\n";

    bool write_images = true;

    Buffer<uint16_t> full_input(Z, X + pad*2, Y + pad*2);
    Buffer<uint16_t> oned_input(Z *(X + pad*2), Y + pad*2);
    Buffer<uint16_t> interleaved_input(Z *(X + pad*2) * 2, Y + pad*2);
    std::vector<Buffer<uint16_t>> inputs;
    for (int i=0; i<Z; ++i) {
      inputs.emplace_back(Buffer<uint16_t>(X + pad*2, Y + pad*2));
    }
    
    for (int y = 0; y < full_input.dim(2).extent(); y++) {
      for (int x = 0; x < full_input.dim(1).extent(); x++) {
        for (int z = 0; z < input_copy_stencil.dim(0).extent(); z++) {
          auto x_coord = x-pad<0 ? 0 : x-pad >= X ? X-1 : x-pad;
          auto y_coord = y-pad<0 ? 0 : y-pad >= Y ? Y-1 : y-pad;
          full_input(z, x, y) = input_copy_stencil(z, x_coord, y_coord);
          oned_input(z + Z*x, y) = input_copy_stencil(z, x_coord, y_coord);
          std::cout << z+Z*x <<" "<< y << " " << input_copy_stencil(z, x_coord, y_coord) << std::endl;
          interleaved_input(2*(z + Z*x) + 0, y) = input_copy_stencil(z, x_coord, y_coord);
          interleaved_input(2*(z + Z*x) + 1, y) = input_copy_stencil(z, x_coord, y_coord);
          
          inputs[z](x, y) = input_copy_stencil(z, x_coord, y_coord);
          //std::cout << "input " << z << "," << x << "," << y << " = " << std::hex << +full_input(z,x,y) << std::dec << std::endl;
        } } }
    //std::cout << "input 3,2 = 31 ?= " << +full_input(3,2) << std::endl;
    //save_image(full_input, "bin/input.mat");
    std::cout << "num_dims=" << oned_input.dimensions() << std::endl
              << oned_input.dim(0).extent() << "x" << oned_input.dim(1).extent()
              << std::endl;
    if (write_images) {
      save_image(oned_input, "bin/input.pgm");
      std::cout << "wrote an image\n";
      // save_image(interleaved_input, "bin/input_interleaved.png");
      // for (size_t i=0; i<inputs.size(); ++i) {
      //   save_image(inputs[i], "bin/input_" + std::to_string(i) + ".mat");
      // }
    }

    
    int imgsize_x = X;
    int imgsize_y = Y;
    processor.output = Buffer<uint16_t>(imgsize_x, imgsize_y, W);

    std::cout << "output has dims: " << processor.output.dim(0).extent() << "x"
              << processor.output.dim(1).extent() << "x"
              << processor.output.dim(2).extent() << "\n";


    auto output = processor.process_command(argc, argv);
    //std::cout << "output 0 0 ?= " << +processor.output(8,4,0) << std::endl;

    Buffer<uint16_t> oned_output(imgsize_x * imgsize_y, W);
    std::vector<Buffer<uint16_t>> outputs;
    for (int i=0; i<W; ++i) {
      outputs.emplace_back(Buffer<uint16_t>(imgsize_x, imgsize_y));
    }

    for (int w = 0; w < processor.output.dim(2).extent(); w++) {
      for (int y = 0; y < processor.output.dim(1).extent(); y++) {
        for (int x = 0; x < processor.output.dim(0).extent(); x++) {
          oned_output(x + y*imgsize_x, w) = processor.output(x,y,w);
          outputs[w](x,y) = processor.output(x,y,w);
          //std::cout << x << "," << y << "," << w << " = " << +processor.output(x,y,w) << " = " << std::hex << +processor.output(x,y,w) << std::dec << std::endl;
        } } }
    if (write_images) {//if (processor.output(0,0,0,0) == 198) {
      save_image(oned_output, "bin/output_gold.mat");
      //save_image(processor.output, "bin/output_gold.mat");
      // for (size_t i=0; i<outputs.size(); ++i) {
      //   save_image(outputs[i], "bin/output_" + std::to_string(i) + ".mat");
      // }
    }

    
    return output;
}  
