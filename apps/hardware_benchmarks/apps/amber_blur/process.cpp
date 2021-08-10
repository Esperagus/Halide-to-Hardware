#include <cstdio>
#include "hardware_process_helper.h"
#include "halide_image_io.h"

#if defined(WITH_CPU)
   #include "amber_blur.h"
#endif

#if defined(WITH_COREIR)
    #include "coreir_interpret.h"
#endif

#if defined(WITH_CLOCKWORK)
    #include "rdai_api.h"
    #include "clockwork_sim_platform.h"
    #include "amber_blur_clockwork.h"
#endif

using namespace Halide::Tools;
using namespace Halide::Runtime;

#define DTYPE uint16_t
//#define DTYPE uint8_t

int main( int argc, char **argv ) {
  std::map<std::string, std::function<void()>> functions;
  OneInOneOut_ProcessController<DTYPE> processor("amber_blur");

  #if defined(WITH_CPU)
      auto cpu_process = [&]( auto &proc ) {
        amber_blur( proc.input, proc.output );
      };
      functions["cpu"] = [&](){ cpu_process( processor ); } ;
  #endif
  
  #if defined(WITH_COREIR)
      auto coreir_process = [&]( auto &proc ) {
          run_coreir_on_interpreter<>( "bin/design_top.json",
                                       proc.input, proc.output,
                                       "self.in_arg_0_0_0", "self.out_0_0" );
      };
      functions["coreir"] = [&](){ coreir_process( processor ); };
  #endif
  
  #if defined(WITH_CLOCKWORK)
      auto clockwork_process = [&]( auto &proc ) {
        RDAI_Platform *rdai_platform = RDAI_register_platform( &rdai_clockwork_sim_ops );
        if ( rdai_platform ) {
          printf( "[RUN_INFO] found an RDAI platform\n" );
          amber_blur_clockwork( proc.input, proc.output );
          RDAI_unregister_platform( rdai_platform );
        } else {
          printf("[RUN_INFO] failed to register RDAI platform!\n");
        }
      };
      functions["clockwork"] = [&](){ clockwork_process( processor ); };
  #endif

  // Add all defined functions
  processor.run_calls = functions;

  auto env_sch = getenv("schedule_j");
  auto schedule_j = env_sch ? atoi(env_sch) : 4;
  std::cout << "using schedule = " << schedule_j << std::endl;

  int output_tile_width  = 62;
  int output_tile_height = output_tile_width;
  
  //int input_width  = 1242;
  int host_tiling, glb_tiling;
  switch (schedule_j) {
  case 1:
    processor.inputs_preset = true;
    host_tiling = 5;
    glb_tiling = 2;
    break;
  case 2:
  case 3:
    processor.inputs_preset = true;
    host_tiling = 9;
    glb_tiling = 7;
    output_tile_width = 94;
    output_tile_height = 62;
    break;
  case 4:
    processor.inputs_preset = false;
    host_tiling = 1;
    glb_tiling = 1;
    break;
  default:
    processor.inputs_preset = false;
    host_tiling = 1;
    glb_tiling = 1;
    break;
  }

  int num_tiles          = host_tiling * glb_tiling;
  int output_width       = num_tiles * output_tile_width;
  int output_height      = num_tiles * output_tile_height;

  std::cout << "Running with output size: " << output_width << "x" << output_height << std::endl;
  processor.input  = Buffer<DTYPE>(output_width+2, output_height+2);
  processor.output = Buffer<DTYPE>(output_width, output_height);

  if (schedule_j == 2 || schedule_j == 3) {
    // load this 6000x4000 image
    std::cout << "Using a big tern image" << std::endl;
    //processor.input = load_and_convert_image("../../images/tern_biggray.png"); // 8bit
    processor.input = load_and_convert_image("../../images/tern_biggray.16b.png"); // 16bit
    
  } else {
    for (int y = 0; y < processor.input.dim(1).extent(); y++) {
      for (int x = 0; x < processor.input.dim(0).extent(); x++) {
        processor.input(x, y) = x + y;
      }
    }
  } 
  //processor.input   = Buffer<DTYPE>(64, 64);
  //processor.output  = Buffer<DTYPE>(62, 62);
  
  return processor.process_command(argc, argv);
  
}
