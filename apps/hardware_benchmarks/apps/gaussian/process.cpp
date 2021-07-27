#include <cstdio>
#include "hardware_process_helper.h"
#include "halide_image_io.h"

#if defined(WITH_CPU)
   #include "gaussian.h"
#endif

#if defined(WITH_COREIR)
    #include "coreir_interpret.h"
#endif

#if defined(WITH_CLOCKWORK)
    #include "rdai_api.h"
    #include "clockwork_sim_platform.h"
    #include "gaussian_clockwork.h"
#endif

using namespace Halide::Tools;
using namespace Halide::Runtime;

int main( int argc, char **argv ) {
  std::map<std::string, std::function<void()>> functions;
  OneInOneOut_ProcessController<uint8_t> processor("gaussian");
  //OneInOneOut_ProcessController<uint16_t> processor("gaussian");

  #if defined(WITH_CPU)
      auto cpu_process = [&]( auto &proc ) {
        gaussian( proc.input, proc.output );
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
          gaussian_clockwork( proc.input, proc.output );
          RDAI_unregister_platform( rdai_platform );
        } else {
          printf("[RUN_INFO] failed to register RDAI platform!\n");
        }
      };
      functions["clockwork"] = [&](){ clockwork_process( processor ); };
  #endif

  // Add all defined functions
  processor.run_calls = functions;

  //int input_width  = 1242;
  int input_width  = 64;
  int input_height = input_width;
  processor.input  = Buffer<uint8_t>(input_width, input_height);
  processor.output = Buffer<uint8_t>(input_width-2, input_height-2);
  
  //processor.inputs_preset = true;
  //for (int y = 0; y < processor.input.dim(1).extent(); y++) {
  //    for (int x = 0; x < processor.input.dim(0).extent(); x++) {
  //      processor.input(x, y) = x + y;
  //    }
  //}
        
  
  //processor.input   = Buffer<uint16_t>(64, 64);
  //processor.output  = Buffer<uint16_t>(62, 62);
  
  return processor.process_command(argc, argv);
  
}
