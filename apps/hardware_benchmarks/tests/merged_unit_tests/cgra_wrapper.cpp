#include <cassert>
#include <cstdint>
#include <iostream>

#include "cgra_wrapper.h"

#include "coreir/libs/commonlib.h"
#include "coreir/libs/float.h"

using namespace CoreIR;

using namespace std;

#include "test_utils.h"

extern "C" {
uint8_t* _halide_buffer_get_host(const halide_buffer_t* buf);
}

CGRAWrapper::CGRAWrapper() {
  c = CoreIR::newContext();

  CoreIRLoadLibrary_commonlib(c);
  CoreIRLoadLibrary_float(c);
  if (!loadFromFile(c, "./conv_3_3_app.json")) {
    cout << "Error: Could not load json for CGRA accelerator!" << endl;
    c->die();
  }
  c->runPasses({"rungenerators", "flattentypes", "flatten", "wireclocks-coreir"});
  m = c->getNamespace("global")->getModule("DesignTop");
  cout << "Module..." << endl;
  m->print();
  state = new CoreIR::SimulatorState(m);
  resetSim("self.in_arg_0_0_0", m, *state);
  cout << "Initialized simulator..." << endl;
}

CGRAWrapper::~CGRAWrapper() {
  delete state;
  CoreIR::deleteContext(c);
}

// Q: How do we start to stream from buffer? Note that this function must
// write to the simulator and read out and buffer its data to a temporary
// buffer in order to save it
void CGRAWrapper::subimage_to_stream(halide_buffer_t* buf, int32_t subImageOffset,
    int stride_0, int subimage_extent_0,
    int stride_1, int subimage_extent_1,
    int stride_2, int subimage_extent_2,
    int stride_3, int subimage_extent_3) {

  cout << "subimage to stream on buffer at offset " << subImageOffset << " with extents..." << endl;
  cout << "\t" << subimage_extent_0 << endl;
  cout << "\t" << subimage_extent_1 << endl;
  cout << "\t" << subimage_extent_2 << endl;
  cout << "\t" << subimage_extent_3 << endl;

  assert(stride_3 == 1 && subimage_extent_3 == 1);

  CoordinateVector<int> writeIdx({"y", "x", "c"}, {subimage_extent_1 - 1,
      subimage_extent_0 - 1,
      subimage_extent_2 - 1});

  uint16_t* hostBuf = (uint16_t*) _halide_buffer_get_host(buf);
  while (!writeIdx.allDone()) {
    cout << "Write index = " << writeIdx.coordString() << endl;
    auto i = writeIdx.coord("x");
    auto j = writeIdx.coord("y");
    auto k = writeIdx.coord("c");
    // TODO: Replace with 4th dimension stride when that comes up
    auto m = 1;

    int offset = subImageOffset + 
      stride_0 * i +
      stride_1 * j + 
      stride_2 * k + 
      stride_3 * m;
    uint16_t nextInPixel = hostBuf[offset];

    string input_name = "self.in_arg_0_0_0";
    cout << "Next pixel = " << nextInPixel << endl;
    state->setValue("self.in_en", BitVector(1, true));
    state->setValue(input_name, BitVector(16, nextInPixel));
    state->exeCombinational();


    bool valid_value = state->getBitVec("self.valid").to_type<bool>();
    if (valid_value) {

      auto output_bv = state->getBitVec("self.out_0_0");
      int output_value;
      output_value = output_bv.to_type<int>();
      cout << "\tOutput value = " << output_value << endl;
    }
    
    pixelOutputs.push_back(23);
    pixelOutputs.push_back(23);
    pixelOutputs.push_back(23);

    state->exeSequential();

    writeIdx.increment();
  }

  pixelOutputs.clear();
  assert(pixelOutputs.size() == 0);

  //for (int i = 0; i < subimage_extent_0; i++) {
    //for (int j = 0; j < subimage_extent_1; j++) {
      //for (int k = 0; k < subimage_extent_2; k++) {
        //for (int m = 0; m < subimage_extent_3; m++) {
        //}
      //}
    //}
  //}
}

void CGRAWrapper::stream_to_subimage(halide_buffer_t* buf, int32_t subImageOffset,
    int stride_0, int subimage_extent_0,
    int stride_1, int subimage_extent_1,
    int stride_2, int subimage_extent_2,
    int stride_3, int subimage_extent_3) {

  cout << "subimage to stream on buffer with extents..." << endl;
  cout << "\t" << subimage_extent_0 << endl;
  cout << "\t" << subimage_extent_1 << endl;
  cout << "\t" << subimage_extent_2 << endl;
  cout << "\t" << subimage_extent_3 << endl;
  
  uint16_t* dataPtr = (uint16_t*)_halide_buffer_get_host(buf);
  int nextPix = 0;
  for (int i = 0; i < subimage_extent_0; i++) {
    for (int j = 0; j < subimage_extent_1; j++) {
      for (int k = 0; k < subimage_extent_2; k++) {
        for (int m = 0; m < subimage_extent_3; m++) {
          int offset = subImageOffset + 
            stride_0 * i +
            stride_1 * j + 
            stride_2 * k + 
            stride_3 * m;
          assert(nextPix < ((int) pixelOutputs.size()));
          dataPtr[offset] = pixelOutputs[nextPix];
          nextPix++;
        }
      }
    }
  }

  cout << "nextPix = " << nextPix << endl;
  cout << "pixels  = " << pixelOutputs.size() << endl;
  //assert(nextPix == ((int) pixelOutputs.size()));
}