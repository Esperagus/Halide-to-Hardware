#include <cstdio>
#include <functional>
#include <map>
#include <string>

#include "HalideBuffer.h"

template <class T>
class ProcessController {
 public:
  ProcessController(std::string app_name) :
    design_name(app_name) {
  }
  
  int process_command(int argc, char **argv);
  
  virtual int make_image_def(std::vector<std::string> args);
  virtual int make_run_def(std::vector<std::string> args);
  virtual int make_compare_def(std::vector<std::string> args);
  virtual int make_test_def(std::vector<std::string> args);
  virtual int make_eval_def(std::vector<std::string> args);

  virtual void print_usage();

  // names
  std::string hardware_name;
  std::string design_name;
  
};

template <class T>
class OneInOneOut_ProcessController : public ProcessController<T> {
 public:
 OneInOneOut_ProcessController(std::string app_name, std::map<std::string, std::function<void()>> ops) :
   ProcessController<T>(app_name), inputs_preset(false), run_calls(ops), design_name(app_name) { }
 OneInOneOut_ProcessController(std::string app_name) :
   ProcessController<T>(app_name), inputs_preset(false), design_name(app_name) { }

  // overridden methods
  virtual int make_image_def(std::vector<std::string> args);
  virtual int make_run_def(std::vector<std::string> args);
  virtual int make_compare_def(std::vector<std::string> args);
  virtual int make_test_def(std::vector<std::string> args);
  virtual int make_eval_def(std::vector<std::string> args);

  // buffers
  Halide::Runtime::Buffer<T> input;
  Halide::Runtime::Buffer<T> output;
  bool inputs_preset;
  std::map<std::string, std::function<void()>> run_calls;

  // names
  std::string design_name;

};

template <class T>
class ManyInOneOut_ProcessController : public ProcessController<T> {
 public:
  ManyInOneOut_ProcessController(std::string app_name, std::vector<std::string> filenames,
                                std::map<std::string, std::function<void()>> ops) :
    ProcessController<T>(app_name), input_filenames(filenames), inputs_preset(false), design_name(app_name) {
    for (auto filename : filenames) {
      inputs[filename] = Halide::Runtime::Buffer<T>();
    }
    run_calls = ops;
  }
  ManyInOneOut_ProcessController(std::string app_name, std::vector<std::string> filenames) :
    ProcessController<T>(app_name), input_filenames(filenames), inputs_preset(false), design_name(app_name) {
    for (auto filename : filenames) {
      inputs[filename] = Halide::Runtime::Buffer<T>();
    }
  }

  // overridden methods
  virtual int make_image_def(std::vector<std::string> args);
  virtual int make_run_def(std::vector<std::string> args);
  virtual int make_compare_def(std::vector<std::string> args);
  virtual int make_test_def(std::vector<std::string> args);
  virtual int make_eval_def(std::vector<std::string> args);

  // buffers
  std::vector<std::string> input_filenames;
  std::map<std::string, Halide::Runtime::Buffer<T>> inputs;
  Halide::Runtime::Buffer<T> output;
  bool inputs_preset;
  std::map<std::string, std::function<void()>> run_calls;

  // names
  std::string design_name;

};
