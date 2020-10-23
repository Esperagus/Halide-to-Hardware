#include <iostream>
#include <fstream>

#include "CodeGen_C.h"
#include "CodeGen_Internal.h"
#include "CoreIRCompute.h"
#include "HWBufferUtils.h"
#include "Substitute.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "Simplify.h"

#include "coreir.h"
#include "coreir/libs/commonlib.h"
#include "coreir/libs/float.h"
#include "lakelib.h"
#include "cgralib.h"

namespace Halide {
namespace Internal {

using std::ostream;
using std::endl;
using std::string;
using std::vector;
using std::map;
using std::ostringstream;
using std::ofstream;
using std::cout;

namespace {

std::ostream& operator<<(std::ostream& os, const std::vector<int>& vec) {
  os << "[";
  for (size_t i=0; i<vec.size(); ++i) {
    os << vec.at(i);
    if (i < vec.size() - 1) {
      os << ",";
    }
  }
  os << "]";
  return os;
};

std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& vec) {
  os << "[";
  for (size_t i=0; i<vec.size(); ++i) {
    os << vec.at(i);
    if (i < vec.size() - 1) {
      os << ",";
    }
  }
  os << "]";
  return os;
};

vector<CoreIR::Wireable*> get_wires(CoreIR::Wireable* base_wire, const vector<size_t> ports) {
  int num_ports = 1;
  for (const auto& port_length : ports) {
    num_ports *= port_length;
  }
//  cout << "we have " << num_ports << " ports\n";
  vector<CoreIR::Wireable*> all_wires(num_ports);

  vector<uint> port_idxs(ports.size());

  for (int idx = 0; idx < num_ports; ++idx) {
    // find the wire associated with the indices
    CoreIR::Wireable* cur_wire = base_wire;
    for (const auto& port_idx : port_idxs) {
      cur_wire = cur_wire->sel(port_idx);
    }

    // add the wire to our list
    all_wires.at(idx) = cur_wire;

    // increment  index
    port_idxs.at(0) += 1;
    for (size_t dim = 0; dim < port_idxs.size(); ++dim) {
      if (port_idxs.at(dim) >= ports.at(dim)) {
        port_idxs.at(dim) = 0;
        if (dim + 1 < port_idxs.size()) {
          port_idxs.at(dim+1) += 1;
        }
      }
    }
  }

  return all_wires;
}

vector<CoreIR::Wireable*> get_wires(CoreIR::Wireable* base_wire, const vector<size_t> ports, string wire_prefix) {
  int num_ports = 1;
  for (const auto& port_length : ports) {
    num_ports *= port_length;
  }

  vector<CoreIR::Wireable*> all_wires(num_ports);

  vector<uint> port_idxs(ports.size());

  for (int idx = 0; idx < num_ports; ++idx) {
    // find the wire associated with the indices
    string selstr = "";
    for (const auto& port_idx : port_idxs) {
      selstr += std::to_string(port_idx) + ".";
    }
    //sel_str = sel_str.substr(0, sel_str.size()-1);
    selstr.pop_back();

    // add the wire to our list
    all_wires.at(idx) = base_wire->sel(wire_prefix + selstr);

    // increment  index
    port_idxs.at(0) += 1;
    for (size_t dim = 0; dim < port_idxs.size(); ++dim) {
      if (port_idxs.at(dim) >= ports.at(dim)) {
        port_idxs.at(dim) = 0;
        if (dim + 1 < port_idxs.size()) {
          port_idxs.at(dim+1) += 1;
        }
      }
    }
  }

  return all_wires;
}


void connect_wires(CoreIR::ModuleDef *def, vector<CoreIR::Wireable*> in_wires, vector<CoreIR::Wireable*> out_wires) {
  assert(in_wires.size() == out_wires.size());

  for (size_t idx=0; idx<in_wires.size(); ++idx) {
    def->connect(in_wires.at(idx), out_wires.at(idx));
  }
}


map<string, string> coreir_generators(CoreIR::Context* context) {
  // set up coreir generation
  static map<string, string> gens;

  if(!gens.empty()) return gens;

  // add all generators from coreirprims
  context->getNamespace("coreir");
  std::vector<string> corelib_gen_names = {"mul", "add", "sub",
                                           "udiv", "sdiv", "urem", "srem",
                                           "and", "or", "xor", "not",
                                           "eq", "neq",
                                           "ult", "ugt", "ule", "uge",
                                           "slt", "sgt", "sle", "sge",
                                           "shl", "ashr", "lshr",
                                           "mux", "const", "wire"};

  for (auto gen_name : corelib_gen_names) {
    gens[gen_name] = "coreir." + gen_name;
    internal_assert(context->hasGenerator(gens[gen_name]))
      << "could not find " << gen_name << "\n";
  }

  // add all generators from commonlib
  CoreIRLoadLibrary_commonlib(context);
  std::vector<string> commonlib_gen_names = {"umin", "smin", "umax", "smax",
                                             "counter", //"linebuffer",
                                             "muxn", "abs", "absd",
                                             "reg_array", "reshape", "transpose_reshape"
  };
  for (auto gen_name : commonlib_gen_names) {
    gens[gen_name] = "commonlib." + gen_name;
    internal_assert(context->hasGenerator(gens[gen_name]))
      << "could not find " << gen_name << "\n";
  }

  // add all generators from fplib which include floating point operators
  CoreIRLoadLibrary_float(context);
  std::vector<string> fplib_gen_names = {"fmul", "fadd", "fsub", "fdiv",
                                         "fneg", "frem",
                                         "fabs", "fmin", "fmax", "fmux",
                                         "feq", "fneq",
                                         "flt", "fgt", "fle", "fge",
                                         "frnd", "fflr", "fceil",
                                         "fsqrt", "fsqr", "fpower", "fexp", "fln",
                                         "fsin", "fcos", "ftan", "ftanh",
                                         "fasin", "facos", "fatan", "fatan2",

  };

  for (auto gen_name : fplib_gen_names) {
    // floating point library does not start with "f"
    gens[gen_name] = "float." + gen_name.substr(1);
    internal_assert(context->hasGenerator(gens[gen_name]))
      << "could not find " << gen_name << "\n";
  }
  gens["fconst"] = "coreir.const";

  // add all modules from corebit
  context->getNamespace("corebit");
  std::vector<string> corebitlib_mod_names = {"bitand", "bitor", "bitxor", "bitnot",
                                              "bitmux", "bitconst"};
  //                                              "bitlt", "bitle", "bitgt", "bitge","bitxnor"};
  for (auto mod_name : corebitlib_mod_names) {
    // these were renamed to using the corebit library
    gens[mod_name] = "corebit." + mod_name.substr(3);
    internal_assert(context->hasModule(gens[mod_name]))
      << "could not find " << mod_name << "\n";
  }


  // add all modules from memory
  context->getNamespace("memory");
  std::vector<string> memorylib_gen_names = {"ram2","rom2",
                                             "fifo",};

  for (auto gen_name : memorylib_gen_names) {
    gens[gen_name] = "memory." + gen_name;
    internal_assert(context->hasGenerator(gens[gen_name]))
      << "could not find " << gen_name << "\n";
  }

  // add all generators from lakelib which include some cgra libs
  CoreIRLoadLibrary_lakelib(context);
  std::vector<string> lakelib_gen_names = {"linebuffer", "unified_buffer",
                                           "new_unified_buffer"};

  for (auto gen_name : lakelib_gen_names) {
    gens[gen_name] = "lakelib." + gen_name;
    internal_assert(context->hasGenerator(gens[gen_name]))
      << "could not find " << gen_name << "\n";
  }

  CoreIRLoadLibrary_cgralib(context);
  gens["lake"] = "cgralib.Mem";
  gens["Mem"] = "cgralib.Mem";
  internal_assert(context->hasGenerator(gens["lake"])) << "could not find " << "cwlib.Mem" << "\n";
  
  // passthrough is now just a mantle wire
  gens["passthrough"] = "mantle.wire";
  assert(context->hasGenerator(gens["passthrough"]));

  return gens;

}

int inst_bitwidth(int input_bitwidth) {
  // FIXME: properly create bitwidths 1, 8, 16, 32
  if (input_bitwidth == 1) {
    return 1;
  } else {
    return 16;
  }
}

uint num_bits(uint N) {
  if (N==0) { return 1; }

  uint num_shifts = 0;
  uint temp_value = N;
  while (temp_value > 0) {
    temp_value  = temp_value >> 1;
    num_shifts++;
  }
  return num_shifts;
}

struct Storage_Def {
  CoreIR::Type* ptype;
  CoreIR::Wireable* wire;
  CoreIR::Wireable* reg = NULL;
  bool was_written = false;
  bool was_read = false;

  Storage_Def(CoreIR::Type* ptype, CoreIR::Wireable* wire) :
    ptype(ptype), wire(wire) {}

  bool is_reg() {return (reg != NULL); };
};

class CreateCoreIRModule : public CodeGen_C {
  using CodeGen_C::visit;
  
  CoreIR::ModuleDef* def;
  CoreIR::Context* context;
  std::map<std::string,std::string> gens;
  CoreIR::Wireable* self = NULL;
  CoreIR::Wireable* output_wire = NULL;

  // keep track of coreir dag
  std::map<std::string,CoreIR::Wireable*> hw_wire_set;
  std::map<std::string,std::shared_ptr<Storage_Def>> hw_store_set; // not needed
  std::map<std::string,std::shared_ptr<CoreIR_Inst_Args>> hw_def_set; // not needed
  std::map<std::string,CoreIR::Wireable*> hw_input_set;
  std::set<std::string> hw_output_set; // not needed

  void record_inputs(CoreIR_Interface);

  bool is_wire(string);
  bool is_input(string);
  bool is_output(string);
  bool is_storage(string);
  bool is_defined(string);

  CoreIR::Wireable* get_wire(std::string name, Expr e, std::vector<uint> indices={});
  void rename_wire(std::string new_name, std::string in_name, Expr in_expr, std::vector<uint> indices={});
  void add_wire(std::string name, CoreIR::Wireable* wire, std::vector<uint> indices={});
  
  // coreir operators
  void visit_unaryop(Type t, Expr a, const char* op_sym, std::string op_name);
  void visit(const Not *op);
  void visit_binop(Type t, Expr a, Expr b, const char* op_sym, std::string op_name);
  void visit(const Mul *op);
  void visit(const Div *op);
  void visit(const Mod *op);
  void visit(const Add *op);
  void visit(const Sub *op);
  void visit(const And *op);
  void visit(const Or *op);
  void visit(const EQ *op);
  void visit(const NE *op);
  void visit(const LT *op);
  void visit(const LE *op);
  void visit(const GT *op);
  void visit(const GE *op);
  void visit(const Max *op);
  void visit(const Min *op);
  void visit(const Cast *op);
  void visit_ternop(Type t, Expr a, Expr b, Expr c, const char*  op_sym1, const char* op_sym2, std::string op_name);
  void visit(const Select *op);
  void visit(const Call *op);
  void visit(const Load *op);
  
public:
  CreateCoreIRModule(ostream& stream, CoreIR::ModuleDef* def, CoreIR_Interface iface, CoreIR::Context* context) :
    CodeGen_C(stream, Target(), CPlusPlusImplementation, ""), def(def), context(context), self(def->sel("self")) {
    gens = coreir_generators(context);
    record_inputs(iface);
    output_wire = self->sel("out_" + iface.output.name);
  }

  void connect_output(Expr e) {
    string name = print_expr(e);
    auto wire = get_wire(name, e);
    def->connect(wire, output_wire);
    stream << "// connected " << name << " to the output port" << endl;
  }

  void add_coreir_inst(CoreIR_Inst_Args coreir_inst) {
    stream << "// adding coreir inst " << coreir_inst.ref_name << std::endl;
    hw_def_set[coreir_inst.ref_name] = std::make_shared<CoreIR_Inst_Args>(coreir_inst);
  }
  
};

void add_coreir_compute(Expr e, CoreIR::ModuleDef* def, CoreIR_Interface iface,
                        vector<CoreIR_Inst_Args> coreir_insts, CoreIR::Context* context) {
  //ofstream compute_debug_file("bin/compute_debug_" + iface.name + ".cpp");
  ostringstream compute_debug_str;
  
  CreateCoreIRModule ccm(compute_debug_str, def, iface, context);
  //CreateCoreIRModule ccm(std::cout, def, iface, context);
  compute_debug_str.str("");
  compute_debug_str.clear();

  for (auto coreir_inst : coreir_insts) {
    ccm.add_coreir_inst(coreir_inst);
  }

  e.accept(&ccm);

  //ofstream compute_debug_file("/dev/null");
  ofstream compute_debug_file("bin/clockwork_compute_debug.cpp", std::ios::app);
  ccm.connect_output(e);
  compute_debug_file << iface.name << "() {" << endl
                     << compute_debug_str.str()
                     << "}" << endl << endl;
  compute_debug_file.close();
}


void add_coreir_compute(Expr e, CoreIR::ModuleDef* def, CoreIR_Interface iface, CoreIR::Context* context) {
  vector<CoreIR_Inst_Args> coreir_insts;
  add_coreir_compute(e, def, iface, coreir_insts, context);
}

void CreateCoreIRModule::record_inputs(CoreIR_Interface iface) {
  for (size_t i=0; i<iface.inputs.size(); ++i) {
    auto input_bundle = iface.inputs[i];
    string bundle_name = "in" + std::to_string(i) + "_" + input_bundle.name;
    auto bundle_wire = self->sel(bundle_name);

    if (false) {//if (input_bundle.ports.size() == 1) {
      auto input_name = print_name(input_bundle.ports[0].name);
      hw_input_set[input_name] = bundle_wire;
      stream << "// recording input " << input_name << std::endl;
      continue;
    }

    // Use slice to index part of each bundled port
    //uint bit_start = 0;
    //for (auto input : input_bundle.ports) {
    //  uint bit_end = bit_start + input.bitwidth;
    //  CoreIR::Values sliceArgs = {{"width", CoreIR::Const::make(context, input_bundle.total_bitwidth)},
    //                              {"lo", CoreIR::Const::make(context, bit_start)},
    //                              {"hi", CoreIR::Const::make(context, bit_end)}};
    //  auto slice = def->addInstance(input.name + "_slice","coreir.slice",sliceArgs);
    //  
    //  def->connect(bundle_wire, slice->sel("in"));
    //  hw_input_set[print_name(input.name)] = slice->sel("out");
    //  stream << "// recording input " << print_name(input.name) << std::endl;
    //
    //  bit_start = bit_end;
    //}

    for (size_t j=0; j<input_bundle.ports.size(); ++j) {
      auto input = input_bundle.ports.at(j);
      hw_input_set[print_name(input.name)] = bundle_wire->sel(j);
      stream << "// recording input " << print_name(input.name) << std::endl;
    }
    
  }

  for (auto index : iface.indices) {
    string index_name = index.name;
    auto index_wire = self->sel(index_name);
    hw_input_set[print_name(index_name)] = index_wire;
  }
}

/////////////////////////////////////////////////////
// Functions to identify expressions and variables //
/////////////////////////////////////////////////////
int get_const_bitwidth(const Expr e) {
  if (const IntImm* e_int = e.as<IntImm>()) {
    return e_int->type.bits();

  } else if (const UIntImm* e_uint = e.as<UIntImm>()) {
    return e_uint->type.bits();

  } else if (const FloatImm* e_float = e.as<FloatImm>()) {
    return e_float->type.bits();

  } else {
    internal_error << "invalid constant expr\n";
    return -1;
  }
}

bool is_iconst(const Expr e) {
  if (e.as<IntImm>() || e.as<UIntImm>()) {
    return true;
  } else {
    return false;
  }
}

bool is_fconst(const Expr e) {
  if (e.as<FloatImm>()) {
    return true;
  } else {
    return false;
  }
}

bool CreateCoreIRModule::is_wire(string var_name) {
  bool wire_exists = hw_wire_set.count(var_name) > 0;
  return wire_exists;
}

bool CreateCoreIRModule::is_input(string var_name) {
	bool input_exists = hw_input_set.count(var_name) > 0;
  return input_exists;
}

bool CreateCoreIRModule::is_storage(string var_name) {
  bool storage_exists = hw_store_set.count(var_name) > 0;
  return storage_exists;
}

bool CreateCoreIRModule::is_defined(string var_name) {
  bool hardware_defined = hw_def_set.count(var_name) > 0;
  return hardware_defined;
}

bool CreateCoreIRModule::is_output(string var_name) {
  bool output_name_matches = hw_output_set.count(var_name) > 0;
  return output_name_matches;
}

//////////////////////////////////////////////
// Functions to wire coreir things together //
//////////////////////////////////////////////
CoreIR::Wireable* index_wire(CoreIR::Wireable* in_wire, std::vector<uint> indices) {
  CoreIR::Wireable* current_wire = in_wire;
  for (int i=indices.size()-1; i >= 0; --i) {
    current_wire = current_wire->sel(indices[i]);
  }
  return current_wire;
}

// This function is used when the output is going to be connected to a useable wire
CoreIR::Wireable* CreateCoreIRModule::get_wire(string name, Expr e, std::vector<uint> indices) {
  //cout << "connect for " << name << "\n";
  if (is_iconst(e)) {
    // Create a constant.
    int const_value = id_const_value(e);
    string const_name = const_value >= 0 ?
      unique_name("const_p" + std::to_string(std::abs(const_value)) + "_" + name) :
      unique_name("const_n" + std::to_string(std::abs(const_value)) + "_" + name);
    CoreIR::Wireable* const_inst;

    uint const_bitwidth = get_const_bitwidth(e);
    if (const_bitwidth == 1) {
      const_inst = def->addInstance(const_name, gens["bitconst"], {{"value",CoreIR::Const::make(context,(bool)const_value)}});
    } else {
      int bw = inst_bitwidth(const_bitwidth);
      const_inst = def->addInstance(const_name, gens["const"], {{"width", CoreIR::Const::make(context,bw)}},
                                    {{"value",CoreIR::Const::make(context,BitVector(bw,const_value))}});
    }

    stream << "// created const: " << const_name << " with name " << name << "\n";
    return const_inst->sel("out");

  } else if (is_fconst(e)) {
    // Create a constant with floating point value
    float fconst_value = id_fconst_value(e);
    string const_name = unique_name("fconst" + std::to_string((int)fconst_value) + "_" + name);
    CoreIR::Wireable* const_inst;

    uint const_bitwidth = get_const_bitwidth(e);
    int bw = inst_bitwidth(const_bitwidth);
    const_inst = def->addInstance(const_name, gens["fconst"], {{"width", CoreIR::Const::make(context,bw)}},
                                  {{"value",CoreIR::Const::make(context,BitVector(bw,bfloat16_t(fconst_value).to_bits()))}});
    const_inst->getMetaData()["float_value"] = "Equivalent float = " + std::to_string(fconst_value);

    stream << "// created fconst: " << const_name << " with name " << name << " with float value " << fconst_value << "\n";
    return const_inst->sel("out");

  } else if (is_input(name)) {
    // Return an input wire
    CoreIR::Wireable* input_wire = hw_input_set[name];

    stream << "// " << name << " added as input " << name << endl;
    auto current_wire = index_wire(input_wire, indices);

    return current_wire;

  } else if (is_storage(name)) {
    // Return a reusable element
    auto pt_struct = hw_store_set[name];
    CoreIR::Wireable* current_wire;

    if (!pt_struct->was_written) {
      internal_assert(pt_struct->is_reg());
      current_wire = pt_struct->reg->sel("out");
    } else {
      current_wire = pt_struct->wire->sel("out");
    }

    current_wire = index_wire(current_wire, indices);
    pt_struct->was_read = true;

    return current_wire;

  } else if (is_wire(name)) {
    // Return an existing module wire.

    CoreIR::Wireable* wire = hw_wire_set[name];
    if (wire==NULL) {
      user_warning << "wire was not defined: " << name << "\n";
      stream       << "// wire was not defined: " << name << endl;
      return self->sel("in");
    }
    CoreIR::Wireable* current_wire = index_wire(wire, indices);

    return current_wire;

  } else if (is_defined(name)) {
    // hardware element defined, but not added yet
    std::shared_ptr<CoreIR_Inst_Args> inst_args = hw_def_set[name];
    assert(inst_args);
    stream << "// creating element called: " << name << endl;
    //cout << "named " << inst_args->gen <<endl;
    string inst_name = unique_name(inst_args->name);
    CoreIR::Wireable* inst = def->addInstance(inst_name, inst_args->gen, inst_args->args, inst_args->genargs);
    add_wire(name, inst->sel(inst_args->selname));

    if (inst_args->gen == gens["ram2"]) {
      add_wire(name + "_waddr", inst->sel("waddr"));
      add_wire(name + "_wdata", inst->sel("wdata"));
      add_wire(name + "_raddr", inst->sel("raddr"));

      //attach a read enable
      CoreIR::Wireable* rom_ren = def->addInstance(inst_name + "_ren", gens["bitconst"], {{"value", CoreIR::Const::make(context,true)}});
      def->connect(rom_ren->sel("out"), inst->sel("ren"));

      auto ram_wen = def->addInstance(name + "_wen", gens["bitconst"], {{"value",CoreIR::Const::make(context,false)}});
      def->connect(ram_wen->sel("out"), inst->sel("wen"));
    }

    auto ref_name = inst_args->ref_name;

    return inst->sel(inst_args->selname);

  } else {
    user_warning << "ERROR- invalid wire: " << name << "\n";
    stream << "// invalid wire: couldn't find " << name
           << "\n// hw_wire_set contains: ";
    for (auto x : hw_wire_set) {
      stream << " " << x.first;
    }
    stream << "\n";

    return self->sel("in");
  }
}

// This adds another usable wire, generally the output of a new module.
void CreateCoreIRModule::add_wire(string out_name, CoreIR::Wireable* in_wire, vector<uint> out_indices) {
  //cout << "add wire to " << out_name << "\n";
  if (is_storage(out_name)) {
    auto pt_struct = hw_store_set[out_name];

    if (!pt_struct->was_read && !pt_struct->was_written &&
        out_indices.empty() && !pt_struct->is_reg()) {
      // this passthrough hasn't been used, so remove it
      auto pt_wire = hw_store_set[out_name]->wire;
      internal_assert(CoreIR::Instance::classof(pt_wire));
      CoreIR::Instance* pt_inst = &(cast<CoreIR::Instance>(*pt_wire));
      def->removeInstance(pt_inst);
      hw_store_set.erase(out_name);

      hw_wire_set[out_name] = in_wire;

    } else {
      // use the found passthrough
      if (pt_struct->was_written && pt_struct->was_read) {
        // create a new passthrough, since this one is already connected
        string pt_name = "pt" + out_name + "_" + unique_name('p');
        CoreIR::Type* ptype = pt_struct->ptype;
        stream << "// created passthrough with name " << pt_name << endl;

        CoreIR::Wireable* pt = def->addInstance(pt_name, gens["passthrough"], {{"type",CoreIR::Const::make(context,ptype)}});
        // FIXME: copy over all stores (since wire might not encompass entire passthrough)
        pt_struct->wire = pt;

        pt_struct->was_read = false;
      }

      //cout << "// wiring to " << out_name << endl;
      // disconnect associated wire in reg, and connect new wire
      if (pt_struct->is_reg()) {
        stream << "// disconnecting wire for reg " << out_name << endl;
        CoreIR::Wireable* d_wire = pt_struct->reg->sel("in");
        for (int i=out_indices.size()-1; i >= 0; --i) {
          //cout << "selecting in reg add_wire: " << out_indices[i] << endl;
          d_wire = d_wire->sel(out_indices[i]);
        }
        //cout << "select in add_wire done" << endl;
        def->disconnect(d_wire);
        def->connect(in_wire, d_wire);
      }

      pt_struct->was_written = true;
      stream << "// added passthrough wire to " << out_name << endl;
      //cout << "// added passthrough wire to " << out_name << endl;

      // connect wire to passthrough
      CoreIR::Wireable* current_wire = pt_struct->wire->sel("in");
      for (int i=out_indices.size()-1; i >= 0; --i) {
        //cout << "selecting in add_wire: " << out_indices[i] << endl;
        current_wire = current_wire->sel(out_indices[i]);
      }
      def->connect(in_wire, current_wire);

    }

  } else {
    internal_assert(out_indices.empty());
    // wire is being stored for use as an input
    hw_wire_set[out_name] = in_wire;
  }
}

// This renames a wire with a new name without creating a new module.
void CreateCoreIRModule::rename_wire(string new_name, string in_name, Expr in_expr, vector<uint> indices) {
  //cout << "// rename wire for " << in_name << " to " << new_name << "\n";
  // if this is a definition, then just copy the pointer
  if (is_defined(in_name) && !is_wire(in_name)) {
    assert(indices.empty());
    // wire is only defined but not created yet

    hw_def_set[new_name] = hw_def_set[in_name];
    hw_def_set[new_name]->ref_name = in_name;

    assert(hw_def_set[new_name]);
    stream << "// added/modified in hw_def_set: " << new_name << " = " << in_name << "\n";
    stream << "//   for a module called: " << hw_def_set[new_name]->name << endl;
    return;

  } else if (is_const(in_expr)) {
    assert(indices.empty());
    // add hardware definition, but don't create it yet
    int const_value = id_const_value(in_expr);
    string const_name = const_value >= 0 ?
      "const_p" + std::to_string(std::abs(const_value)) + "_" + in_name :
      "const_n" + std::to_string(std::abs(const_value)) + "_" + in_name;
    string gen_const;
    CoreIR::Values args, genargs;

    uint const_bitwidth = get_const_bitwidth(in_expr);
    if (const_bitwidth == 1) {
      gen_const = gens["bitconst"];
      args = {{"value",CoreIR::Const::make(context,const_value)}};
      genargs = CoreIR::Values();
    } else {
      int bw = inst_bitwidth(const_bitwidth);
      gen_const = gens["const"];
      args = {{"width", CoreIR::Const::make(context,bw)}};
      genargs = {{"value",CoreIR::Const::make(context,BitVector(bw,const_value))}};
    }

    CoreIR_Inst_Args const_args(const_name, in_name, "out", gen_const, args, genargs);
    hw_def_set[new_name] = std::make_shared<CoreIR_Inst_Args>(const_args);

    stream << "// defined const: " << const_name << " with name " << new_name << "\n";
    return;

  } else if (is_fconst(in_expr)) {
    assert(indices.empty());
    float fconst_value = id_fconst_value(in_expr);
    string const_name = unique_name("fconst" + std::to_string((int)fconst_value) + "_" + in_name);

    uint const_bitwidth = get_const_bitwidth(in_expr);
    int bw = inst_bitwidth(const_bitwidth);
    CoreIR::Values args = {{"width", CoreIR::Const::make(context,bw)}};
    CoreIR::Values genargs = {{"value",CoreIR::Const::make(context,BitVector(bw,bfloat16_t(fconst_value).to_bits()))}};

    CoreIR_Inst_Args const_args(const_name, in_name, "out", gens["fconst"], args, genargs);
    hw_def_set[new_name] = std::make_shared<CoreIR_Inst_Args>(const_args);

    stream << "// defined fconst: " << const_name << " with name " << new_name << "\n";
    return;

  } else if (is_storage(in_name) && indices.empty()) {
    // if this is simply another reference (no indexing), just add a new name
    if (is_storage(new_name)) {
      stream << "// removing another passthrough: " << new_name << " = " << in_name << endl;
      //def->connect(hw_store_set[in_name]->wire->sel("out"), hw_store_set[new_name]->wire->sel("in"));

      auto pt_wire = hw_store_set[new_name]->wire;
      internal_assert(CoreIR::Instance::classof(pt_wire));
      CoreIR::Instance* pt_inst = &(cast<CoreIR::Instance>(*pt_wire));
      def->removeInstance(pt_inst);
      hw_store_set[new_name] = hw_store_set[in_name];


      hw_store_set[in_name]->was_read = true;
    } else {
      stream << "// creating another passthrough reference: " << new_name << " = " << in_name << endl;
      hw_store_set[new_name] = hw_store_set[in_name];
    }

    if (is_output(new_name)) {
      if (hw_store_set[new_name]->is_reg()) {
        def->connect(hw_store_set[new_name]->reg->sel("out"), self->sel("out"));
      } else {
        def->connect(hw_store_set[new_name]->wire->sel("out"), self->sel("out"));
      }
      stream << "// connecting passthrough to output " << new_name << endl;
    }
    return;

  } else {
    // the wire is defined, so store the pointer with its new name
    CoreIR::Wireable* temp_wire = get_wire(in_name, in_expr, indices);
    CoreIR::Wireable* in_wire = temp_wire;

    if (in_wire!=NULL && is_output(new_name)) {
      internal_assert(indices.empty()) << "output had indices selected\n";
      stream << "// " << new_name << " added as an output from " << in_name << "\n";
      def->connect(in_wire, self->sel("out"));

    } else if (in_wire) {
      add_wire(new_name, in_wire);
      stream << "// added/modified in wire_set: " << new_name << " = " << in_name << "\n";

    } else {
      stream << "// " << in_name << " not found (prob going to fail)\n";
    }
  }
}
}

void CreateCoreIRModule::visit_unaryop(Type t, Expr a, const char*  op_sym, string op_name) {
  string a_name = print_expr(a);
  string print_sym = op_sym;
  string out_var = print_assignment(t, print_sym + "(" + a_name + ")");

  // return if this variable is cached
  if (is_wire(out_var)) { return; }

  CoreIR::Wireable* a_wire = get_wire(a_name, a);
  if (a_wire != NULL) {
    string unaryop_name = op_name + a_name;
    CoreIR::Wireable* coreir_inst;

    // properly cast to generator or module
    internal_assert(gens.count(op_name) > 0) << op_name << " is not one of the names Halide recognizes\n";

    // check if it is a generator or module
    if (context->getNamespace("float")->hasGenerator(gens[op_name].substr(6))) {
      coreir_inst = def->addInstance(unaryop_name, gens[op_name],
                                     {{"exp_bits", CoreIR::Const::make(context,8)},
                                         {"frac_bits", CoreIR::Const::make(context,7)}});

    } else if (context->hasGenerator(gens[op_name])) {
      internal_assert(context->getGenerator(gens[op_name]));
      uint bw = inst_bitwidth(a.type().bits());
      coreir_inst = def->addInstance(unaryop_name, gens[op_name], {{"width", CoreIR::Const::make(context,bw)}});

    } else {
      internal_assert(context->getModule(gens[op_name]));
      coreir_inst = def->addInstance(unaryop_name, gens[op_name]);
    }

    def->connect(a_wire, coreir_inst->sel("in"));
    add_wire(out_var, coreir_inst->sel("out"));

  } else {
    // invalid operand
    out_var = "";
    print_assignment(t, print_sym + "(" + a_name + ")");
    if (a_wire == NULL) { stream << "// input 'a' was invalid!!" << endl; }
  }

  // print out operation
  if (is_input(a_name)) { stream << "// " << op_name <<"a: self.in "; } else { stream << "// " << op_name << "a: " << a_name << " "; }
  stream << "o: " << out_var << " with bitwidth:" << t.bits() << endl;
}


void CreateCoreIRModule::visit_binop(Type t, Expr a, Expr b, const char*  op_sym, string op_name) {
  string a_name = print_expr(a);
  string b_name = print_expr(b);
  string out_var = print_assignment(t, a_name + " " + op_sym + " " + b_name);

  // return if this variable is cached
  if (is_wire(out_var)) { return; }

  CoreIR::Wireable* a_wire = get_wire(a_name, a);
  CoreIR::Wireable* b_wire = get_wire(b_name, b);

  if (a_wire != NULL && b_wire != NULL) {
    internal_assert(a.type().bits() == b.type().bits()) << "function " << op_name << " with "
                                                        << a_name << "(" << a.type().bits() << "bits) and "
                                                        << b_name << "(" << b.type().bits() << "bits\n";
    uint bw = inst_bitwidth(a.type().bits());
    string binop_name = op_name + a_name + b_name + out_var;
    CoreIR::Wireable* coreir_inst;

    // properly cast to generator or module
    internal_assert(gens.count(op_name) > 0) << op_name << " is not one of the names Halide recognizes\n";
    if (context->getNamespace("float")->hasGenerator(gens[op_name].substr(6))) {
      coreir_inst = def->addInstance(binop_name, gens[op_name],
                                     {{"exp_bits", CoreIR::Const::make(context,8)},
                                         {"frac_bits", CoreIR::Const::make(context,7)}});
    } else if (context->hasGenerator(gens[op_name])) {
      coreir_inst = def->addInstance(binop_name, gens[op_name], {{"width", CoreIR::Const::make(context,bw)}});
    } else {
      coreir_inst = def->addInstance(binop_name, gens[op_name]);
    }

    def->connect(a_wire, coreir_inst->sel("in0"));
    def->connect(b_wire, coreir_inst->sel("in1"));
    add_wire(out_var, coreir_inst->sel("out"));

  } else {
    out_var = "";
    if (a_wire == NULL) { stream << "// input 'a' was invalid!!" << endl; }
    if (b_wire == NULL) { stream << "// input 'b' was invalid!!" << endl; }
  }

  if (is_input(a_name)) { stream << "// " << op_name <<"a: self.in "; } else { stream << "// " << op_name << "a: " << a_name << " "; }
  if (is_input(b_name)) { stream << "// " << op_name <<"b: self.in "; } else { stream << "// " << op_name << "b: " << b_name << " "; }
  stream << "o: " << out_var << " with obitwidth:" << t.bits() << endl;
}

void CreateCoreIRModule::visit_ternop(Type t, Expr a, Expr b, Expr c, const char*  op_sym1, const char* op_sym2, string op_name) {
  string a_name = print_expr(a);
  string b_name = print_expr(b);
  string c_name = print_expr(c);

  string out_var = print_assignment(t, a_name + " " + op_sym1 + " " + b_name + " " + op_sym2 + " " + c_name);
  // return if this variable is cached
  if (is_wire(out_var)) { return; }

  CoreIR::Wireable* a_wire = get_wire(a_name, a);
  CoreIR::Wireable* b_wire = get_wire(b_name, b);
  CoreIR::Wireable* c_wire = get_wire(c_name, c);

  if (a_wire != NULL && b_wire != NULL && c_wire != NULL) {
    internal_assert(b.type().bits() == c.type().bits());
    uint inst_bw = inst_bitwidth(b.type().bits());
    string ternop_name = op_name + a_name + b_name + c_name;
    CoreIR::Wireable* coreir_inst;

    // properly cast to generator or module

    internal_assert(gens.count(op_name) > 0) << op_name << " is not one of the names Halide recognizes\n";
    if (context->getNamespace("float")->hasGenerator(gens[op_name].substr(6))) {
      coreir_inst = def->addInstance(ternop_name, gens[op_name],
                                     {{"exp_bits", CoreIR::Const::make(context,8)},
                                         {"frac_bits", CoreIR::Const::make(context,7)}});
    } else if (context->hasGenerator(gens[op_name])) {
      coreir_inst = def->addInstance(ternop_name, gens[op_name], {{"width", CoreIR::Const::make(context,inst_bw)}});
    } else {
      coreir_inst = def->addInstance(ternop_name, gens[op_name]);
    }

    // wiring names are different for each operator
    if (op_name.compare("bitmux")==0 || op_name.compare("mux")==0) {
      def->connect(a_wire, coreir_inst->sel("sel"));
      def->connect(b_wire, coreir_inst->sel("in1"));
      def->connect(c_wire, coreir_inst->sel("in0"));
    } else {
      def->connect(a_wire, coreir_inst->sel("in0"));
      def->connect(b_wire, coreir_inst->sel("in1"));
      def->connect(c_wire, coreir_inst->sel("in2"));
    }
    add_wire(out_var, coreir_inst->sel("out"));

  } else {
    out_var = "";

    if (a_wire == NULL) { stream << "// input 'a' was invalid!!" << endl; }
    if (b_wire == NULL) { stream << "// input 'b' was invalid!!" << endl; }
    if (c_wire == NULL) { stream << "// input 'c' was invalid!!" << endl; }
  }

  if (is_input(a_name)) { stream << "// " << op_name <<"a: self.in "; } else { stream << "// " << op_name << "a: " << a_name << " "; }
  if (is_input(b_name)) { stream << "// " << op_name <<"b: self.in "; } else { stream << "// " << op_name << "b: " << b_name << " "; }
  if (is_input(c_name)) { stream << "// " << op_name <<"c: self.in "; } else { stream << "// " << op_name << "c: " << c_name << " "; }
  stream << "o: " << out_var << endl;

}

void CreateCoreIRModule::visit(const Mul *op) {
  internal_assert(op->a.type() == op->b.type());
  if (op->a.type().is_float()) {
    visit_binop(op->type, op->a, op->b, "f*", "fmul");
  } else {
    visit_binop(op->type, op->a, op->b, "*", "mul");
  }
}
void CreateCoreIRModule::visit(const Add *op) {
  internal_assert(op->a.type() == op->b.type());
  if (op->a.type().is_float()) {
    visit_binop(op->type, op->a, op->b, "f+", "fadd");
  } else {
    visit_binop(op->type, op->a, op->b, "+", "add");
  }
  // check if we can instantiate a MAD instead
  /*
    if (const Mul* mul = op->a.as<Mul>()) {
    visit_ternop(op->type, mul->a, mul->b, op->b, "*", "+", "MAD");
    } else if (const Mul* mul = op->b.as<Mul>()) {
    visit_ternop(op->type, mul->a, mul->b, op->a, "*", "+", "MAD");
    } else {
    visit_binop(op->type, op->a, op->b, "+", "add");
    }
  */

}
void CreateCoreIRModule::visit(const Sub *op) {
  internal_assert(op->a.type() == op->b.type());
  if (op->a.type().is_float()) {
    visit_binop(op->type, op->a, op->b, "f-", "fsub");
  } else {
    visit_binop(op->type, op->a, op->b, "-", "sub");
  }
}

void CreateCoreIRModule::visit(const Div *op) {
  int shift_amt;
  if (is_const_power_of_two_integer(op->b, &shift_amt)) {
    uint param_bitwidth = op->a.type().bits();
    Expr shift_expr = UIntImm::make(UInt(param_bitwidth), shift_amt);
    if (op->a.type().is_uint()) {
      internal_assert(op->b.type().is_uint());
      visit_binop(op->type, op->a, shift_expr, ">>", "lshr");
    } else {
      internal_assert(!op->b.type().is_uint());
      visit_binop(op->type, op->a, shift_expr, ">>", "ashr");
    }
  } else {
    if (op->a.type().is_float()) {
      visit_binop(op->type, op->a, op->b, "f/", "fdiv");

    } else if (op->a.type().is_uint()) {
      internal_assert(op->a.type().bits() == op->b.type().bits());
      visit_binop(op->type, op->a, op->b, "u/", "udiv");

    } else {
      internal_assert(!op->b.type().is_uint());
      internal_assert(op->a.type().bits() > 1);
      visit_binop(op->type, op->a, op->b, "s/", "sdiv");
    }
    //stream << "// divide is not fully supported" << endl;
    //user_warning << "WARNING: divide is not fully supported!!!!\n";
    //visit_binop(op->type, op->a, op->b, "/", "div");
  }
}

void CreateCoreIRModule::visit(const Mod *op) {
  int num_bits;
  if (is_const_power_of_two_integer(op->b, &num_bits)) {
    // equivalent to masking the bottom bits
    uint param_bitwidth = op->a.type().bits();
    uint mask = (1<<num_bits) - 1;
    Expr mask_expr = UIntImm::make(UInt(param_bitwidth), mask);
    visit_binop(op->type, op->a, mask_expr, "&", "and");

  } else if (op->type.is_int()) {
    //stream << "// mod is not fully supported" << endl;
    //print_expr(lower_euclidean_mod(op->a, op->b));
    visit_binop(op->type, op->a, op->b, "s%", "srem");
  } else {
    //stream << "// mod is not fully supported" << endl;
    visit_binop(op->type, op->a, op->b, "u%", "urem");
  }

}

void CreateCoreIRModule::visit(const And *op) {
  if (op->a.type().bits() == 1) {
    internal_assert(op->b.type().bits() == 1);
    visit_binop(op->type, op->a, op->b, "&&", "bitand");
  } else {
    visit_binop(op->type, op->a, op->b, "&&", "and");
  }
}
void CreateCoreIRModule::visit(const Or *op) {
  if (op->a.type().bits() == 1) {
    internal_assert(op->b.type().bits() == 1);
    visit_binop(op->type, op->a, op->b, "||", "bitor");
  } else {
    visit_binop(op->type, op->a, op->b, "||", "or");
  }
}

void CreateCoreIRModule::visit(const EQ *op) {
  internal_assert(op->a.type().bits() == op->b.type().bits());
  internal_assert(op->a.type() == op->b.type());
  if (op->a.type().is_float()) {
    visit_binop(op->type, op->a, op->b, "f==", "feq");
  } else {
    if (op->a.type().bits() == 1) {
      visit_binop(op->type, op->a, op->b, "~^", "bitxnor");
    } else {
      visit_binop(op->type, op->a, op->b, "==", "eq");
    }
  }
}
void CreateCoreIRModule::visit(const NE *op) {
  internal_assert(op->a.type().bits() == op->b.type().bits());
  internal_assert(op->a.type() == op->b.type());
  if (op->a.type().is_float()) {
    visit_binop(op->type, op->a, op->b, "f!=", "fneq");
  } else {
    if (op->a.type().bits() == 1) {
      visit_binop(op->type, op->a, op->b, "^", "bitxor");
    } else {
      visit_binop(op->type, op->a, op->b, "!=", "neq");
    }
  }
}

void CreateCoreIRModule::visit(const LT *op) {
  internal_assert(op->a.type() == op->b.type());

  if (op->a.type().is_float()) {
    visit_binop(op->type, op->a, op->b, "f<", "flt");

  } else if (op->a.type().is_uint()) {
    internal_assert(op->a.type().bits() == op->b.type().bits());
    if (op->a.type().bits() == 1) {
      Expr not_a = Not::make(op->a);
      visit_binop(op->type, not_a, op->b, "&&", "bitand");
      stream << "// created ~a * b for bitlt" << std::endl;
    } else {
      visit_binop(op->type, op->a, op->b, "<", "ult");
    }

  } else {
    internal_assert(!op->b.type().is_uint());
    internal_assert(op->a.type().bits() > 1);
    visit_binop(op->type, op->a, op->b, "s<", "slt");
  }
}
void CreateCoreIRModule::visit(const LE *op) {
  internal_assert(op->a.type() == op->b.type());

  if (op->a.type().is_float()) {
    visit_binop(op->type, op->a, op->b, "f<=", "fle");

  } else if (op->a.type().is_uint()) {
    internal_assert(op->a.type().bits() == op->b.type().bits());
    if (op->a.type().bits() == 1) {
      Expr not_a = Not::make(op->a);
      visit_binop(op->type, op->a, op->b, "||", "bitor");
      stream << "// created ~a + b for bitle" << std::endl;
    } else {
      visit_binop(op->type, op->a, op->b, "<=", "ule");
    }
  } else {
    internal_assert(!op->b.type().is_uint());
    internal_assert(op->a.type().bits() > 1);
    visit_binop(op->type, op->a, op->b, "s<=", "sle");
  }
}
void CreateCoreIRModule::visit(const GT *op) {
  internal_assert(op->a.type() == op->b.type());

  if (op->a.type().is_float()) {
    visit_binop(op->type, op->a, op->b, "f>", "fgt");

  } else if (op->a.type().is_uint()) {
    internal_assert(op->a.type().bits() == op->b.type().bits());
    if (op->a.type().bits() == 1) {
      Expr not_b = Not::make(op->b);
      visit_binop(op->type, op->a, not_b, "&&", "bitand");
      stream << "// created a * ~b for bitgt" << std::endl;
    } else {
      visit_binop(op->type, op->a, op->b, ">", "ugt");
    }
  } else {
    internal_assert(!op->b.type().is_uint());
    internal_assert(op->a.type().bits() > 1);
    visit_binop(op->type, op->a, op->b, "s>", "sgt");
  }
}
void CreateCoreIRModule::visit(const GE *op) {
  internal_assert(op->a.type() == op->b.type());

  if (op->a.type().is_float()) {
    visit_binop(op->type, op->a, op->b, "f>=", "fge");

  } else if (op->a.type().is_uint()) {
    internal_assert(op->a.type().bits() == op->b.type().bits());
    if (op->a.type().bits() == 1) {
      Expr not_b = Not::make(op->b);
      visit_binop(op->type, op->a, not_b, "||", "bitor");
      stream << "// created a + ~b for bitge" << std::endl;
    } else {
      visit_binop(op->type, op->a, op->b, ">=", "uge");
    }
  } else {
    internal_assert(!op->b.type().is_uint());
    internal_assert(op->a.type().bits() > 1);
    visit_binop(op->type, op->a, op->b, "s>=", "sge");
  }
}

void CreateCoreIRModule::visit(const Max *op) {
  internal_assert(op->a.type() == op->b.type());

  if (op->a.type().is_float()) {
    visit_binop(op->type, op->a, op->b, "<fmax>",  "fmax");
  } else if (op->type.is_uint()) {
    visit_binop(op->type, op->a, op->b, "<max>",  "umax");
  } else {
    visit_binop(op->type, op->a, op->b, "<smax>", "smax");
  }
}
void CreateCoreIRModule::visit(const Min *op) {
  internal_assert(op->a.type() == op->b.type());

  if (op->a.type().is_float()) {
    visit_binop(op->type, op->a, op->b, "<fmin>",  "fmin");
  } else if (op->type.is_uint()) {
    visit_binop(op->type, op->a, op->b, "<min>",  "umin");
  } else {
    visit_binop(op->type, op->a, op->b, "<smin>", "smin");
  }
}
void CreateCoreIRModule::visit(const Not *op) {
  // operator must have a one-bit/bool input
  assert(op->a.type().bits() == 1);
  visit_unaryop(op->type, op->a, "!", "bitnot");
}

void CreateCoreIRModule::visit(const Select *op) {
  internal_assert(op->true_value.type() == op->false_value.type());

  if (op->true_value.type().is_float()) {
    visit_ternop(op->type, op->condition, op->true_value, op->false_value, "f?",":", "fmux");
  } else if (op->type.bits() == 1) {
    // use a special op for bitwidth 1
    visit_ternop(op->type, op->condition, op->true_value, op->false_value, "?",":", "bitmux");
  } else {
    visit_ternop(op->type, op->condition, op->true_value, op->false_value, "?",":", "mux");
  }
}

void CreateCoreIRModule::visit(const Load *op) {
  Type t = op->type;
  bool type_cast_needed =
    !allocations.contains(op->name) ||
    allocations.get(op->name).type != t;

  string id_index = print_expr(op->index);
  string name = print_name(op->name);
  ostringstream rhs;
  if (type_cast_needed) {
    rhs << "(("
        << print_type(op->type)
        << " *)"
        << name
        << ")";
  } else {
    rhs << name;
  }
  rhs << "["
      << id_index
      << "][load]";

  string out_var = print_assignment(op->type, rhs.str());

  // generate coreir
  if (is_const(op->index)) {
    string in_var = name + "_" + id_index;
    rename_wire(out_var, in_var, Expr());

    //} else if (is_defined(op->name) && hw_def_set[op->name]->gen==gens["rom2"]) {
  } else if (is_defined(op->name) && hw_def_set[op->name]->gen==gens["lake"]) {
    stream << "loading from rom " << op->name << " with gen " << hw_def_set[op->name]->gen << std::endl;

    std::shared_ptr<CoreIR_Inst_Args> inst_args = hw_def_set[op->name];
    string inst_name = unique_name(inst_args->name);
    CoreIR::Wireable* inst = def->addInstance(inst_name, inst_args->gen, inst_args->args, inst_args->genargs);
    add_wire(out_var, inst->sel(inst_args->selname));

    // attach the read address
    CoreIR::Wireable* raddr_wire = get_wire(id_index, op->index);
    //def->connect(raddr_wire, inst->sel("raddr")); // for a rom2
    def->connect(raddr_wire, inst->sel("data_in_0"));
    //attach a read enable
    //CoreIR::Wireable* rom_ren = def->addInstance(inst_name + "_ren", gens["bitconst"], {{"value", CoreIR::Const::make(context,true)}});
    //def->connect(rom_ren->sel("out"), inst->sel("ren"));

  } else if (is_defined(op->name) && hw_def_set[op->name]->gen==gens["ram2"]) {
    stream << "loading from sram " << name << std::endl;

    add_wire(out_var, get_wire(name, Expr()));

    // attach the read address
    CoreIR::Wireable* raddr_wire = get_wire(id_index, op->index);
    auto ram_raddr = get_wire(name+"_raddr", Expr());
    def->disconnect(ram_raddr);
    def->connect(raddr_wire, ram_raddr);

  } else { // variable load: use a mux for where data originates
    internal_error << "variable load not supported yet";
  }

}

void CreateCoreIRModule::visit(const Cast *op) {
  stream << "[cast]";
  string in_var = print_expr(op->value);
  string out_var = print_assignment(op->type, "(" + print_type(op->type) + ")(" + in_var + ")");

  // casting from 1 to 16 bits
  if (op->type.bits() > 1 && op->value.type().bits() == 1) {
    stream << "// casting from 1 to 16 bits" << endl;
    Expr one_uint16 = UIntImm::make(UInt(16), 1);
    Expr zero_uint16 = UIntImm::make(UInt(16), 0);
    visit_ternop(op->type, op->value, one_uint16, zero_uint16, "?", ":", "mux");

  // casting from 16 to 1 bit
  } else if (op->type.bits() == 1 && op->value.type().bits() > 1) {
    stream << "// casting from 16 to 1 bit" << endl;
    Expr zero_uint16 = UIntImm::make(UInt(op->value.type().bits()), 0);
    visit_binop(op->type, op->value, zero_uint16, "!=", "neq");

  } else if (!is_iconst(in_var)) {
    // only add to list, don't duplicate constants
    stream << "renaming " << in_var << " to " << out_var << std::endl;
    rename_wire(out_var, in_var, op->value);
  }
}

void CreateCoreIRModule::visit(const Call *op) {
  // add bitand, bitor, bitxor, bitnot
  if (op->is_intrinsic(Call::bitwise_and)) {
    internal_assert(op->args.size() == 2);
    Expr a = op->args[0];
    Expr b = op->args[1];
    if (op->type.bits() == 1) {
      // use a special op for bitwidth 1
      visit_binop(op->type, a, b, "&&", "bitand");
    } else {
      visit_binop(op->type, a, b, "&", "and");
    }
  } else if (op->is_intrinsic(Call::bitwise_or)) {
    internal_assert(op->args.size() == 2);
    Expr a = op->args[0];
    Expr b = op->args[1];
    if (op->type.bits() == 1) {
      // use a special op for bitwidth 1
      visit_binop(op->type, a, b, "||", "bitor");
    } else {
      visit_binop(op->type, a, b, "|", "or");
    }
  } else if (op->is_intrinsic(Call::bitwise_xor)) {
    internal_assert(op->args.size() == 2);
    Expr a = op->args[0];
    Expr b = op->args[1];
    if (op->type.bits() == 1) {
      // use a special op for bitwidth 1
      visit_binop(op->type, a, b, "^", "bitxor");
    } else {
      visit_binop(op->type, a, b, "^", "xor");
    }
  } else if (op->is_intrinsic(Call::bitwise_not)) {
    internal_assert(op->args.size() == 1);
    Expr a = op->args[0];
    if (op->type.bits() == 1) {
      // use a special op for bitwidth 1
      visit_unaryop(op->type, a, "!", "bitnot");
    } else {
      visit_unaryop(op->type, a, "~", "not");
    }

  } else if (op->is_intrinsic(Call::shift_left)) {
    internal_assert(op->args.size() == 2);
    Expr a = op->args[0];
    Expr b = op->args[1];
    stream << "[shift left] ";
    visit_binop(op->type, a, b, "<<", "shl");
  } else if (op->is_intrinsic(Call::shift_right)) {
    internal_assert(op->args.size() == 2);
    Expr a = op->args[0];
    Expr b = op->args[1];
    stream << "[shift right] ";
    if (a.type().is_uint()) {
      internal_assert(b.type().is_uint());
      visit_binop(op->type, a, b, ">>", "lshr");
    } else {
      internal_assert(!b.type().is_uint());
      visit_binop(op->type, a, b, ">>", "ashr");
    }
  } else if (op->is_intrinsic(Call::abs)) {
    internal_assert(op->args.size() == 1);
    Expr a = op->args[0];
    stream << "[abs] ";
    visit_unaryop(op->type, a, "abs", "abs");
  } else if (op->is_intrinsic(Call::absd)) {
    internal_assert(op->args.size() == 2);
    Expr a = op->args[0];
    Expr b = op->args[1];
    stream << "[absd] ";
    visit_binop(op->type, a, b, "|-|", "absd");

  } else if (op->is_intrinsic(Call::reinterpret)) {
    string in_var = print_expr(op->args[0]);
    print_reinterpret(op->type, op->args[0]);

    stream << "// reinterpreting " << op->args[0] << " as " << in_var << endl;

    // generate coreir: expecting to find the expr is a constant
    rename_wire(in_var, in_var, op->args[0]);


  } else if (op->name == "floor_f32") {
    internal_assert(op->args.size() == 1);
    Expr a = op->args[0];
    visit_unaryop(op->type, a, "floor", "fflr");
  } else if (op->name == "round_f32") {
    internal_assert(op->args.size() == 1);
    Expr a = op->args[0];
    visit_unaryop(op->type, a, "round", "frnd");
  } else if (op->name == "ceil_f32") {
    internal_assert(op->args.size() == 1);
    Expr a = op->args[0];
    visit_unaryop(op->type, a, "ceil", "fceil");

    
  } else if (op->name == "exp_f32") {
    internal_assert(op->args.size() == 1);
    Expr a = op->args[0];
    visit_unaryop(op->type, a, "exp", "fexp");

  } else if (op->name == "sqrt_f32") {
    internal_assert(op->args.size() == 1);
    Expr a = op->args[0];
    visit_unaryop(op->type, a, "sqrt", "fsqrt");

  } else if (op->name == "log_f32") {
    internal_assert(op->args.size() == 1);
    Expr a = op->args[0];
    visit_unaryop(op->type, a, "ln", "fln");

  } else if (op->name == "sin_f32") {
    internal_assert(op->args.size() == 1);
    Expr a = op->args[0];
    visit_unaryop(op->type, a, "sin", "fsin");
  } else if (op->name == "cos_f32") {
    internal_assert(op->args.size() == 1);
    Expr a = op->args[0];
    visit_unaryop(op->type, a, "cos", "fcos");
  } else if (op->name == "tan_f32") {
    internal_assert(op->args.size() == 1);
    Expr a = op->args[0];
    visit_unaryop(op->type, a, "tan", "ftan");

  } else if (op->name == "asin_f32") {
    internal_assert(op->args.size() == 1);
    Expr a = op->args[0];
    visit_unaryop(op->type, a, "asin", "fasin");
  } else if (op->name == "acos_f32") {
    internal_assert(op->args.size() == 1);
    Expr a = op->args[0];
    visit_unaryop(op->type, a, "acos", "facos");
  } else if (op->name == "atan_f32") {
    internal_assert(op->args.size() == 1);
    Expr a = op->args[0];
    visit_unaryop(op->type, a, "atan", "fatan");
  } else if (op->name == "atan2_f32") {
    internal_assert(op->args.size() == 2);
    Expr a = op->args[0];
    Expr b = op->args[1];
    visit_binop(op->type, a, b, "atan2", "fatan2");


// This intrisic was removed:
//  } else if (op->is_intrinsic(Call::address_of)) {
//    user_warning << "ignoring address_of\n";
//    stream       << "ignoring address_of" << endl;
//

  } else if (op->name == "linebuffer") {
    stream << "shouldn't be seeing " << op->name << endl;
    cout << "shouldn't be seeing " << op->name << endl;
    
  } else if (op->name == "hwbuffer") {
    stream << "shouldn't be seeing " << op->name << endl;
    cout << "shouldn't be seeing " << op->name << endl;

  } else if (op->name == "write_stream") {
    stream << "shouldn't be seeing " << op->name << endl;
    cout << "shouldn't be seeing " << op->name << endl;

  } else if (op->name == "read_stream") {
    stream << "shouldn't be seeing " << op->name << endl;
    cout << "shouldn't be seeing " << op->name << endl;

  } else if (ends_with(op->name, ".stencil") ||
             ends_with(op->name, ".stencil_update")) {

    if (is_defined(op->name) && hw_def_set[op->name]->gen==gens["rom2"]) {
      internal_assert(op->args.size() == 1);
      string id_index = print_expr(op->args[0]);
      auto name = op->name;
      ostringstream rhs;
      
      rhs << name;
      rhs << "["
          << id_index
          << "][call]";

      string out_var = print_assignment(op->type, rhs.str());
      if (is_wire(out_var)) { return; }

      stream << "// loading from rom " << name << " with gen " << hw_def_set[name]->gen << std::endl;

      std::shared_ptr<CoreIR_Inst_Args> inst_args = hw_def_set[name];
      string inst_name = unique_name("curve");
      CoreIR::Wireable* inst = def->addInstance(inst_name, inst_args->gen, inst_args->args, inst_args->genargs);
      add_wire(out_var, inst->sel(inst_args->selname));

      // attach the read address
      CoreIR::Wireable* raddr_wire = get_wire(id_index, op->args[0]);
      def->connect(raddr_wire, inst->sel("raddr"));
      // attach a read enable
      CoreIR::Wireable* rom_ren = def->addInstance(inst_name + "_ren", gens["bitconst"], {{"value", CoreIR::Const::make(context,true)}});
      def->connect(rom_ren->sel("out"), inst->sel("ren"));
      
    } else {
      stream << "shouldn't be seeing " << op->name << endl;
      cout << "shouldn't be seeing " << op->name << endl;
    }

  } else if (op->name == "dispatch_stream") {
    stream << "shouldn't be seeing " << op->name << endl;
    cout << "shouldn't be seeing " << op->name << endl;

  } else {
    stream << "couldn't find op named " << op->name << endl;
    cout << "couldn't find op named " << op->name << endl;
    //CodeGen_C::visit(op);
    IRVisitor::visit(op);
  }
}

class ROMInit : public IRVisitor {
  vector<int> strides;
  using IRVisitor::visit;

  bool is_const(const Expr e) {
    if (e.as<IntImm>() || e.as<UIntImm>()) {
      return true;
    } else {
      return false;
    }
  }

  void visit(const Store *op) {
    if (op->name == allocname) {
      auto value_expr = op->value;
      auto index_expr = op->index;
      //cout << "store: " << Stmt(op) << std::endl;
      //internal_assert(is_const(value_expr) && is_const(index_expr));
      if (is_const(value_expr) && is_const(index_expr)) {
        int index = id_const_value(index_expr);
        int value = id_const_value(value_expr);
        init_values["init"][index] = value;
        //init_values[index] = value;
      }
    }
    IRVisitor::visit(op);
  }

  void visit(const Provide *op) {
    if (op->name == allocname) {
      //internal_assert(op->args.size() == 1);
      internal_assert(op->values.size() == 1);
      internal_assert(op->args.size() == strides.size())
        << "args size=" << op->args.size() << " while strides size=" << strides.size() << "\n";
      auto value_expr = op->values[0];
      //auto index_expr = op->args[0];
      int index = 0;
      for (size_t i=0; i<strides.size(); ++i) {
        internal_assert(is_const(op->args[i]));
        index += id_const_value(op->args[i]) * strides[i];
      }
      
      //cout << "store: " << Stmt(op) << std::endl;
      //internal_assert(is_const(value_expr) && is_const(index_expr));
      internal_assert(is_const(value_expr));

      //int index = id_const_value(index_expr);
      int value = id_const_value(value_expr);

      init_values["init"][index] = value;
      //init_values[index] = value;
    }
    IRVisitor::visit(op);
  }

public:
  nlohmann::json init_values;
  string allocname;
  ROMInit(string allocname, vector<int> strides) : strides(strides), allocname(allocname) {}
};

// returns a map with all the initialization values for a rom
nlohmann::json rom_init(Stmt s, vector<int> strides, string allocname) {
  ROMInit rom_init(allocname, strides);
  s.accept(&rom_init);
  return rom_init.init_values;
}

CoreIR_Inst_Args rom_to_coreir(string alloc_name, vector<int> rom_size, Stmt body, CoreIR::Context* context) {
    CoreIR_Inst_Args rom_args;
    rom_args.ref_name = alloc_name;
    rom_args.name = "rom_" + alloc_name;

    int total_size = 1;
    vector<int> stride(rom_size.size());
    //for (auto len : size) {
    for (size_t i=0; i<rom_size.size(); ++i) {
      if (i==0) {
        stride[i] = 1;
      } else {
        stride[i] = stride[i-1] * rom_size[i-1];
      }
      total_size *= rom_size[i];
    }

    auto gens = coreir_generators(context);

    // set initial values for rom
    nlohmann::json jdata = rom_init(body, stride, alloc_name);
    jdata["mode"] = "sram";

    rom_args.gen = gens["lake"];
    rom_args.genargs = {{"mode",CoreIR::Const::make(context,"lake")},
                     {"config",CoreIR::Const::make(context, jdata)}};
    CoreIR::Values modparams = {{"width",CoreIR::Const::make(context,16)},
                                {"num_inputs",CoreIR::Const::make(context,1)},
                                {"num_outputs",CoreIR::Const::make(context,1)}};
    rom_args.selname = "data_out_0";
    rom_args.args = modparams;

    //rom_args.gen = gens["rom2"];
    ////jdata["init"][0] = 0;
    //rom_args.args = {{"width",CoreIR::Const::make(context,16)},
    //                 {"depth",CoreIR::Const::make(context,total_size)}};
    //CoreIR::Values modparams = {{"init", CoreIR::Const::make(context, jdata)}};
    //rom_args.selname = "rdata";
    rom_args.genargs = modparams;
    
    return rom_args;
}

CoreIR::Type* interface_to_type(CoreIR_Interface iface, CoreIR::Context* context) {
  // add the output
  CoreIR::Type* out_type = iface.output.bitwidth == 1 ?
    context->Bit() :
    context->Bit()->Arr(inst_bitwidth(iface.output.bitwidth));
  CoreIR::RecordParams recordparams = {
    {"out_" + iface.output.name, out_type}
  };

  // add each input
  for (size_t i=0; i<iface.inputs.size(); ++i) {
    auto input_bundle = iface.inputs[i];
    
    // calculate the bundle bitwidth
    //uint bundle_bitwidth = 0;
    //for (auto input : input_bundle.ports) {
    //  bundle_bitwidth += input.bitwidth;
    //}
    //CoreIR::Type* bundle_type = context->BitIn()->Arr(bundle_bitwidth);
    //string bundle_name = "in" + std::to_string(i) + "_" + input_bundle.name;
    //recordparams.push_back({bundle_name, bundle_type});

    // Check for consistent bitwidth within a bundle
    size_t num_bundles = input_bundle.ports.size();
    internal_assert(num_bundles > 0);
    uint bitwidth = input_bundle.ports.at(0).bitwidth;
    for (auto input : input_bundle.ports) {
      internal_assert(bitwidth == input.bitwidth);
    }

    // Create an input array for each bundle
    CoreIR::Type* bundle_type;
    if (false) {//(num_bundles == 1) {
      bundle_type = context->BitIn()->Arr(bitwidth);
    } else {
      bundle_type = bitwidth == 1 ?
        context->BitIn()->Arr(num_bundles) :
        context->BitIn()->Arr(inst_bitwidth(bitwidth))->Arr(num_bundles);
    }
    string bundle_name = "in" + std::to_string(i) + "_" + input_bundle.name;
    recordparams.push_back({bundle_name, bundle_type});
  }

  // Add each index as an input
  for (auto index : iface.indices) {
    string index_name = index.name;
    CoreIR::Type* index_type = context->BitIn()->Arr(inst_bitwidth(index.bitwidth));
    recordparams.push_back({index_name, index_type});
  }

  return context->Record(recordparams);
}


void convert_compute_to_coreir(Expr e, CoreIR_Interface iface, CoreIR::Context* context) {
  CoreIR::Type* compute_type = interface_to_type(iface, context);

  // define the module
  auto global_ns = context->getNamespace("global");
  auto design = global_ns->newModuleDecl(iface.name, compute_type);
  auto def = design->newModuleDef();

  // convert halide expr into coreir module definition and store in context
  add_coreir_compute(e, def, iface, context);
  design->setDef(def);
}

void convert_compute_to_coreir(Expr e, CoreIR_Interface iface,
                               vector<CoreIR_Inst_Args> coreir_insts, CoreIR::Context* context) {
  CoreIR::Type* compute_type = interface_to_type(iface, context);

  // define the module
  auto global_ns = context->getNamespace("global");
  auto design = global_ns->newModuleDecl(iface.name, compute_type);
  auto def = design->newModuleDef();

  // convert halide expr into coreir module definition and store in context
  add_coreir_compute(e, def, iface, coreir_insts, context);
  design->setDef(def);
}


}
}
