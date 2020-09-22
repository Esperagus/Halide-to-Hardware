#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <utility>

#include "CodeGen_Clockwork_Target.h"
#include "CodeGen_Internal.h"
#include "CoreIRCompute.h"
#include "HWBufferUtils.h"
#include "HWBufferSimplifications.h"
#include "IR.h"
#include "IREquality.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "Lerp.h"
#include "Param.h"
#include "Simplify.h"
#include "Substitute.h"
#include "Var.h"

namespace Halide {
namespace Internal {

using std::ostream;
using std::endl;
using std::string;
using std::vector;
using std::map;
using std::set;
using std::ostringstream;
using std::ofstream;

namespace {

class ContainForLoop : public IRVisitor {
    using IRVisitor::visit;
    void visit(const For *op) {
        found = true;
        return;
    }

public:
    bool found;

    ContainForLoop() : found(false) {}
};

bool contain_for_loop(Stmt s) {
    ContainForLoop cfl;
    s.accept(&cfl);
    return cfl.found;
}

class ContainsCall : public IRVisitor {
    using IRVisitor::visit;
    void visit(const Call *op) {
      if (match_any || calls.count(op->name) > 0) {
        found_calls.emplace_back(op->name);
      }
      IRVisitor::visit(op);
    }

    void visit(const Load *op) {
      if (match_any || calls.count(op->name) > 0) {
        found_calls.emplace_back(op->name);
      }
      IRVisitor::visit(op);
    }
  
public:
  vector<string> found_calls;
  set<string> calls;
  bool match_any;

  ContainsCall(set<string> calls, bool match) : calls(calls), match_any(match) {}
};

vector<string> contains_call(Expr e, set<string> calls) {
    ContainsCall cc(calls, false);
    e.accept(&cc);
    return cc.found_calls;
}

vector<string> contains_call(Expr e) {
    ContainsCall cc({}, true);
    e.accept(&cc);
    return cc.found_calls;
}

bool contains_call(vector<Expr> args, vector<bool>& calls_found) {
  bool one_found = false;
    for (auto& e : args) {
      ContainsCall cc({}, true);
      e.accept(&cc);
      bool found = cc.found_calls.size() > 0;
      calls_found.push_back(found);
      one_found = one_found || found;
    }
    return one_found;
}


string printname(const string &name) {
    ostringstream oss;

    // Prefix an underscore to avoid reserved words (e.g. a variable named "while")
    size_t start;
    //for (start=0; start<name.size(); ++start) {
    //  if (isalnum(name[start])) {
    //    break;
    //  }
    //}
    while (start<name.size() && !isalnum(name[start])) {
      start++;
    }

    internal_assert(start < name.size()) << "what to do about " << name << "\n";

    for (size_t i = start; i < name.size(); i++) {
      // vivado HLS compiler doesn't like '__'
        if (!isalnum(name[i])) {
            oss << "_";
        }
        else oss << name[i];
    }
    return oss.str();
}

string type_to_c_type(Type type) {
  ostringstream oss;
  
  if (type.is_float()) {
    if (type.bits() == 32) {
      oss << "float";
    } else if (type.bits() == 64) {
      oss << "double";
    } else if (type.bits() == 16 && !type.is_bfloat()) {
      oss << "float16_t";
    } else if (type.bits() == 16 && type.is_bfloat()) {
      //oss << "bfloat16_t";
      oss << "uint16_t";
    } else {
      user_error << "Can't represent a float with this many bits in C: " << type << "\n";
    }
    if (type.is_vector()) {
      oss << type.lanes();
    }
    
  } else {
    switch (type.bits()) {
    case 1:
      // bool vectors are always emitted as uint8 in the C++ backend
      if (type.is_vector()) {
        oss << "uint8x" << type.lanes() << "_t";
      } else {
        oss << "bool";
      }
      break;
    case 8: case 16: case 32: case 64:
      if (type.is_uint()) {
        oss << 'u';
      }
      oss << "int" << type.bits();
      if (type.is_vector()) {
        oss << "x" << type.lanes();
      }
      oss << "_t";
      break;
    default:
      user_error << "Can't represent an integer with this many bits in C: " << type << "\n";
    }
  }

  return oss.str();
}

string removedots(const string &name) {
    ostringstream oss;
    
    for (size_t i = 0; i < name.size(); i++) {
        if (name[i] == '.') {
          oss << "_";
        } else if (name[i] == '$') {
          oss << "_";
        } else {
          oss << name[i];
        }
    }
    return oss.str();
}

class RemoveCallIndexes : public IRMutator {
    using IRMutator::visit;
    Expr visit(const Call *op) {
      //return Call::make(op->type, op->name, {}, op->call_type, op->func, op->value_index, op->image, op->param);
      return Variable::make(op->type, printname(op->name));
    }
    Expr visit(const Load *op) {
      //return Call::make(op->type, op->name, {}, op->call_type, op->func, op->value_index, op->image, op->param);
      return Variable::make(op->type, printname(op->name));
    }

public:
    RemoveCallIndexes() {}
};

Expr remove_call_args(Expr e) {
    RemoveCallIndexes rci;
    return rci.mutate(e);
}

std::string strip_stream(std::string input) {
  std::string output = input;
  if (ends_with(input, "_stencil_update_stream")) {
    output = input.substr(0, input.find("_stencil_update_stream"));
  } else if (ends_with(input, "_stencil_stream")) {
    output = input.substr(0, input.find("_stencil_stream"));
  }
  return output;
}

class RenameAllocation : public IRMutator {
    const string &orig_name;
    const string &new_name;

    using IRMutator::visit;

    Expr visit(const Load *op) {
        if (op->name == orig_name) {
            Expr index = mutate(op->index);
            return Load::make(op->type, new_name, index, op->image, op->param, op->predicate, ModulusRemainder());
        } else {
            return IRMutator::visit(op);
        }
    }

    Stmt visit(const Store *op) {
        if (op->name == orig_name) {
            Expr value = mutate(op->value);
            Expr index = mutate(op->index);
            return Store::make(new_name, value, index, op->param, op->predicate, ModulusRemainder());
        } else {
            return IRMutator::visit(op);
        }
    }

    Stmt visit(const Free *op) {
        if (op->name == orig_name) {
            return Free::make(new_name);
        } else {
            return IRMutator::visit(op);
        }
    }

    Stmt visit(const ProducerConsumer *op) {
      if (op->name == orig_name) {
        auto body = mutate(op->body);
        return ProducerConsumer::make(new_name, op->is_producer, body);
      } else {
        return IRMutator::visit(op);
      }
    }

public:
    RenameAllocation(const string &o, const string &n)
        : orig_name(o), new_name(n) {}
};

class RenameRealize : public IRMutator {
    const string &orig_name;
    const string &new_name;

    using IRMutator::visit;

    Expr visit(const Call *op) {
        if (op->name == orig_name ) {
          vector<Expr> new_args(op->args.size());

          // Mutate the args
          for (size_t i = 0; i < op->args.size(); i++) {
            const Expr &old_arg = op->args[i];
            Expr new_arg = mutate(old_arg);
            new_args[i] = std::move(new_arg);
          }
          return Call::make(op->type, new_name, new_args, op->call_type, op->func,
                            op->value_index, op->image, op->param);
        } else {
            return IRMutator::visit(op);
        }
    }

  Stmt visit(const Provide *op) {
    if (op->name == orig_name) {
      vector<Expr> new_args(op->args.size());
      vector<Expr> new_values(op->values.size());

      // Mutate the args
      for (size_t i = 0; i < op->args.size(); i++) {
        const Expr &old_arg = op->args[i];
        Expr new_arg = mutate(old_arg);
        new_args[i] = new_arg;
      }

      for (size_t i = 0; i < op->values.size(); i++) {
        const Expr &old_value = op->values[i];
        Expr new_value = mutate(old_value);
        new_values[i] = new_value;
      }

      return Provide::make(new_name, new_values, new_args);
    } else {
      return IRMutator::visit(op);
    }
  }

  Stmt visit(const Realize *op) {
    if (op->name == orig_name) {
      Stmt body = mutate(op->body);
      Expr condition = mutate(op->condition);
      return Realize::make(new_name, op->types, op->memory_type, op->bounds,
                           std::move(condition), std::move(body));
    } else {
      return IRMutator::visit(op);
    }
}



public:
    RenameRealize(const string &o, const string &n)
        : orig_name(o), new_name(n) {}
};

struct OutputInfoExtract : public IRVisitor {
    std::vector<std::pair<std::string, Expr>> info;

    OutputInfoExtract(const std::string &output_buf_name)
        : output_name(output_buf_name)
    {}

protected:
    using IRVisitor::visit;

    void visit(const For *op) {
        std::string loop_name = printname(op->name);
        Expr loop_max = simplify(op->min + op->extent);
        loop_info.push(loop_name, loop_max);
        IRVisitor::visit(op);
    }

    void visit(const Provide *op) {
        if(printname(op->name) == output_name) {
            for(Expr e : op->args) {
                const Variable *v = e.as<Variable>();
                internal_assert(v && loop_info.contains(printname(v->name)));
                std::string loop_name = printname(v->name);
                info.push_back({loop_name, loop_info.get(loop_name)});
            }
        } else {
            IRVisitor::visit(op);
        }
    }

private:
    std::string output_name;
    Scope<Expr> loop_info;
};

}

CodeGen_Clockwork_Target::CodeGen_Clockwork_Target(const string &name, const Target& target)
  : target_name(name),
    hdrc(hdr_stream, target, CodeGen_Clockwork_C::CPlusPlusHeader), 
    srcc(src_stream, target, CodeGen_Clockwork_C::CPlusPlusImplementation),
    clkc(clk_stream, target, CodeGen_Clockwork_C::CPlusPlusImplementation) { clkc.is_clockwork = true; }
    //clkc(std::cout, target, CodeGen_Clockwork_C::CPlusPlusImplementation) { clkc.is_clockwork = true; }


void print_clockwork_codegen(string appname, ofstream& stream);
void print_clockwork_execution_header(string appname, ofstream& stream);
void print_clockwork_execution_cpp(string appname, const vector<HW_Arg>& closure_args, ofstream& stream);

CodeGen_Clockwork_Target::~CodeGen_Clockwork_Target() {
    hdr_stream << "#endif\n";
    clkc.memory_stream << endl
                       << "  return prg;" << endl
                       << "}";

    std::cout << "outputting clockwork target named " << target_name << std::endl;

    // write the header and the source streams into files
    //string src_name = output_base_path + target_name + ".cpp";
    //string hdr_name = output_base_path + target_name + ".h";
    string clk_debug_name = output_base_path + target_name + "_debug.cpp";
    string clk_memory_name = output_base_path + target_name + "_memory.cpp";
    string clk_memory_header_name = output_base_path + target_name + "_memory.h";
    string clk_compute_name = output_base_path + target_name + "_compute.h";

    //ofstream src_file(src_name.c_str());
    //ofstream hdr_file(hdr_name.c_str());
    ofstream clk_debug_file(clk_debug_name.c_str());
    ofstream clk_memory_file(clk_memory_name.c_str());
    ofstream clk_memory_header_file(clk_memory_header_name.c_str());
    ofstream clk_compute_file(clk_compute_name.c_str());

    //src_file << src_stream.str() << endl;
    //hdr_file << hdr_stream.str() << endl;
    clk_debug_file << clk_stream.str() << endl;
    clk_memory_file << clkc.memory_oss.str() << endl;
    clk_memory_header_file << "prog " << target_name << "();" << std::endl;
    clk_compute_file << clkc.compute_oss.str() << endl;
    
    //src_file.close();
    //hdr_file.close();
    clk_debug_file.close();
    clk_memory_file.close();
    clk_memory_header_file.close();
    clk_compute_file.close();

    saveToFile(clkc.context->getGlobal(), output_base_path + target_name + "_compute.json", NULL);

    string clk_codegen_name = output_base_path + "clockwork_codegen.cpp";
    string clk_exec_h_name = output_base_path + "clockwork_testscript.h";
    string clk_exec_cpp_name = output_base_path + "clockwork_testscript.cpp";
    
    ofstream clk_codegen_file(clk_codegen_name.c_str());
    ofstream clk_exec_h_file(clk_exec_h_name.c_str());
    ofstream clk_exec_cpp_file(clk_exec_cpp_name.c_str());

    print_clockwork_codegen(target_name, clk_codegen_file);
    std::cout << "printed codegen" << std::endl;
    print_clockwork_execution_header(target_name, clk_exec_h_file);
    std::cout << "printed execution header" << std::endl;
    print_clockwork_execution_cpp(target_name, closure_args, clk_exec_cpp_file);
    std::cout << "printed execution cpp" << std::endl;

    clk_codegen_file.close();
    clk_exec_h_file.close();
    clk_exec_cpp_file.close();
}

//CodeGen_Clockwork_Target::CodeGen_Clockwork_C(std::ostream &s, Target target, OutputKind output_kind) :
//CodeGen_Clockwork_Base(s, target, output_kind), is_clockwork(false), memory_stream( { }


namespace {
const string clockwork_header_includes =
    "#include <assert.h>\n"
    "#include <stdio.h>\n"
    "#include <stdlib.h>\n"
    "#include <hls_stream.h>\n"
    "#include \"Stencil.h\"\n";

const string local_halide_helpers =
    "\n"
    "template <typename T>\n"
    "static inline T halide_cpp_min(const T&a, const T&b) { return (a < b) ? a : b; }\n\n"
    "template <typename T>\n"
    "static inline T halide_cpp_max(const T&a, const T&b) { return (a > b) ? a : b; }\n\n";
}

void CodeGen_Clockwork_Target::init_module() {
    debug(1) << "CodeGen_Clockwork_Target::init_module\n";

    // wipe the internal streams
    hdr_stream.str("");
    hdr_stream.clear();
    src_stream.str("");
    src_stream.clear();
    clk_stream.str("");
    clk_stream.clear();

    // initialize the header file
    string module_name = "HALIDE_CODEGEN_CLOCKWORK_TARGET_" + target_name + "_H";
    std::transform(module_name.begin(), module_name.end(), module_name.begin(), toupper);
    hdr_stream << "#ifndef " << module_name << '\n'
               << "#define " << module_name << "\n\n"
               << clockwork_header_includes << '\n';

    // initialize the source file
    src_stream << "#include \"" << target_name << ".h\"\n\n"
               << "#include \"Linebuffer.h\"\n"
               << "#include \"halide_math.h\"\n"
               << local_halide_helpers;

    // initialize the clockwork compute file
    clkc.compute_stream << "#pragma once" << endl
                        << "#include \"hw_classes.h\"" << endl
                        << "#include \"clockwork_standard_compute_units.h\"" << endl << endl;

    // initialize the clockwork memory file
    clkc.memory_stream << "#include \"app.h\"\n"
                       << "#include \"ubuffer.h\"\n"
                       << "#include \"codegen.h\"\n"
                       << "#include \"prog.h\"\n\n";

    ofstream compute_debug_file("bin/clockwork_compute_debug.cpp");
    compute_debug_file.close();

    clkc.context = CoreIR::newContext();
}

void CodeGen_Clockwork_Target::add_kernel(Stmt s,
                                          const string &name,
                                          const vector<HW_Arg> &args) {
    debug(1) << "CodeGen_Clockwork_Target::add_kernel " << name << "\n";

    closure_args = args;
    //hdrc.add_kernel(s, name, args);
    //srcc.add_kernel(s, name, args);
    clkc.add_kernel(s, target_name, args);
}

void CodeGen_Clockwork_Target::dump() {
    std::cerr << src_stream.str() << std::endl;
}


string CodeGen_Clockwork_Target::CodeGen_Clockwork_C::print_stencil_pragma(const string &name) {
    ostringstream oss;
    internal_assert(stencils.contains(name));
    HW_Stencil_Type stype = stencils.get(name);
    if (stype.type == HW_Stencil_Type::StencilContainerType::Stream ||
        stype.type == HW_Stencil_Type::StencilContainerType::AxiStream) {
        oss << "#pragma HLS STREAM variable=" << printname(name) << " depth=" << stype.depth << "\n";
        if (stype.depth <= 100) {
            // use shift register implementation when the FIFO is shallow
            oss << "#pragma HLS RESOURCE variable=" << printname(name) << " core=FIFO_SRL\n\n";
        }
    } else if (stype.type == HW_Stencil_Type::StencilContainerType::Stencil) {
        oss << "#pragma HLS ARRAY_PARTITION variable=" << printname(name) << ".value complete dim=0\n\n";
    } else {
        internal_error;
    }
    return oss.str();
}


void CodeGen_Clockwork_Target::CodeGen_Clockwork_C::add_kernel(Stmt stmt,
                                                               const string &name,
                                                               const vector<HW_Arg> &args) {

    if (is_clockwork) {
      //stream << "prog " << name << "() {" << std::endl;
      memory_stream << "prog " << name << "() {" << endl
                    << "  prog prg;" << std::endl
                    << "  prg.compute_unit_file = \"" << name << "_compute.h\";" << std::endl
                    << "  prg.name = \"" << name << "\";" << endl
                    << std::endl;
      loop_list.emplace_back("prg");
      //mem_bodyname = "prg";
    }
  
    //stream << "void " << name << "(\n";
    
    for (size_t i = 0; i < args.size(); i++) {
        string arg_name = "arg_" + std::to_string(i);
        if (args[i].is_stencil) {
            HW_Stencil_Type stype = args[i].stencil_type;
            internal_assert(args[i].stencil_type.type == HW_Stencil_Type::StencilContainerType::AxiStream ||
                            args[i].stencil_type.type == HW_Stencil_Type::StencilContainerType::Stencil);
            //stream << print_stencil_type(args[i].stencil_type) << " ";
            if (args[i].stencil_type.type == HW_Stencil_Type::StencilContainerType::AxiStream) {
              //stream << "&";  // hls_stream needs to be passed by reference
            }
            //stream << arg_name;
            allocations.push(args[i].name, {args[i].stencil_type.elemType});
            stencils.push(args[i].name, args[i].stencil_type);
        } else {
          //stream << print_type(args[i].scalar_type) << " " << arg_name;
        }

    }
    
    //memory_stream << std::endl;
    
    if (is_header()) {
      //stream << ");\n";
    } else {
        stream << ")\n";
        open_scope();

        // add HLS pragma at function scope
        stream << "#pragma HLS DATAFLOW\n"
               << "#pragma HLS INLINE region\n"
               << "#pragma HLS INTERFACE s_axilite port=return bundle=config\n";
        for (size_t i = 0; i < args.size(); i++) {
            string arg_name = "arg_" + std::to_string(i);
            if (args[i].is_stencil) {
                if (ends_with(args[i].name, ".stream")) {
                    // stream arguments use AXI-stream interface
                    stream << "#pragma HLS INTERFACE axis register "
                           << "port=" << arg_name << " name=" << arg_name << "\n";
                } else {
                    // stencil arguments use AXI-lite interface
                    stream << "#pragma HLS INTERFACE s_axilite "
                           << "port=" << arg_name
                           << " bundle=config\n";
                    stream << "#pragma HLS ARRAY_PARTITION "
                           << "variable=" << arg_name << ".value complete dim=0\n";
                }
            } else {
                // scalar arguments use AXI-lite interface
                stream << "#pragma HLS INTERFACE s_axilite "
                       << "port=" << arg_name << " bundle=config\n";
            }
        }
        stream << "\n";
        //return;

        // create alias (references) of the arguments using the names in the IR
        do_indent();
        stream << "// alias the arguments\n";
        int output_count = 0;
        for (size_t i = 0; i < args.size(); i++) {
            string arg_name = "arg_" + std::to_string(i);
            do_indent();
            if (args[i].is_stencil) {
                HW_Stencil_Type stype = args[i].stencil_type;
                memory_stream << "// " << print_stencil_type(args[i].stencil_type) << " &"
                              << printname(args[i].name) << " = " << arg_name << ";\n";
                string io_name = strip_stream(printname(args[i].name));
                if (args[i].is_output) {
                  memory_stream << "  prg.add_output(\"" << io_name << "\");" << endl;
                  output = io_name;
                  output_count++;
                } else {
                  memory_stream << "  prg.add_input(\"" << io_name << "\");" << endl;
                  inputs.push_back(io_name);
                }
                add_buffer(io_name, stype.elemType.bits());
                //add_buffer(io_name, 16);
                stream << print_stencil_type(args[i].stencil_type) << " &"
                       << printname(args[i].name) << " = " << arg_name << ";\n";
            } else {
                memory_stream << print_type(args[i].scalar_type) << " &"
                              << printname(args[i].name) << " = " << arg_name << ";\n";
                stream << print_type(args[i].scalar_type) << " &"
                       << printname(args[i].name) << " = " << arg_name << ";\n";
            }
        }

        memory_stream << endl;
        //stream << "\n";

        // print body
        print(stmt);

        // TLAST
        if(output_count == 1) {

            // Get output buffer closure argument
            const HW_Arg *output_arg = nullptr;
            for(size_t i = 0; i < args.size(); i++) {
                if(args[i].is_output) {
                    output_arg = &args[i];
                    break;
                }
            }
            
            if(output_arg) {
                // Memory
                memory_stream << "\n// TLAST Generation";

                // Add "tlast" output
                memory_stream << "\n  prg.add_output(\"tlast\");\n";
                memory_stream << "  prg.buffer_port_widths[\"tlast\"] = 1;\n";

                OutputInfoExtract ie { printname(output_arg->name) };
                stmt.accept(&ie);
                size_t n_loop_vars = ie.info.size();
                internal_assert(n_loop_vars > 0) << "no loop variables found for output buffer";
                std::string inner_lvar_name = ie.info[0].first;
                
                memory_stream << "  auto hcompute_tlast_op = " << inner_lvar_name << "->add_op(\"op_hcompute_tlast\");\n";
                memory_stream << "  hcompute_tlast_op->add_function(\"hcompute_tlast\");\n";
                memory_stream << "  hcompute_tlast_op->add_store(\"tlast\"";
                for(const auto &p : ie.info) {
                    memory_stream << ", \"" << p.first << "\"";
                }
                memory_stream << ");";

                for(const auto &p : ie.info) {
                    memory_stream << "\n  hcompute_tlast_op->compute_unit_needs_index_variable(\"" << p.first << "\");";
                }
                memory_stream << endl;

                // Compute
                std::ostringstream comp_args, bound_checks;
                comp_args << "const int " << ie.info[0].first;
                bound_checks << "(" << ie.info[0].first << " == " << print_expr(ie.info[0].second) << ")";
                for(size_t i = 1; i < ie.info.size(); i++) {
                    comp_args << ", const int " << ie.info[i].first;
                    bound_checks << " && (" << ie.info[i].first << " == " << print_expr(ie.info[i].second) << ")";
                }
                compute_stream << "\nhw_uint<1> hcompute_tlast(" << comp_args.str() << ") {\n";
                compute_stream << "  return (" << bound_checks.str() << ") ? 1 : 0;\n";
                compute_stream << "}";
            }

        }

        close_scope("kernel hls_target" + printname(name));
    }
    //stream << "\n";

    for (size_t i = 0; i < args.size(); i++) {
        // Remove buffer arguments from allocation scope
        if (args[i].stencil_type.type == HW_Stencil_Type::StencilContainerType::Stream) {
            allocations.pop(args[i].name);
            stencils.pop(args[i].name);
        }
    }
}

void print_clockwork_codegen(string appname, ofstream& stream) {
  stream << "#include \"" << appname << "_compute.h\"" << endl
         << "#include \"" << appname << "_memory.cpp\"" << endl
         << endl
         << "int main(int argc, char **argv) {" << endl
         << "  prog prg = " << appname << "();" << endl
         << "  std::vector<std::string> args(argv + 1, argv + argc);" << endl
         << "  size_t i=0;" << endl
         << "  while (i < args.size()) {" << endl
         << "    if (args[i] == \"opt\") {" << endl
         << "      generate_optimized_code(prg);" << endl
         << "    } else if (args[i] == \"unopt\") {" << endl
         << "      generate_unoptimized_code(prg);" << endl
         << "    }" << endl
         << "    i += 1;" << endl
         << "  }" << endl
         << "  return 0;" << endl
         << "}" << endl;
}

void print_clockwork_execution_header(string appname, ofstream& stream) {
  stream << "#ifndef RDAI_CLOCKWORK_WRAPPER\n"
         << "#define RDAI_CLOCKWORK_WRAPPER\n"
         << "\n"
         << "#include \"rdai_api.h\"\n"
         << endl
         << "/**\n"
         << " * Run clockwork kernel "<< appname << "\n"
         << "\n"
         << " * @param mem_obj_list List of input and output buffers\n"
         << " * NOTE: last element in mem_obj_list points to output buffer\n"
         <<  " *       whereas the remaining elements point to input buffers.\n"
         << " */\n"
         << "void run_clockwork_program(RDAI_MemObject **mem_obj_list);\n"
         << "\n"
         << "#endif // RDAI_CLOCKWORK_WRAPPER";
}

void print_clockwork_execution_cpp_16bit(string appname, const vector<HW_Arg>& closure_args, ofstream& stream) {
    stream << "#include \"clockwork_testscript.h\"\n"
           << "#include \"unoptimized_" << appname << ".h\"\n"
           << "#include \"hw_classes.h\"\n"
           << "\n"
           << "void run_clockwork_program(RDAI_MemObject **mem_object_list) {\n";

    size_t num_buffers = closure_args.size();

    // get sizes of buffer elements
    vector<int> elt_sizes(num_buffers);
    for(size_t i = 0; i < num_buffers; i++) {
        elt_sizes[i] = closure_args[i].stencil_type.elemType.bits();
    }

    // emit buffer declarations
    stream << "\t// input and output memory objects\n";
    for(size_t i = 0; i < num_buffers; i++) {
        ostringstream oss;
        oss << type_to_c_type(closure_args[i].stencil_type.elemType);
        string type_name = oss.str();
        stream << "\t" << type_name << " *" << printname(closure_args[i].name) << " = (" << type_name << "* )";
        stream << " mem_object_list[" << i << "]->host_ptr;\n";
    }
    stream << "\n";

    // emit input and output stream declarations;
    stream << "\t// input and output stream declarations\n";
    for(size_t i = 0; i < num_buffers; i++) {
        stream << "\tHWStream< hw_uint<16> > " << printname(closure_args[i].name) << "_stream;\n";
    }
    stream << "\tHWStream< hw_uint<1> > tlast_stream;\n";
    stream << "\n";

    // copy inputs from buffers to streams
    if(num_buffers > 1) {
        for(size_t i = 0; i < num_buffers - 1; i++) {
            string stream_name = printname(closure_args[i].name) + "_stream";
            stream << "\t// provision input stream " << stream_name << "\n";
            Region bounds = closure_args[i].stencil_type.bounds;
            for(size_t j = 0; j < bounds.size(); j++) {
                size_t k = bounds.size() - j - 1;
                ostringstream oss;
                oss << "l" << k;
                string varname = oss.str();
                stream << "\tfor(int "<< varname <<" = 0; "<< varname <<" < "<< bounds[k].extent<<"; "<< varname <<"++) {\n";
            }
        
            Expr temp_stride = 1;
            Expr temp_arg = Variable::make(Int(32), "l0");
            for(size_t j = 1; j < bounds.size(); j++) {
                ostringstream oss;
                oss << "l" << j;
                string varname = oss.str();
                temp_stride = temp_stride * bounds[j-1].extent;
                temp_arg = temp_arg + (Variable::make(Int(32), varname) * temp_stride);
            } 
            temp_arg = simplify(temp_arg);
            stream << "\t\thw_uint<16> in_val;\n";
            stream << "\t\tset_at<0, 16, 16>(in_val, ";
            stream <<  "hw_uint<16>(" << printname(closure_args[i].name) << "[" << temp_arg <<"]));\n";
            stream << "\t\t" << stream_name << ".write(in_val);\n";

            stream << "\t";
            for(size_t j = 0; j < bounds.size(); j++) {
                stream << "} ";
            }
            stream << "\n";
        }
    }
    stream << "\n\n";

    // emit kernel call
    stream << "\t// invoke clockwork program\n";
    stream << "\tunoptimized_" << appname << "(\n";
    for(size_t i = 0; i < num_buffers; i++) {
        stream << "\t\t" << printname(closure_args[i].name) << "_stream,\n";
    }
    stream << "\t\ttlast_stream\n";
    stream << "\t);\n\n";

    // copy output from stream to buffer
    {
        HW_Arg stencil_arg = closure_args[num_buffers-1];
        string stream_name = printname(stencil_arg.name) + "_stream";
        stream << "\t// provision output buffer\n";
        Region bounds = stencil_arg.stencil_type.bounds;
        for(size_t i = 0; i < bounds.size(); i++) {
            size_t j = bounds.size() - i - 1;
            ostringstream oss;
            oss << "l" << j;
            string varname = oss.str();
            stream << "\tfor(int "<< varname << " = 0; " << varname << " < " << bounds[j].extent << "; " << varname << "++) {\n";
        }
        Expr temp_stride = 1;
        Expr temp_arg = Variable::make(Int(32), "l0");
        for(size_t i = 1; i < bounds.size(); i++) {
            ostringstream oss;
            oss << "l" << i;
            string varname = oss.str();
            temp_stride = temp_stride * bounds[i-1].extent;
            temp_arg = temp_arg + (Variable::make(Int(32), varname) * temp_stride);
        }
        temp_arg = simplify(temp_arg);
        int elt_size = elt_sizes[num_buffers-1];
        stream << "\t\thw_uint<16> actual = " << stream_name << ".read();\n";
        stream << "\t\tint actual_lane = actual.extract<0, "<< elt_size-1 << ">();\n";
        stream << "\t\t" << printname(stencil_arg.name) << "[" << temp_arg << "] = "; 
        stream << "(" << type_to_c_type(stencil_arg.stencil_type.elemType) << ") actual_lane;\n";
        stream << "\t";
        for(size_t i = 0; i < bounds.size(); i++) {
            stream << "} ";
        }
        stream << "\n";
    }
    

    stream << "}\n";
}

void print_clockwork_execution_cpp(string appname, const vector<HW_Arg>& closure_args, ofstream& stream) {
    stream << "#include \"clockwork_testscript.h\"\n"
           << "#include \"unoptimized_" << appname << ".h\"\n"
           << "#include \"hw_classes.h\"\n"
           << "\n"
           << "void run_clockwork_program(RDAI_MemObject **mem_object_list) {\n";

    size_t num_buffers = closure_args.size();

    // get sizes of buffer elements
    vector<int> elt_sizes(num_buffers);
    for(size_t i = 0; i < num_buffers; i++) {
        elt_sizes[i] = closure_args[i].stencil_type.elemType.bits();
    }

    // emit buffer declarations
    stream << "\t// input and output memory objects\n";
    //std::cout << "num_buffers=" << num_buffers << "  ending with " << closure_args[num_buffers-1].name << std::endl;
    for(size_t i = 0; i < num_buffers; i++) {
      //std::cout << printname(closure_args[i].name) << std::endl;
        ostringstream oss;
        oss << type_to_c_type(closure_args[i].stencil_type.elemType);
        string type_name = oss.str();
        stream << "\t" << type_name << " *" << printname(closure_args[i].name) << " = (" << type_name << "* )";
        stream << " mem_object_list[" << i << "]->host_ptr;\n";
    }
    stream << "\n";

    // emit input and output stream declarations;
    stream << "\t// input and output stream declarations\n";
    for(size_t i = 0; i < num_buffers; i++) {
        stream << "\tHWStream< hw_uint<" << elt_sizes[i] << "> > " << printname(closure_args[i].name) << "_stream;\n";
    }
    stream << "\tHWStream < hw_uint<1> > tlast_stream;\n";
    stream << "\n";

    // copy inputs from buffers to streams
    if(num_buffers > 1) {
        for(size_t i = 0; i < num_buffers - 1; i++) {
            string stream_name = printname(closure_args[i].name) + "_stream";
            stream << "\t// provision input stream " << stream_name << "\n";
            Region bounds = closure_args[i].stencil_type.bounds;
            for(size_t j = 0; j < bounds.size(); j++) {
                size_t k = bounds.size() - j - 1;
                ostringstream oss;
                oss << "l" << k;
                string varname = oss.str();
                stream << "\tfor(int "<< varname <<" = 0; "<< varname <<" < "<< bounds[k].extent<<"; "<< varname <<"++) {\n";
            }
        
            Expr temp_stride = 1;
            Expr temp_arg = Variable::make(Int(32), "l0");
            for(size_t j = 1; j < bounds.size(); j++) {
                ostringstream oss;
                oss << "l" << j;
                string varname = oss.str();
                temp_stride = temp_stride * bounds[j-1].extent;
                temp_arg = temp_arg + (Variable::make(Int(32), varname) * temp_stride);
            } 
            temp_arg = simplify(temp_arg);
            stream << "\t\thw_uint<"<<elt_sizes[i]<<"> in_val;\n";
            stream << "\t\tset_at<0, " << elt_sizes[i] << ", " << elt_sizes[i] << ">(in_val, ";
            stream <<  "hw_uint<" << elt_sizes[i] << ">(" << printname(closure_args[i].name) << "[" << temp_arg <<"]));\n";
            stream << "\t\t" << stream_name << ".write(in_val);\n";

            stream << "\t";
            for(size_t j = 0; j < bounds.size(); j++) {
                stream << "} ";
            }
            stream << "\n";
        }
    }
    stream << "\n\n";

    // emit kernel call
    stream << "\t// invoke clockwork program\n";
    stream << "\tunoptimized_" << appname << "(\n";
    for(size_t i = 0; i < num_buffers; i++) {
        stream << "\t\t" << printname(closure_args[i].name) << "_stream,\n";
    }
    stream << "\t\ttlast_stream\n";
    stream << "\t);\n\n";

    // copy output from stream to buffer
    {
        HW_Arg stencil_arg = closure_args[num_buffers-1];
        string stream_name = printname(stencil_arg.name) + "_stream";
        stream << "\t// provision output buffer\n";
        Region bounds = stencil_arg.stencil_type.bounds;
        for(size_t i = 0; i < bounds.size(); i++) {
            size_t j = bounds.size() - i - 1;
            ostringstream oss;
            oss << "l" << j;
            string varname = oss.str();
            stream << "\tfor(int "<< varname << " = 0; " << varname << " < " << bounds[j].extent << "; " << varname << "++) {\n";
        }
        Expr temp_stride = 1;
        Expr temp_arg = Variable::make(Int(32), "l0");
        for(size_t i = 1; i < bounds.size(); i++) {
            ostringstream oss;
            oss << "l" << i;
            string varname = oss.str();
            temp_stride = temp_stride * bounds[i-1].extent;
            temp_arg = temp_arg + (Variable::make(Int(32), varname) * temp_stride);
        }
        temp_arg = simplify(temp_arg);
        int elt_size = elt_sizes[num_buffers-1];
        stream << "\t\thw_uint<" << elt_size << "> actual = " << stream_name << ".read();\n";
        stream << "\t\tint actual_lane = actual.extract<0, "<< elt_size-1 << ">();\n";
        stream << "\t\t" << printname(stencil_arg.name) << "[" << temp_arg << "] = "; 
        stream << "(" << type_to_c_type(stencil_arg.stencil_type.elemType) << ") actual_lane;\n";
        stream << "\t";
        for(size_t i = 0; i < bounds.size(); i++) {
            stream << "} ";
        }
        stream << "\n";
    }
    

    stream << "}\n";
}

/** Substitute an Expr for another Expr in a graph. Unlike substitute,
 * this only checks for shallow equality. */
class VarGraphSubstituteExpr : public IRGraphMutator2 {
    Expr find, replace;
public:

    using IRGraphMutator2::mutate;

    Expr mutate(const Expr &e) override {
        if (equal(find, e)) {
            return replace;
        } else {
            return IRGraphMutator2::mutate(e);
        }
    }

    VarGraphSubstituteExpr(const Expr &find, const Expr &replace) : find(find), replace(replace) {}
};

Expr var_graph_substitute(const Expr &name, const Expr &replacement, const Expr &expr) {
    return VarGraphSubstituteExpr(name, replacement).mutate(expr);
}

class ContainedLoopNames : public IRVisitor {
  std::vector<std::string> loopnames;
  
  using IRVisitor::visit;
  void visit(const Variable *op) override {
    for (auto loopname : loopnames) {
      //string var_as_loop = "loop_" +printname(op->name);
      string var_as_loop = printname(op->name);
      if (loopname == var_as_loop) {
        used_loopnames[loopname] = op;
        return;
      }
    }
  }

public:
  map<string, const Variable*> used_loopnames;
  ContainedLoopNames(vector<string> loopnames) : loopnames(loopnames) {}
};

std::map<std::string, const Variable*> used_loops(Expr e, std::vector<std::string> loopnames) {
  ContainedLoopNames cln(loopnames);
  e.accept(&cln);
  return cln.used_loopnames;
}

struct Compute_Argument {
    std::string name;

    bool is_stencil;
    bool is_output;
    bool is_dynamic;
    Type type;

    std::vector<Expr> args;
    std::vector<bool> args_is_dynamic;
    std::string bufname;
    Expr call;
};

struct CArg_compare {
    bool operator() (const Compute_Argument& arg1, const Compute_Argument& arg2) const {
      return !equal(arg1.call, arg2.call);
    }
};

Expr add_floor_to_divs(Expr);
void print_dynamic_args(const vector<Expr>& args,
                        const vector<bool>& dynamic_calls_found,
                        const Scope<Expr>& scope,
                        ostream& memory_stream) {
  for (int argi=args.size()-1; argi>=0; --argi) {
    auto arg = args[argi];
    ostringstream arg_print;
    if (dynamic_calls_found[argi]) {
      const Call* index_call = arg.as<Call>();
      internal_assert(index_call) <<
        "dynamic argument must be a call:" << Expr(index_call) << "\n";

      memory_stream << ", \"" << printname(index_call->name) << "\"";
      //std::cout << ", \"" << printname(index_call->name) << "\"";
      for (int argi=index_call->args.size()-1; argi>=0; --argi) {
        auto dynamic_index = index_call->args[argi];
        ostringstream index_print;
        index_print << add_floor_to_divs(expand_expr(dynamic_index, scope));
        memory_stream << ", \"" << removedots(index_print.str()) << "\"";
        //std::cout << ", \"" << removedots(index_print.str()) << "\"";
      }

    } else {
      arg_print << expand_expr(arg, scope);
      memory_stream << ", \"" << removedots(arg_print.str()) << "\"";
    }
    //std::cout << ");" << std::endl;
  }
}

void merge_arguments(vector<Compute_Argument>& compute_args,
                     map<string, vector<Compute_Argument> >& merged_args,
                     vector<string>& arg_order) {
  for (auto compute_arg : compute_args) {
    if (merged_args.count(compute_arg.bufname) == 0) {
      merged_args[compute_arg.bufname] = {compute_arg};
      arg_order.emplace_back(compute_arg.bufname);
    } else {
      merged_args.at(compute_arg.bufname).emplace_back(compute_arg);
    }
  }
}

Expr substitute_compute_args(Expr expr, vector<Compute_Argument>& compute_args) {
  Expr new_expr = expr;
  for (size_t i=0; i<compute_args.size(); ++i) {
    string new_name = printname(compute_args[i].name);
    if (false) { //if (merged_args[compute_args[i].bufname].size() == 1) {
      new_name = printname(compute_args[i].bufname);
    }
    auto var_replacement = Variable::make(compute_args[i].type, new_name);
    //compute_stream << "// replacing " << Expr(compute_args[i].call) << " with " << var_replacement << std::endl;
    new_expr = var_graph_substitute(Expr(compute_args[i].call), var_replacement, new_expr);
  }
  
  return new_expr;
}

Expr substitute_loop_vars(Expr expr, map<string, const Variable*>& loopname_map) {
  Expr new_expr = expr;
  
  for (auto loopname_pair : loopname_map) {
    auto used_loopname = loopname_pair.first;
    
    const Variable* old_loopvar = loopname_pair.second;
    auto var_replacement = Variable::make(old_loopvar->type, used_loopname);
    //compute_stream << "// replacing " << Expr(compute_args[i].call) << " with " << var_replacement << std::endl;
    new_expr = var_graph_substitute(Expr(old_loopvar), var_replacement, new_expr);
  }

  return new_expr;
}

void add_stores_to_coreir_interface(CoreIR_Interface& iface,
                                    vector<string>& arg_order,
                                    map<string, vector<Compute_Argument> >& merged_args) {
  for (auto argname : arg_order) {
    //uint total_bitwidth = merged_args[argname].size() * esize;
    uint total_bitwidth = 0;
    for (auto arg : merged_args[argname]) {
      total_bitwidth += arg.type.bits();
    }
    
    vector<CoreIR_Port> iports;
    for (auto arg_component : merged_args[argname]) {
      uint esize = arg_component.type.bits();
      if (false) {//merged_args[argname].size() == 1) {
        iports.push_back(CoreIR_Port({printname(arg_component.bufname), esize}));
      } else {
        iports.push_back(CoreIR_Port({printname(arg_component.name), esize}));
      }
    }
    iface.inputs.push_back(CoreIR_PortBundle({printname(argname), total_bitwidth, iports}));
  }
}

string rom_to_c(ROM_data rom);
void output_roms(const vector<string>& found_roms, map<string, ROM_data>& roms,
                 std::ostream& compute_stream, vector<CoreIR_Inst_Args>& coreir_insts,
                 CoreIR::Context* context) {
  for (auto found_rom : found_roms) {
    auto pc_str = rom_to_c(roms[found_rom]);
    compute_stream << pc_str << std::endl;

    vector<int> rom_size;
    if (const Realize* rom_r = roms[found_rom].stmt.as<Realize>()) {
      for (auto bound : rom_r->bounds) {
        rom_size.emplace_back(to_int(bound.extent));
      }
    } else if (const Allocate* rom_a = roms[found_rom].stmt.as<Allocate>()) {
      for (auto bound : rom_a->extents) {
        rom_size.emplace_back(to_int(bound));
      }
    }
    auto coreir_inst = rom_to_coreir(found_rom, rom_size, roms[found_rom].produce, context);
    coreir_insts.push_back(coreir_inst);
  }
}




class Compute_Closure : public Closure {
public:
  Compute_Closure(Stmt s, std::string output_string, std::set<string> ignoring)  {
    for (auto s : ignoring) {
      //ScopedBinding<> p(ignore, s);
      ignore.push(s);
    }

    s.accept(this);
    output_name = output_string;
  }

  vector<Compute_Argument> arguments();

protected:
  using Closure::visit;
  std::string output_name;
  map<string, Compute_Argument> var_comparg;

  void visit(const Load *op);
  void visit(const Call *op);
  void visit(const Provide *op);
  std::set<Compute_Argument, CArg_compare> unique_args;
  //std::map<string, int> unique_argstrs;
  std::set<string> unique_argstrs;

};

void Compute_Closure::visit(const Provide *op) {
  // only go through RHS, not LHS
  for (size_t i = 0; i < op->values.size(); i++) {
    op->values[i].accept(this);
  }
}

void Compute_Closure::visit(const Load *op) {
  //Closure::visit(op);

  if (!ignore.contains(op->name)) {
    debug(3) << "adding to closure.\n";

    ostringstream arg_print;
    arg_print << Expr(op);
      
    //auto comp_arg = Compute_Argument({"", false, false, op->type, op->args, op->name, Expr(op)});
    //if (unique_args.count(comp_arg) == 0) {
    if (unique_argstrs.count(printname(arg_print.str())) == 0) {
      string argname = unique_name(op->name);
      if (argname == op->name) { argname = unique_name(op->name); }
        
      unique_argstrs.insert(printname(arg_print.str()));
      vars[argname] = op->type;
      vector<bool> calls_found;
      bool is_dynamic = contains_call({op->index}, calls_found);
      var_comparg[argname] = Compute_Argument({argname, false, false, is_dynamic, op->type,
            {op->index}, calls_found, op->name, Expr(op)});;

    } else {
      //std::cout << "not adding " << Expr(op) << std::endl;
    }
    // do not recurse
    return;
  } else {
    debug(3) << "not adding to closure.\n";
  }

  Closure::visit(op);
  
}

void Compute_Closure::visit(const Call *op) {
  if (op->call_type == Call::Intrinsic &&
      (ends_with(op->name, ".stencil") || ends_with(op->name, ".stencil_update"))) {
    // consider call to stencil and stencil_update
    debug(3) << "visit call " << op->name << ": ";
    if (!ignore.contains(op->name)) {
      debug(3) << "adding to closure.\n";

      ostringstream arg_print;
      arg_print << Expr(op);
      
      //auto comp_arg = Compute_Argument({"", false, false, op->type, op->args, op->name, Expr(op)});
      //if (unique_args.count(comp_arg) == 0) {
      if (unique_argstrs.count(printname(arg_print.str())) == 0) {
        string argname = unique_name(op->name);
        //std::cout << "adding arg " << argname << Expr(op);
        if (argname == op->name) { argname = unique_name(op->name); }
        
        unique_argstrs.insert(printname(arg_print.str()));
        vars[argname] = op->type;
        vector<bool> calls_found;
        bool is_dynamic = contains_call(op->args, calls_found);
        var_comparg[argname] = Compute_Argument({argname, false, false, is_dynamic, op->type,
              op->args, calls_found, op->name, Expr(op)});;

      } else {
        //std::cout << "not adding " << Expr(op) << std::endl;
      }
      // do not recurse
      return;
    } else {
      debug(3) << "not adding to closure.\n";
    }
  }
  IRVisitor::visit(op);
}

vector<Compute_Argument> Compute_Closure::arguments() {
    vector<Compute_Argument> res;
    for (const std::pair<string, Closure::Buffer> &i : buffers) {
      std::cout << "buffer: " << i.first << " " << i.second.size;
        //if (i.second.read) std::cout << " (read)";
        //if (i.second.write) std::cout << " (write)";
        std::cout << "\n";
        if (i.second.read) {
          if (var_comparg.count(i.first) > 0) {
            res.push_back(var_comparg[i.first]);
          } else {
            res.push_back({i.first, true, true, false, Type()});
          }
        }
    }
    //internal_assert(buffers.empty()) << "we expect no references to buffers in a hw pipeline.\n";
    for (const std::pair<string, Type> &i : vars) {
      //std::cout << "var: " << i.first << "\n";
      if (i.first.find(".stencil") != string::npos ||
          ends_with(i.first, ".stream") ||
          ends_with(i.first, ".stencil_update")) {
          
          bool is_output = (starts_with(i.first, output_name));
          if (var_comparg.count(i.first) > 0) {
            res.push_back(var_comparg[i.first]);
          } else {
            res.push_back({i.first, true, is_output, false, Type()});
          }

        } else if (ends_with(i.first, ".stencil_update")) {
          //internal_error << "we don't expect to see a stencil_update type in Compute_Closure.\n";
        } else {
            // it is a scalar variable
          //res.push_back({i.first, false, true, i.second, HW_Stencil_Type()});
        }
    }
    return res;
}

// Polyhedral analysis likes floor with div
class DivWithFloor : public IRMutator {
  using IRMutator::visit;
  Expr visit(const Div *op) {
    return Call::make(op->type, "floor", {Expr(op)}, Call::CallType::PureIntrinsic);
  }
public:
  DivWithFloor() {}
};
Expr add_floor_to_divs(Expr e) {
    DivWithFloor dwf;
    return dwf.mutate(e);
}

class CodeGen_C_Expr : public CodeGen_C {
  using IRPrinter::visit;
  void visit(const Min *op) override;
  void visit(const Max *op) override;
  void visit(const Provide *op) override;
  void visit(const Call *op) override;
  void visit(const Realize *op) override;
  void visit(const Allocate *op) override;
  
public:
  CodeGen_C_Expr(std::ostream& dest, Target target) : CodeGen_C(dest, target) {
    indent += 2;
  }

  void print_output(Expr e) {
    string name = print_expr(e);
    do_indent();
    stream << "return " << name << ";";
  }
};

void CodeGen_C_Expr::visit(const Min *op) {
    // clang doesn't support the ternary operator on OpenCL style vectors.
    // See: https://bugs.llvm.org/show_bug.cgi?id=33103
    if (op->type.is_scalar()) {
      print_expr(Call::make(op->type, "min", {op->a, op->b}, Call::Extern));
    } else {
      ostringstream rhs;
      rhs << print_type(op->type) << "::min(" << print_expr(op->a) << ", " << print_expr(op->b) << ")";
      print_assignment(op->type, rhs.str());
    }
}
void CodeGen_C_Expr::visit(const Max *op) {
    // clang doesn't support the ternary operator on OpenCL style vectors.
    // See: https://bugs.llvm.org/show_bug.cgi?id=33103
    if (op->type.is_scalar()) {
        print_expr(Call::make(op->type, "max", {op->a, op->b}, Call::Extern));
    } else {
        ostringstream rhs;
        rhs << print_type(op->type) << "::max(" << print_expr(op->a) << ", " << print_expr(op->b) << ")";
        print_assignment(op->type, rhs.str());
    }
}
void CodeGen_C_Expr::visit(const Provide *op) {
  internal_assert(op->values.size() == 1);
  //internal_assert(op->args.size() == 1) << "provide has 2+ args: " << Stmt(op) << "\n";
  vector<string> indices;
  for (auto index : op->args) {
    string id_index = print_expr(index);
    indices.emplace_back(id_index);
  }

  string id_value = print_expr(op->values[0]);
  //string id_index = print_expr(op->args[0]);
  do_indent();
  stream << op->name;
  for (auto id_index : indices) {
    stream << "[" << id_index << "]";
  }
  stream << " = " << id_value << ";\n";
}
void CodeGen_C_Expr::visit(const Call *op) {
  if (op->name == "floor_f32") {
    internal_assert(op->args.size() == 1);
    print_assignment(op->type, "floorf(" + print_expr(op->args[0]) + ")");
  } else if (op->name == "round_f32") {
    internal_assert(op->args.size() == 1);
    print_assignment(op->type, "roundf(" + print_expr(op->args[0]) + ")");
  } else if (op->name == "ceil_f32") {
    internal_assert(op->args.size() == 1);
    print_assignment(op->type, "ceilf(" + print_expr(op->args[0]) + ")");

  } else if (op->name == "log_f32") {
    internal_assert(op->args.size() == 1);
    print_assignment(op->type, "logf(" + print_expr(op->args[0]) + ")");
  } else if (op->name == "exp_f32") {
    internal_assert(op->args.size() == 1);
    print_assignment(op->type, "expf(" + print_expr(op->args[0]) + ")");
  } else if (op->name == "sqrt_f32") {
    internal_assert(op->args.size() == 1);
    print_assignment(op->type, "sqrtf(" + print_expr(op->args[0]) + ")");
    
  } else if (op->name == "sin_f32") {
    internal_assert(op->args.size() == 1);
    print_assignment(op->type, "sinf(" + print_expr(op->args[0]) + ")");
  } else if (op->name == "cos_f32") {
    internal_assert(op->args.size() == 1);
    print_assignment(op->type, "cosf(" + print_expr(op->args[0]) + ")");
  } else if (op->name == "tan_f32") {
    internal_assert(op->args.size() == 1);
    print_assignment(op->type, "tanf(" + print_expr(op->args[0]) + ")");
  } else if (op->name == "tanh_f32") {
    internal_assert(op->args.size() == 1);
    print_assignment(op->type, "tanhf(" + print_expr(op->args[0]) + ")");
  } else if (op->name == "asin_f32") {
    internal_assert(op->args.size() == 1);
    print_assignment(op->type, "asinf(" + print_expr(op->args[0]) + ")");
  } else if (op->name == "acos_f32") {
    internal_assert(op->args.size() == 1);
    print_assignment(op->type, "acosf(" + print_expr(op->args[0]) + ")");
  } else if (op->name == "atan_f32") {
    internal_assert(op->args.size() == 1);
    print_assignment(op->type, "atanf(" + print_expr(op->args[0]) + ")");
  } else if (op->name == "atan2_f32") {
    internal_assert(op->args.size() == 2);
    string y = print_expr(op->args[0]);
    string x = print_expr(op->args[1]);
    print_assignment(op->type, "atan2f(" + y + ", " + x + ")");
    
  } else if (ends_with(op->name, ".stencil")) {
    //internal_assert(op->args.size() == 1);
    //string id_index = print_expr(op->args[0]);
    vector<string> indices;
    for (auto index : op->args) {
      string id_index = print_expr(index);
      indices.emplace_back(id_index);
    }

    ostringstream rhs;
    do_indent();
    rhs << printname(op->name);// << "[" << id_index << "]";
    for (auto id_index : indices) {
      rhs << "[" << id_index << "]";
    }

    print_assignment(op->type, rhs.str());

  } else {
    CodeGen_C::visit(op);
  }
}
void CodeGen_C_Expr::visit(const Realize *op) {
  do_indent();
  stream << type_to_c_type(op->types[0]) << " " << op->name;
  for (size_t i = 0; i < op->bounds.size(); i++) {
    stream << "[";
    stream << op->bounds[i].extent;
    stream << "]";
  }
  if (op->memory_type != MemoryType::Auto) {
    stream << " in " << op->memory_type;
  }
  if (!is_one(op->condition)) {
    stream << " if ";
    print(op->condition);
  }
  stream << ";";

  print(op->body);
}

void CodeGen_C_Expr::visit(const Allocate *op) {
    string op_name = print_name(op->name);
    string op_type = print_type(op->type, AppendSpace);

    internal_assert(!op->new_expr.defined());
    internal_assert(!is_zero(op->condition));
    int32_t constant_size;
    string size_id;
    constant_size = op->constant_allocation_size();
    if (constant_size > 0) {
      int64_t stack_bytes = constant_size * op->type.bytes();

      if (stack_bytes > ((int64_t(1) << 31) - 1)) {
        user_error << "Total size for allocation "
                   << op->name << " is constant but exceeds 2^31 - 1.\n";
      } else {
        size_id = print_expr(Expr(static_cast<int32_t>(constant_size)));
      }
    } else {
      internal_error << "Size for allocation " << op->name
                     << " is not a constant.\n";
    }
          
    Allocation alloc;
    alloc.type = op->type;
    allocations.push(op->name, alloc);

    do_indent();
    stream << op_type;

    stream << op_name
           << "[" << size_id << "];\n";

    op->body.accept(this);

    // Should have been freed internally
    //internal_assert(!allocations.contains(op->name)) << op->name << " was not freed correctly" << "\n";
}

string return_c_expr(Expr e) {
  ostringstream arg_print;
  CodeGen_C_Expr compute_codegen(arg_print, Target());
  arg_print.str("");
  arg_print.clear();

  //std::cout << e << std::endl;
  e.accept(&compute_codegen);
  compute_codegen.print_output(e);

  // return output
  return arg_print.str();
}

string return_c_stmt(Stmt s) {
  ostringstream arg_print;
  CodeGen_C_Expr compute_codegen(arg_print, Target());
  arg_print.str("");
  arg_print.clear();

  //std::cout << s << std::endl;
  s.accept(&compute_codegen);

  // return output
  return arg_print.str();
}

string rom_to_c(ROM_data rom) {
  auto produce = ProducerConsumer::make_produce(rom.name, rom.produce);
  if (const Realize* rom_r = rom.stmt.as<Realize>()) {
    auto realize = Realize::make(rom_r->name, rom_r->types, rom_r->memory_type,
                                 rom_r->bounds, rom_r->condition, produce);

    //std::cout << "rom " << rom.name << " being renamed " << printname(rom.name) << std::endl;
    Stmt new_realize = RenameRealize(rom.name, printname(rom.name)).mutate(Stmt(realize));
    auto combined = return_c_stmt(new_realize);

    // return output
    return combined;
  } else if (const Allocate* rom_r = rom.stmt.as<Allocate>()) {
    auto allocate = Allocate::make(rom_r->name, rom_r->type, rom_r->memory_type,
                                  rom_r->extents, rom_r->condition, produce);
    auto combined = return_c_stmt(allocate);
    return combined;

  } else {
    internal_error << "rom data ill formed\n";
    return "error";
  }
}

void CodeGen_Clockwork_Target::CodeGen_Clockwork_C::visit(const Provide *op) {
  // Don't output Provides to ROMs
  if (roms.count(op->name) > 0) {
    CodeGen_Clockwork_Base::visit(op);
    return;
  }
  
  // Debug output the provide we trying to do
  memory_stream << endl << "//store is: " << expand_expr(Stmt(op), scope);
  //std::cout << endl << "//store is: " << expand_expr(Stmt(op), scope);

  // Output the function in relation to the loop level
  auto mem_bodyname = loop_list.back();
  std::string func_name = printname(unique_name("hcompute_" + op->name));
  memory_stream << "  auto " << func_name  << " = "
                << mem_bodyname << "->add_op(\""
                << "op_" + func_name << "\");" << endl;
  memory_stream << "  " << func_name << "->add_function(\"" << func_name << "\");" << endl;

  // Find what the interface of this Provide is by using a closure
  CoreIR_Interface iface;
  iface.name = func_name;
  set<string> rom_set;
  for (auto rom_pair : roms) {
    rom_set.emplace(rom_pair.first);
  }
  Compute_Closure c(expand_expr(Stmt(op), scope), op->name, rom_set);
  vector<Compute_Argument> compute_args = c.arguments();

  // Add each load/call/used-data for this provide
  for (size_t i=0; i<compute_args.size(); ++i) {
    auto& arg = compute_args[i];
    string buffer_name = printname(arg.bufname);
    add_buffer(buffer_name, arg.type.bits());

    if (arg.is_dynamic) { //(contains_call(arg.args)) {
      memory_stream << "  " << func_name << "->add_dynamic_load(\""
                    << buffer_name << "\"";

      print_dynamic_args(arg.args, arg.args_is_dynamic, scope, memory_stream);
      
    } else {
      memory_stream << "  " << func_name << "->add_load(\""
                    << buffer_name << "\"";
      //for (auto index : arg.args) {
      for (int argi=arg.args.size()-1; argi>=0; --argi) {
        auto index = arg.args[argi];
        ostringstream index_print;
        index_print << add_floor_to_divs(expand_expr(index, scope));
        memory_stream << ", \"" << removedots(index_print.str()) << "\"";
      }
    }
    
    memory_stream << ");" << std::endl;
  }

  // Add the store/provide
  uint store_size = op->values[0].type().bits();
  add_buffer(printname(op->name), store_size);

  vector<bool> dynamic_stores_found;
  if (contains_call(op->args, dynamic_stores_found)) {
    memory_stream << "  " << func_name << "->add_dynamic_store(\""
                  << printname(op->name) << "\"";
    print_dynamic_args(op->args, dynamic_stores_found, scope, memory_stream);
    memory_stream << ");\n";
    
  } else {
    memory_stream << "  " << func_name << "->add_store(\""
                  << printname(op->name) << "\"";
    //for (auto arg : op->args) {
    for (int argi=op->args.size()-1; argi>=0; --argi) {
      auto arg = op->args[argi];
      ostringstream arg_print;
      arg_print << expand_expr(arg, scope);
      memory_stream << ", \"" << removedots(arg_print.str()) << "\"";
    }
    memory_stream << ");\n";
  }
  iface.output = CoreIR_Port({printname(op->name), store_size});

  // Output the compute. Starting with merging the arguments to the same buffer
  map<string, vector<Compute_Argument> > merged_args;
  vector<string> arg_order;
  merge_arguments(compute_args, merged_args, arg_order);

  // Output the compute function signature including store variables
  compute_stream << std::endl << "//store is: " << Stmt(op);
  internal_assert(op->values.size() == 1);
  //auto output_bits = output == printname(op->name) ? 16 : op->values[0].type().bits();
  auto output_bits = op->values[0].type().bits();
  compute_stream << "hw_uint<" << output_bits << "> " << func_name << "(";
  for (size_t i=0; i<arg_order.size(); ++i) {
    if (i != 0) { compute_stream << ", "; }
    auto argname = arg_order[i];
    
    uint total_bitwidth = 0;
    for (auto arg : merged_args[argname]) {
      total_bitwidth += arg.type.bits();
    }
    //uint total_bitwidth = merged_args[argname].size() * esize;
    compute_stream << "hw_uint<" << total_bitwidth << ">& " << printname(argname);
  }

  // Substitute each store variable in the compute statement
  Expr new_expr = expand_expr(substitute_in_all_lets(op->values[0]), scope);
  new_expr = substitute_compute_args(new_expr, compute_args);
  
  // Add each used loop variable to memory, compute signature, coreir interface
  auto loopname_map = used_loops(new_expr, loop_list);
  bool first_var = arg_order.size() == 0;
  for (auto loopname_pair : loopname_map) {
    auto used_loopname = loopname_pair.first;
    memory_stream << "  " << func_name << "->compute_unit_needs_index_variable(\""
                  << used_loopname << "\");" << std::endl;
    
    iface.indices.emplace_back(CoreIR_Port({used_loopname, 16})); // loops use 16 bit variables
    if (!first_var) { compute_stream << ", "; } else { first_var = false; }
    compute_stream << "const int _" << used_loopname;
  }
  compute_stream << ") {\n";

  // Add each of the stores to the coreir interface
  add_stores_to_coreir_interface(iface, arg_order, merged_args);

  // Extract each indiviaul store from the merged arguments
  for (auto merged_arg : merged_args) {
    vector<Compute_Argument> arg_components = merged_arg.second;
    auto bufname = printname(merged_arg.first);
    //if (arg_components.size() == 1) { continue; }

    for (size_t i=0; i<arg_components.size(); ++i) {
      auto arg_component = arg_components[i];
      int esize = arg_component.type.bits();
      string type = type_to_c_type(arg_component.type);
      compute_stream << "  " << type << " _" << printname(arg_component.name) << " = "
                     << "(" << type << ") "
                     << bufname << ".extract<" << esize*i << ", " << esize*(i+1)-1 << ">();" << std::endl;
    }
    compute_stream << std::endl;
  }

  // Substitute each loop variable in the compute statement
  new_expr = substitute_loop_vars(new_expr, loopname_map);

  // Output each of the ROMs to the compute (c and coreir)
  auto found_roms = contains_call(new_expr, rom_set);
  vector<CoreIR_Inst_Args> coreir_insts;
  output_roms(found_roms, roms, compute_stream, coreir_insts, context);

  // Output the c expr to the compute
  auto output = return_c_expr(new_expr);
  compute_stream << output << endl;
  compute_stream << "}" << endl;

  // Output the compute to a coreir module
  convert_compute_to_coreir(new_expr, iface, coreir_insts, context);

  CodeGen_Clockwork_Base::visit(op);
}

void CodeGen_Clockwork_Target::CodeGen_Clockwork_C::add_buffer(const string& buffer_name, int width) {
  if (buffers.count(buffer_name) == 0) {
    //std::cout << buffer_name << " added" << std::endl;
    memory_stream << "  prg.buffer_port_widths[\""
                  << buffer_name
                  << "\"] = " << width << ";" << endl;
    buffers.emplace(buffer_name);
  } else {
    //std::cout << buffer_name << " not added" << std::endl;
  }
}

void CodeGen_Clockwork_Target::CodeGen_Clockwork_C::visit(const ProducerConsumer *op) {
  if (op->is_producer) {
    memory_stream << "////producing " << op->name << endl;
  } else {
    memory_stream << endl << "//consuming " << op->name << endl;
  }

  if (roms.count(op->name) > 0 && op->is_producer) {
    roms[op->name].produce = op->body;
  }
    
  CodeGen_Clockwork_Base::visit(op);
}

// almost that same as CodeGen_C::visit(const For *)
// we just add a 'HLS PIPELINE' pragma after the 'for' statement
void CodeGen_Clockwork_Target::CodeGen_Clockwork_C::visit(const For *op) {
    internal_assert(op->for_type == ForType::Serial)
        << "Can only emit serial for loops to HLS C\n";

    string id_min = print_expr(op->min);
    //string id_extent = print_expr(op->extent);
    string id_max = print_expr(simplify(op->extent + op->min));

    do_indent();
    stream << "for (int "
           << printname(op->name)
           << " = " << id_min
           << "; "
           << printname(op->name)
           << " < " << id_max
           << "; "
           << printname(op->name)
           << "++)\n";

    // create a unique name for each loop variable and replace names within the body
    //string loopname = "loop_" + printname(op->name);
    string loopname = printname(unique_name(op->name));
    auto modified_body = substitute(op->name, Variable::make(Int(32), loopname), op->body);
    //string bodyname = mem_bodyname;
    string bodyname = loop_list.back();
    string addloop = bodyname == "prg" ? ".add_loop(" : "->add_loop(";
    memory_stream << "  auto " << loopname << " = "
                  << bodyname << addloop
                  << "\"" << loopname << "\""
                  << ", " << id_min
                  << ", " << id_max
                  << ");\n";

    //loops.emplace(op->name);
    loops.emplace(loopname);

    //string loopname = printname(op->name);
    //add_loop(op);
    
    open_scope();
    // add a 'PIPELINE' pragma if it is an innermost loop
    if (!contain_for_loop(op->body)) {
        //stream << "#pragma HLS DEPENDENCE array inter false\n"
        //       << "#pragma HLS LOOP_FLATTEN off\n";
        stream << "#pragma HLS PIPELINE II=1\n";
    }

    loop_list.emplace_back(loopname);
    //op->body.accept(this);
    modified_body.accept(this);
    internal_assert(loop_list.back() == loopname);
    loop_list.pop_back();
    //mem_bodyname = bodyname;
    
    close_scope("for " + printname(op->name));
}

void CodeGen_Clockwork_Target::CodeGen_Clockwork_C::visit(const LetStmt *op) {
  //std::cout << "saving to scope " << op->name << endl;
  ScopedBinding<Expr> bind(scope, op->name, simplify(expand_expr(op->value, scope)));
  IRVisitor::visit(op);
}

void CodeGen_Clockwork_Target::CodeGen_Clockwork_C::visit(const Call *op) {
  //std::cout << " looking at " << op->name << endl;
  if (op->name == "read_stream") {
    memory_stream << "// Call - read: " << Expr(op) << endl;
  } else if (op->name == "write_stream") {
    memory_stream << "// Call - write: " << Expr(op) << endl;
  }
  CodeGen_Clockwork_Base::visit(op);
}

void CodeGen_Clockwork_Target::CodeGen_Clockwork_C::visit(const Realize *op) {
  auto memtype = identify_realization(Stmt(op), op->name);
  //std::cout << op->name << " is a " << memtype << std::endl;
  if (memtype == ROM_REALIZATION) {
    roms[op->name] = ROM_data({op->name, Stmt(op), Stmt()});;
  }
  CodeGen_Clockwork_Base::visit(op);
}

// most code is copied from CodeGen_C::visit(const Allocate *)
// we want to check that the allocation size is constant, and
// add a 'HLS ARRAY_PARTITION' pragma to the allocation
void CodeGen_Clockwork_Target::CodeGen_Clockwork_C::visit(const Allocate *op) {
    // We don't add scopes, as it messes up the dataflow directives in HLS compiler.
    // Instead, we rename the allocation to a unique name
    //open_scope();

    internal_assert(!op->new_expr.defined());
    internal_assert(!is_zero(op->condition));
    int32_t constant_size;
    constant_size = op->constant_allocation_size();
    if (constant_size > 0) {

    } else {
        internal_error << "Size for allocation " << op->name
                       << " is not a constant.\n";
    }

    // rename allocation to avoid name conflict due to unrolling
    string alloc_name = op->name + unique_name('a');
    Stmt new_body = RenameAllocation(op->name, alloc_name).mutate(op->body);

    Allocation alloc;
    alloc.type = op->type;
    allocations.push(alloc_name, alloc);

    auto new_alloc = Allocate::make(alloc_name, op->type, op->memory_type, op->extents, op->condition, new_body);
    roms[alloc_name] = ROM_data({alloc_name, Stmt(new_alloc), Stmt()});;

    do_indent();
    stream << print_type(op->type) << ' '
           << printname(alloc_name)
           << "[" << constant_size << "];\n";
    // add a 'ARRAY_PARTITION" pragma
    //stream << "#pragma HLS ARRAY_PARTITION variable=" << printname(op->name) << " complete dim=0\n\n";

    new_body.accept(this);

    // Should have been freed internally
    //internal_assert(!allocations.contains(alloc_name)) << "allocation " << alloc_name << " is not freed.\n";

    //close_scope("alloc " + printname(op->name));

}

}
}
