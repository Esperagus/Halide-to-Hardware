#include <iostream>
#include <limits>

#include "CodeGen_Clockwork_Testbench.h"
#include "CodeGen_Internal.h"
#include "Substitute.h"
#include "IROperator.h"
#include "Param.h"
#include "Var.h"
#include "Lerp.h"
#include "Simplify.h"

namespace Halide {
namespace Internal {

using std::ostream;
using std::endl;
using std::string;
using std::vector;
using std::pair;
using std::map;

class Clockwork_Closure : public Closure {
public:
  Clockwork_Closure(Stmt s, std::string output_string)  {
        s.accept(this);
        output_name = output_string;
    }

    vector<Clockwork_Argument> arguments(const Scope<CodeGen_Clockwork_Base::Stencil_Type> &scope);

protected:
    using Closure::visit;
    std::string output_name;

};


vector<Clockwork_Argument> Clockwork_Closure::arguments(const Scope<CodeGen_Clockwork_Base::Stencil_Type> &streams_scope) {
    vector<Clockwork_Argument> res;
    for (const pair<string, Closure::Buffer> &i : buffers) {
        std::cout << "buffer: " << i.first << " " << i.second.size;
        if (i.second.read) std::cout << " (read)";
        if (i.second.write) std::cout << " (write)";
        std::cout << "\n";
        if (i.second.write) {
          res.push_back({i.first, true, true, Type(), CodeGen_Clockwork_Base::Stencil_Type()});
        }
        if (i.second.read) {
          res.push_back({i.first, true, false, Type(), CodeGen_Clockwork_Base::Stencil_Type()});
        }

    }
    //internal_assert(buffers.empty()) << "we expect no references to buffers in a hw pipeline.\n";
    for (const pair<string, Type> &i : vars) {
        std::cout << "var: " << i.first << "\n";
        if(ends_with(i.first, ".stream") ||
           ends_with(i.first, ".stencil") ) {
            CodeGen_Clockwork_Base::Stencil_Type stype = streams_scope.get(i.first);
            if (starts_with(i.first, output_name)) {
              res.push_back({i.first, true, true, Type(), stype});              
            } else {
              res.push_back({i.first, true, false, Type(), stype});              
            }

        } else if (ends_with(i.first, ".stencil_update")) {
            internal_error << "we don't expect to see a stencil_update type in Clockwork_Closure.\n";
        } else {
            // it is a scalar variable
          res.push_back({i.first, false, true, i.second, CodeGen_Clockwork_Base::Stencil_Type()});
        }
    }
    res.push_back({"hw_input", true, false, Type(), CodeGen_Clockwork_Base::Stencil_Type()});
    return res;
}

namespace {
    const string clockwork_headers =
        "#include <hls_stream.h>\n"
        "#include \"Stencil.h\"\n"
        "#include \"clockwork_target.h\"\n";
}



  CodeGen_Clockwork_Testbench::CodeGen_Clockwork_Testbench(ostream &tb_stream, Target target)
      : CodeGen_Clockwork_Base(tb_stream, target, CPlusPlusImplementation, ""),
  //: CodeGen_Clockwork_Base(std::cout, target, CPlusPlusImplementation, ""),
      cg_target("clockwork_target", target) {
    cg_target.init_module();

    stream << clockwork_headers;
}

CodeGen_Clockwork_Testbench::~CodeGen_Clockwork_Testbench() {
}

void CodeGen_Clockwork_Testbench::visit(const ProducerConsumer *op) {
    string target_prefix = "hw_output";
  //string target_prefix = "kernel";
    if (starts_with(op->name, target_prefix) && op->is_producer) {
      Stmt hw_body = op->body;
      std::cout << op->body;

      debug(1) << "compute the closure for " << op->name << '\n';
      std::cout << "compute the closure for " << op->name << '\n';
      string output_name = op->name.substr(target_prefix.length()); 
      Clockwork_Closure c(hw_body, output_name);
      vector<Clockwork_Argument> args = c.arguments(stencils);

      // generate HLS target code using the child code generator
      string ip_name = unique_name("clockwork_target");
      cg_target.add_kernel(hw_body, ip_name, args);

      // emits the target function call
      do_indent();
      stream << ip_name << "("; // avoid starting with '_'
      for(size_t i = 0; i < args.size(); i++) {
        stream << print_name(args[i].name);
        if(i != args.size() - 1)
          stream << ", ";
      }
      stream <<");\n";

    } else {
        CodeGen_Clockwork_Base::visit(op);
    }
}

void CodeGen_Clockwork_Testbench::visit(const Call *op) {
    if (op->name == "stream_subimage") {
        std::ostringstream rhs;
        // add intrinsic functions to convert memory buffers to streams
        // syntax:
        //   stream_subimage(direction, buffer_var, stream_var, address_of_subimage_origin,
        //                   dim_0_stride, dim_0_extent, ...)
        internal_assert(op->args.size() >= 6 && op->args.size() <= 12);
        const StringImm *direction = op->args[0].as<StringImm>();
        string a1 = print_expr(op->args[1]);
        string a2 = print_expr(op->args[2]);
        string a3 = print_expr(op->args[3]);
        if (direction->value == "buffer_to_stream") {
            rhs << "subimage_to_stream(";
        } else if (direction->value == "stream_to_buffer") {
            rhs << "stream_to_subimage(";
        } else {
            internal_error;
        }
        rhs << a1 << ", " << a2 << ", " << a3;
        for (size_t i = 4; i < op->args.size(); i++) {
            rhs << ", " << print_expr(op->args[i]);
        }
        rhs <<");\n";

        do_indent();
        stream << rhs.str();

        id = "0"; // skip evaluation
    } else if (op->name == "buffer_to_stencil") {
        internal_assert(op->args.size() == 2);
        // add a suffix to buffer var, in order to be compatible with CodeGen_C
        string a0 = print_expr(op->args[0]);
        string a1 = print_expr(op->args[1]);
        do_indent();
        stream << "buffer_to_stencil(" << a0 << ", " << a1 << ");\n";
        id = "0"; // skip evaluation
    } else if(op->name == "address_of") {
        std::ostringstream rhs;
        const Load *l = op->args[0].as<Load>();
        internal_assert(op->args.size() == 1 && l);
        rhs << "(("
            << print_type(l->type.element_of())
            << " *)"
            << print_name(l->name)
            << " + "
            << print_expr(l->index)
            << ")";
        print_assignment(op->type, rhs.str());
    } else {
        CodeGen_Clockwork_Base::visit(op);
    }
}

void CodeGen_Clockwork_Testbench::visit(const Realize *op) {
    if (ends_with(op->name, ".stream")) {
        // create a AXI stream type
        internal_assert(op->types.size() == 1);
        allocations.push(op->name, {op->types[0]});
        Stencil_Type stream_type({Stencil_Type::StencilContainerType::AxiStream,
                    op->types[0], op->bounds, 1});
        stencils.push(op->name, stream_type);

        // emits the declaration for the stream
        do_indent();
        stream << print_stencil_type(stream_type) << ' ' << print_name(op->name) << ";\n";
        stream << print_stencil_pragma(op->name);

        // traverse down
        op->body.accept(this);

        // We didn't generate free stmt inside for stream type
        allocations.pop(op->name);
        stencils.pop(op->name);
    } else {
        CodeGen_Clockwork_Base::visit(op);
    }
}

void CodeGen_Clockwork_Testbench::visit(const Block *op) {
    // emit stream_to_buffer call after the bulk of IR containing hardware pipeline
    // This is ugly right now, as the HLS simulation model and DMA programming model
    // are different on the order of pipeline IR and stream_to_buffer call..
    const Evaluate *eval = op->first.as<Evaluate>();
    if (!eval) {
        CodeGen_Clockwork_Base::visit(op);
        return;
    }
    const Call *call = eval->value.as<Call>();
    if (!call) {
        CodeGen_Clockwork_Base::visit(op);
        return;
    }
    if (call->name == "stream_subimage") {
        const StringImm *direction = call->args[0].as<StringImm>();
        if (direction->value == "stream_to_buffer") {
            internal_assert(op->rest.defined());
            op->rest.accept(this);
            op->first.accept(this);
            return;
        }
    }
    CodeGen_Clockwork_Base::visit(op);
    return;
}

}
}