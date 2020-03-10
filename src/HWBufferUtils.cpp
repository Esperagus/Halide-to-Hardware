#include "HWBufferUtils.h"

#include "Func.h"
#include "IRMutator.h"
#include "IRVisitor.h"
#include "Simplify.h"
#include "Substitute.h"

namespace Halide {
namespace Internal {

using std::map;
using std::vector;
using std::string;

std::ostream& operator<<(std::ostream& os, const std::vector<Expr>& vec) {
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

std::ostream& operator<<(std::ostream& os, const std::vector<string>& vec) {
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

int to_int(Expr expr) {
  //FIXMEyikes internal_assert(is_const(simplify(expr)));
  if (is_const(simplify(expr))) {
    return (int)*as_const_int(expr);
  } else {
    return -1;
  }
}

int id_const_value(const Expr e) {
  if (const IntImm* e_int = e.as<IntImm>()) {
    return e_int->value;

  } else if (const UIntImm* e_uint = e.as<UIntImm>()) {
    return e_uint->value;

  } else {
    return -1;
  }
}

std::vector<std::string> get_tokens(const std::string &line, const std::string &delimiter) {
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    std::string token;
    // copied from https://stackoverflow.com/a/7621814
    while ((pos = line.find_first_of(delimiter, prev)) != std::string::npos) {
        if (pos > prev) {
            tokens.emplace_back(line.substr(prev, pos - prev));
        }
        prev = pos + 1;
    }
    if (prev < line.length()) tokens.emplace_back(line.substr(prev, std::string::npos));
    // remove empty ones
    std::vector<std::string> result;
    result.reserve(tokens.size());
    for (auto const &t : tokens)
        if (!t.empty()) result.emplace_back(t);
    return result;
}

// return true if the for loop is not serialized
bool is_parallelized(const For *op) {
  return op->for_type == ForType::Parallel ||
    op->for_type == ForType::Unrolled ||
    op->for_type == ForType::Vectorized;
}

class FirstForName : public IRVisitor {
  using IRVisitor::visit;
  
  void visit(const For *op) {
    var = op->name;
  }
public:
  string var;
  FirstForName() {}
};

std::string first_for_name(Stmt s) {
  FirstForName ffn;
  s.accept(&ffn);
  return ffn.var;
}

class ContainsCall : public IRVisitor {
  string var;
  
  using IRVisitor::visit;
  
  void visit(const Call *op) {
    if (op->name == var) {
      found = true;
    } else {
      IRVisitor::visit(op);
    }
  }
public:
  bool found;
  ContainsCall(string var) : var(var), found(false) {}
};

bool contains_call(Stmt s, string var) {
  ContainsCall cc(var);
  s.accept(&cc);
  return cc.found;
}

class ExamineLoopLevel : public IRVisitor {
  using IRVisitor::visit;
  
  void visit(const For *op) {
    // don't recurse if not parallelized
    if (is_parallelized(op) || is_one(op->extent)) {
      IRVisitor::visit(op);
    }
  }

  void visit(const Provide *op) {
    if (var == "" || op->name == var) {
      found_provide = true;
    }
    IRVisitor::visit(op);
  }

  void visit(const Call *op) {
    if (var == "" || op->name == var) {
      //std::cout << "this is a call to " << var << Expr(op);
      found_call = true;
    }
    IRVisitor::visit(op);
  }

public:
  bool found_provide;
  bool found_call;
  string var;
  ExamineLoopLevel(string var) : found_provide(false), found_call(false), var(var) {}
};

// return if provide creates the var at this level (not recursing into serial loops)
bool provide_at_level(Stmt s, string var) {
  ExamineLoopLevel ell(var);
  s.accept(&ell);
  return ell.found_provide;
}

// return if call uses the var at this level (not recursing into serial loops)
bool call_at_level(Stmt s, string var) {
  ExamineLoopLevel ell(var);
  s.accept(&ell);
  return ell.found_call;
}
  
class ExpandExpr : public IRMutator {
    using IRMutator::visit;
    const Scope<Expr> &scope;
  
    Expr visit(const Variable *var) override {
        if (scope.contains(var->name)) {
            Expr expr = scope.get(var->name);
            debug(3) << "Fully expanded " << var->name << " -> " << expr << "\n";
            return expr;
        } else {
            return var;
        }
    }

    // remove for loops of length 1
    Stmt visit(const For *old_op) override {
      Stmt s = IRMutator::visit(old_op);
      const For *op = s.as<For>();
      
      if (is_one(op->extent)) {
        Stmt new_body = substitute(op->name, op->min, op->body);
        return new_body;
        
      } else {
        return op;
      }
    }

public:
    ExpandExpr(const Scope<Expr> &s) : scope(s) {}

};

// Perform all the substitutions in a scope
Expr expand_expr(Expr e, const Scope<Expr> &scope) {
    ExpandExpr ee(scope);
    Expr result = ee.mutate(e);
    debug(3) << "Expanded " << e << " into " << result << "\n";
    return result;
}

Stmt expand_expr(Stmt e, const Scope<Expr> &scope) {
    ExpandExpr ee(scope);
    Stmt result = ee.mutate(e);
    debug(3) << "Expanded " << e << " into " << result << "\n";
    return result;
}

class FindInnerLoops : public IRVisitor {
  Function func;

  LoopLevel outer_loop_exclusive;
  LoopLevel inner_loop_inclusive;
  bool in_inner_loops;

  using IRVisitor::visit;

  void visit(const ProducerConsumer *op) {
    // match store level at PC node in case the looplevel is outermost
    //std::cout << "looking at pc " << op->name << " where " << outer_loop_exclusive.lock().func()
    //          << outer_loop_exclusive.lock().var().name() << " and " << Var::outermost().name()
    //          << std::endl;
    if (outer_loop_exclusive.lock().func() == op->name && 
        outer_loop_exclusive.lock().var().name() == Var::outermost().name()) {
      in_inner_loops = true;
    }
    IRVisitor::visit(op);
  }

  void visit(const For *op) {
    // scan loops are loops between the outer store level (exclusive) and
    // the inner compute level (inclusive) of the accelerated function
    //std::cout << "considering loop " << op->name << " to inner loops. in=" << in_inner_loops << " sw=" << starts_with(op->name, func.name() + ".") << "\n";
    //if (in_inner_loops && starts_with(op->name, func.name() + ".")) {
    bool added_loop = false;
    if (in_inner_loops) {
      added_loop = true;
      debug(3) << "added loop " << op->name << " to inner loops.\n";
      inner_loops.emplace_back(op->name);
      //std::cout << "added loop to scan loop named " << op->name << " with extent=" << op->extent << std::endl;
    }

    // reevaluate in_scan_loops in terms of loop levels
    if (outer_loop_exclusive.match(op->name)) {
      in_inner_loops = true;
    }
    if (in_inner_loops && inner_loop_inclusive.match(op->name)) {
      returned_inner_loops = inner_loops;
      in_inner_loops = false;
      //std::cout << "updated inner loops: " << returned_inner_loops << std::endl;
    }

    // Recurse
    IRVisitor::visit(op);
    if (added_loop) {
      inner_loops.pop_back();
      //std::cout << "updated loops: " << inner_loops << std::endl;
      in_inner_loops = true;
    }
  }

public:  
  FindInnerLoops(Function f, LoopLevel outer_level, LoopLevel inner_level, bool inside)
    : func(f), outer_loop_exclusive(outer_level), inner_loop_inclusive(inner_level),
      in_inner_loops(inside) { }

  vector<string> inner_loops;
  vector<string> returned_inner_loops;
  
};

// Produce a vector of the loops within a for-loop-nest.
//   Note: this can be used to find the streaming loops between the store and compute level.
vector<string> get_loop_levels_between(Stmt s, Function func,
                                       LoopLevel outer_level_exclusive,
                                       LoopLevel inner_level_inclusive,
                                       bool start_inside) {
  FindInnerLoops fil(func, outer_level_exclusive, inner_level_inclusive, start_inside);
  //std::cout << "find some loops: " << s << std::endl;
  s.accept(&fil);
  //std::cout << "got some loops: " << fil.returned_inner_loops << std::endl;
  return fil.returned_inner_loops;
}

// Because storage folding runs before simplification, it's useful to
// at least substitute in constants before running it, and also simplify the RHS of Let Stmts.
class SubstituteInConstants : public IRMutator {
    using IRMutator::visit;

    Scope<Expr> scope;
    Stmt visit(const LetStmt *op) override {
        Expr value = simplify(mutate(op->value));

        Stmt body;
        if (is_const(value)) {
            ScopedBinding<Expr> bind(scope, op->name, value);
            body = mutate(op->body);
        } else {
            body = mutate(op->body);
        }

        if (body.same_as(op->body) && value.same_as(op->value)) {
            return op;
        } else {
            return LetStmt::make(op->name, value, body);
        }
    }

    Expr visit(const Variable *op) override {
        if (scope.contains(op->name)) {
            return scope.get(op->name);
        } else {
            return op;
        }
    }
};

Stmt substitute_in_constants(Stmt s) {
  SubstituteInConstants sic;
  Stmt subbed = sic.mutate(s);
  return subbed;
}

} // Internal
} // Halide
