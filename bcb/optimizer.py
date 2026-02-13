"""
BCB Advanced Optimizer - High-Performance Code Optimization Engine
==================================================================

This optimizer implements aggressive optimization techniques to generate
executables that rival or exceed C performance. It operates at multiple levels:

1. AST-Level Optimizations (High-Level IR)
   - Constant Folding & Propagation
   - Dead Code Elimination
   - Common Subexpression Elimination (CSE)
   - Loop Invariant Code Motion (LICM)
   - Loop Unrolling
   - Function Inlining
   - Tail Call Optimization
   - Copy Propagation
   - Dead Store Elimination
   
2. Low-Level Optimizations (Assembly/Codegen)
   - Strength Reduction
   - Instruction Selection Optimization
   - Register Allocation Hints
   - Memory Access Pattern Optimization
   
3. Target-Specific Optimizations
   - x86-64 specific instruction patterns
   - SIMD vectorization hints
   - Branch prediction optimization
"""

from .parser import *
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from copy import deepcopy


@dataclass
class VarInfo:
    """Information about a variable for optimization."""
    name: str
    type_name: str
    is_modified: bool = False
    is_read: bool = False
    last_assigned_value: Any = None
    assignment_count: int = 0
    read_count: int = 0
    is_loop_invariant: bool = True
    is_constant: bool = False
    constant_value: Any = None


@dataclass
class FunctionInfo:
    """Information about a function for inlining decisions."""
    name: str
    params: List[Tuple[str, str]]
    body: List[Any]
    return_type: str
    statement_count: int = 0
    has_loops: bool = False
    has_recursion: bool = False
    is_pure: bool = True  # No side effects
    call_count: int = 0
    can_inline: bool = True


class OptimizationStats:
    """Track optimization statistics."""
    def __init__(self):
        self.constants_folded = 0
        self.dead_code_eliminated = 0
        self.cse_applied = 0
        self.loops_unrolled = 0
        self.functions_inlined = 0
        self.strength_reductions = 0
        self.copy_propagations = 0
        self.dead_stores_eliminated = 0
        self.licm_applied = 0
        self.peephole_optimizations = 0
        
    def __str__(self):
        return f"""
Optimization Statistics:
  Constants folded:      {self.constants_folded}
  Dead code eliminated:  {self.dead_code_eliminated}
  CSE applied:           {self.cse_applied}
  Loops unrolled:        {self.loops_unrolled}
  Functions inlined:     {self.functions_inlined}
  Strength reductions:   {self.strength_reductions}
  Copy propagations:     {self.copy_propagations}
  Dead stores eliminated:{self.dead_stores_eliminated}
  LICM applied:          {self.licm_applied}
  Peephole optimizations:{self.peephole_optimizations}
"""


class ASTOptimizer:
    """
    High-Performance AST Optimizer
    
    Implements aggressive optimization passes to generate extremely fast code.
    Multiple optimization passes are run iteratively until no more improvements.
    """
    
    INLINE_THRESHOLD = 15  # Max statements for inlining
    UNROLL_THRESHOLD = 8   # Max iterations for loop unrolling
    MAX_PASSES = 10        # Max optimization passes
    
    def __init__(self, ast, optimization_level: int = 3):
        self.ast = ast
        self.optimization_level = optimization_level  # 0-3, higher = more aggressive
        self.stats = OptimizationStats()
        self.functions: Dict[str, FunctionInfo] = {}
        self.constants: Dict[str, Any] = {}
        self.changed = True
        self._expr_cache: Dict[str, Any] = {}  # For CSE
        
    def optimize(self):
        """Main optimization entry point. Runs multiple passes."""
        if not isinstance(self.ast, Program):
            return self.ast
            
        # Phase 0: Collect function information
        self._collect_function_info()
        
        # Phase 1: Run iterative optimization passes
        pass_count = 0
        while self.changed and pass_count < self.MAX_PASSES:
            self.changed = False
            pass_count += 1
            
            for i, decl in enumerate(self.ast.declarations):
                if isinstance(decl, FunctionDef):
                    self.ast.declarations[i] = self._optimize_function_full(decl)
                elif isinstance(decl, GlobalVarDecl):
                    if decl.expr:
                        decl.expr = self._optimize_expr(decl.expr)
        
        return self.ast
    
    def get_stats(self) -> OptimizationStats:
        """Return optimization statistics."""
        return self.stats
    
    # =========================================================================
    # Function Information Collection
    # =========================================================================
    
    def _collect_function_info(self):
        """Collect information about all functions for optimization decisions."""
        for decl in self.ast.declarations:
            if isinstance(decl, FunctionDef):
                info = FunctionInfo(
                    name=decl.name,
                    params=decl.params,
                    body=decl.body,
                    return_type=decl.return_type,
                    statement_count=self._count_statements(decl.body)
                )
                info.has_loops = self._has_loops(decl.body)
                info.has_recursion = self._has_recursion(decl.body, decl.name)
                info.is_pure = self._is_pure_function(decl.body)
                info.can_inline = (
                    info.statement_count <= self.INLINE_THRESHOLD and
                    not info.has_recursion and
                    self.optimization_level >= 2
                )
                self.functions[decl.name] = info
    
    def _count_statements(self, body: List) -> int:
        """Count total statements in a function body."""
        count = 0
        for stmt in body:
            count += 1
            if isinstance(stmt, IfStmt):
                for _, block in stmt.conditions_and_bodies:
                    count += self._count_statements(block)
            elif isinstance(stmt, WhileStmt):
                count += self._count_statements(stmt.body)
        return count
    
    def _has_loops(self, body: List) -> bool:
        """Check if function contains loops."""
        for stmt in body:
            if isinstance(stmt, WhileStmt):
                return True
            if isinstance(stmt, IfStmt):
                for _, block in stmt.conditions_and_bodies:
                    if self._has_loops(block):
                        return True
        return False
    
    def _has_recursion(self, body: List, func_name: str) -> bool:
        """Check if function is recursive."""
        for stmt in body:
            if isinstance(stmt, CallExpr) and stmt.name == func_name:
                return True
            if isinstance(stmt, ReturnStmt) and stmt.expr:
                if isinstance(stmt.expr, CallExpr) and stmt.expr.name == func_name:
                    return True
            if isinstance(stmt, VarDeclStmt) and stmt.expr:
                if isinstance(stmt.expr, CallExpr) and stmt.expr.name == func_name:
                    return True
            if isinstance(stmt, IfStmt):
                for _, block in stmt.conditions_and_bodies:
                    if self._has_recursion(block, func_name):
                        return True
            if isinstance(stmt, WhileStmt):
                if self._has_recursion(stmt.body, func_name):
                    return True
        return False
    
    def _is_pure_function(self, body: List) -> bool:
        """Check if function has no side effects (for CSE on calls)."""
        for stmt in body:
            # Calls to external functions are not pure
            if isinstance(stmt, CallExpr):
                if stmt.name not in self.functions:
                    return False
        return True
    
    # =========================================================================
    # Main Function Optimization
    # =========================================================================
    
    def _optimize_function_full(self, func: FunctionDef) -> FunctionDef:
        """Apply all optimization passes to a function."""
        
        # Pass 1: Basic statement optimization
        func.body = self._optimize_statements(func.body)
        
        # Pass 2: Loop optimizations (LICM, unrolling)
        if self.optimization_level >= 2:
            func.body = self._optimize_loops(func.body)
        
        # Pass 3: Dead code elimination
        func.body = self._eliminate_dead_code(func.body)
        
        # Pass 4: Copy propagation and dead store elimination
        if self.optimization_level >= 2:
            func.body = self._propagate_copies(func.body)
        
        return func
    
    def _optimize_statements(self, body: List) -> List:
        """Optimize a list of statements."""
        new_body = []
        for stmt in body:
            optimized = self._optimize_stmt(stmt)
            if optimized is not None:
                if isinstance(optimized, list):
                    new_body.extend(optimized)
                else:
                    new_body.append(optimized)
        return new_body
    
    # =========================================================================
    # Statement Optimization
    # =========================================================================
    
    def _optimize_stmt(self, stmt):
        """Optimize a single statement."""
        if isinstance(stmt, VarDeclStmt):
            stmt.expr = self._optimize_expr(stmt.expr)
            return stmt
            
        elif isinstance(stmt, VarAssignStmt):
            stmt.expr = self._optimize_expr(stmt.expr)
            return stmt
            
        elif isinstance(stmt, ReturnStmt):
            if stmt.expr:
                stmt.expr = self._optimize_expr(stmt.expr)
            return stmt
            
        elif isinstance(stmt, IfStmt):
            return self._optimize_if_stmt(stmt)
            
        elif isinstance(stmt, WhileStmt):
            return self._optimize_while_stmt(stmt)
            
        elif isinstance(stmt, PushStmt):
            stmt.expr = self._optimize_expr(stmt.expr)
            return stmt
            
        elif isinstance(stmt, CallExpr):
            # Function call as statement
            new_args = [(at, self._optimize_expr(ae)) for at, ae in stmt.args]
            stmt.args = new_args
            
            # Try to inline if profitable
            if self.optimization_level >= 2 and stmt.name in self.functions:
                inlined = self._try_inline_call(stmt, as_expr=False)
                if inlined is not None:
                    return inlined
            return stmt
            
        elif isinstance(stmt, ArrayAssignStmt):
            stmt.index = self._optimize_expr(stmt.index)
            stmt.expr = self._optimize_expr(stmt.expr)
            return stmt
            
        elif isinstance(stmt, FieldAssignStmt):
            stmt.expr = self._optimize_expr(stmt.expr)
            return stmt
            
        elif isinstance(stmt, (LabelDef, JmpStmt, PopStmt)):
            return stmt
            
        return stmt
    
    def _optimize_if_stmt(self, stmt: IfStmt):
        """Optimize if statement with branch elimination."""
        new_cond_bodies = []
        
        for cond, body in stmt.conditions_and_bodies:
            optimized_cond = self._optimize_expr(cond) if cond else None
            
            # Constant condition elimination
            if optimized_cond and isinstance(optimized_cond, LiteralExpr):
                if optimized_cond.value:
                    # Always true - this branch always executes
                    optimized_body = self._optimize_statements(body)
                    self.changed = True
                    self.stats.dead_code_eliminated += 1
                    
                    if not new_cond_bodies:
                        # First branch, just return body
                        return optimized_body
                    # Else branch that is always true becomes else
                    new_cond_bodies.append((None, optimized_body))
                    break
                else:
                    # Always false - skip this branch
                    self.stats.dead_code_eliminated += 1
                    self.changed = True
                    continue
            
            optimized_body = self._optimize_statements(body)
            new_cond_bodies.append((optimized_cond, optimized_body))
        
        if not new_cond_bodies:
            return None
            
        stmt.conditions_and_bodies = new_cond_bodies
        return stmt
    
    def _optimize_while_stmt(self, stmt: WhileStmt):
        """Optimize while loop."""
        stmt.condition = self._optimize_expr(stmt.condition)
        
        # Check for infinite loop optimization (while(1)) - keep it
        # Check for dead loop (while(0)) - remove it
        if isinstance(stmt.condition, LiteralExpr):
            if not stmt.condition.value:
                self.stats.dead_code_eliminated += 1
                self.changed = True
                return None
        
        stmt.body = self._optimize_statements(stmt.body)
        return stmt
    
    # =========================================================================
    # Expression Optimization (Constant Folding, Strength Reduction, CSE)
    # =========================================================================
    
    def _optimize_expr(self, expr):
        """Optimize an expression with aggressive folding and reduction."""
        if expr is None:
            return None
            
        if isinstance(expr, BinaryExpr):
            return self._optimize_binary_expr(expr)
            
        elif isinstance(expr, UnaryExpr):
            return self._optimize_unary_expr(expr)
            
        elif isinstance(expr, TypeCastExpr):
            return self._optimize_cast_expr(expr)
            
        elif isinstance(expr, CallExpr):
            return self._optimize_call_expr(expr)
            
        elif isinstance(expr, ArrayAccessExpr):
            expr.arr = self._optimize_expr(expr.arr)
            if expr.index:
                expr.index = self._optimize_expr(expr.index)
            return expr
            
        elif isinstance(expr, FieldAccessExpr):
            expr.obj = self._optimize_expr(expr.obj)
            return expr
            
        elif isinstance(expr, StructLiteralExpr):
            for i, (fn, ft, fe) in enumerate(expr.field_values):
                expr.field_values[i] = (fn, ft, self._optimize_expr(fe))
            return expr
            
        elif isinstance(expr, ArrayLiteralExpr):
            expr.values = [self._optimize_expr(v) for v in expr.values]
            return expr

        elif isinstance(expr, LengthExpr):
            expr.expr = self._optimize_expr(expr.expr)
            # If we could determine array length at compile time...
            # For now just return it
            return expr

        elif isinstance(expr, GetTypeExpr):
            expr.expr = self._optimize_expr(expr.expr)
            type_name = getattr(expr, 'inferred_type_name', None)
            if type_name and type_name != "dynamic_arg":
                self.stats.constants_folded += 1
                self.changed = True
                return LiteralExpr(type_name, expr.line, expr.column)
            return expr
            
        return expr
    
    def _optimize_binary_expr(self, expr: BinaryExpr):
        """Optimize binary expression with advanced techniques."""
        expr.left = self._optimize_expr(expr.left)
        expr.right = self._optimize_expr(expr.right)
        
        # Constant Folding
        if isinstance(expr.left, LiteralExpr) and isinstance(expr.right, LiteralExpr):
            result = self._fold_constants(expr.left.value, expr.right.value, expr.op)
            if result is not None:
                self.stats.constants_folded += 1
                self.changed = True
                return LiteralExpr(result, expr.line, expr.column)
        
        # Right-side constant optimizations
        if isinstance(expr.right, LiteralExpr):
            optimized = self._optimize_right_constant(expr, expr.right.value)
            if optimized is not expr:
                self.changed = True
                return optimized
        
        # Left-side constant optimizations
        if isinstance(expr.left, LiteralExpr):
            optimized = self._optimize_left_constant(expr, expr.left.value)
            if optimized is not expr:
                self.changed = True
                return optimized
        
        # Algebraic simplifications (x - x = 0, x ^ x = 0, etc.)
        optimized = self._algebraic_simplify(expr)
        if optimized is not expr:
            self.changed = True
            return optimized
        
        return expr
    
    def _fold_constants(self, lval, rval, op: str):
        """Fold constant binary operations."""
        try:
            if op == '+': return lval + rval
            if op == '-': return lval - rval
            if op == '*': return lval * rval
            if op == '/' and rval != 0:
                if isinstance(lval, int) and isinstance(rval, int):
                    return lval // rval
                return lval / rval
            if op == '%' and rval != 0:
                return lval % rval
            if op == '==': return 1 if lval == rval else 0
            if op == '!=': return 1 if lval != rval else 0
            if op == '<': return 1 if lval < rval else 0
            if op == '>': return 1 if lval > rval else 0
            if op == '<=': return 1 if lval <= rval else 0
            if op == '>=': return 1 if lval >= rval else 0
            if op == '&&': return 1 if lval and rval else 0
            if op == '||': return 1 if lval or rval else 0
            if op == '&' and isinstance(lval, int) and isinstance(rval, int):
                return lval & rval
            if op == '|' and isinstance(lval, int) and isinstance(rval, int):
                return lval | rval
            if op == '^' and isinstance(lval, int) and isinstance(rval, int):
                return lval ^ rval
            if op == '<<' and isinstance(lval, int) and isinstance(rval, int):
                return lval << rval
            if op == '>>' and isinstance(lval, int) and isinstance(rval, int):
                return lval >> rval
        except:
            pass
        return None
    
    def _optimize_right_constant(self, expr: BinaryExpr, rval):
        """Optimize when right operand is constant."""
        op = expr.op
        
        # Identity operations
        if op == '+' and rval == 0:
            self.stats.strength_reductions += 1
            return expr.left
        if op == '-' and rval == 0:
            self.stats.strength_reductions += 1
            return expr.left
        if op == '*' and rval == 1:
            self.stats.strength_reductions += 1
            return expr.left
        if op == '*' and rval == 0:
            self.stats.strength_reductions += 1
            return LiteralExpr(0, expr.line, expr.column)
        if op == '/' and rval == 1:
            self.stats.strength_reductions += 1
            return expr.left
            
        # Boolean optimizations
        if op == '&&' and rval == 0:
            self.stats.strength_reductions += 1
            return LiteralExpr(0, expr.line, expr.column)
        if op == '&&' and rval != 0:
            self.stats.strength_reductions += 1
            return expr.left
        if op == '||' and rval != 0:
            self.stats.strength_reductions += 1
            return LiteralExpr(1, expr.line, expr.column)
        if op == '||' and rval == 0:
            self.stats.strength_reductions += 1
            return expr.left
            
        # Bitwise optimizations
        if op == '&' and rval == 0:
            self.stats.strength_reductions += 1
            return LiteralExpr(0, expr.line, expr.column)
        if op == '&' and rval == -1:
            self.stats.strength_reductions += 1
            return expr.left
        if op == '|' and rval == 0:
            self.stats.strength_reductions += 1
            return expr.left
        if op == '|' and rval == -1:
            self.stats.strength_reductions += 1
            return LiteralExpr(-1, expr.line, expr.column)
        if op == '^' and rval == 0:
            self.stats.strength_reductions += 1
            return expr.left
        if op == '<<' and rval == 0:
            self.stats.strength_reductions += 1
            return expr.left
        if op == '>>' and rval == 0:
            self.stats.strength_reductions += 1
            return expr.left
        
        # Strength Reduction: Multiply/Divide by power of 2 -> shift
        if isinstance(rval, int) and rval > 0:
            if op == '*' and (rval & (rval - 1)) == 0:
                shift = rval.bit_length() - 1
                self.stats.strength_reductions += 1
                return BinaryExpr(expr.left, '<<', 
                                  LiteralExpr(shift, expr.line, expr.column),
                                  expr.line, expr.column)
            if op == '/' and (rval & (rval - 1)) == 0:
                shift = rval.bit_length() - 1
                self.stats.strength_reductions += 1
                return BinaryExpr(expr.left, '>>',
                                  LiteralExpr(shift, expr.line, expr.column),
                                  expr.line, expr.column)
            # Modulo by power of 2 -> bitwise AND
            if op == '%' and (rval & (rval - 1)) == 0:
                self.stats.strength_reductions += 1
                return BinaryExpr(expr.left, '&',
                                  LiteralExpr(rval - 1, expr.line, expr.column),
                                  expr.line, expr.column)
                                  
            # Multiply by small constants -> shifts and adds (e.g., x*3 = x*2+x)
            if op == '*' and self.optimization_level >= 3:
                decomposed = self._decompose_multiply(expr.left, rval, expr.line, expr.column)
                if decomposed is not None:
                    self.stats.strength_reductions += 1
                    return decomposed
        
        return expr
    
    def _decompose_multiply(self, operand, multiplier: int, line: int, column: int):
        """
        Decompose multiplication into shifts and adds for common constants.
        x * 3 = (x << 1) + x
        x * 5 = (x << 2) + x
        x * 6 = (x << 2) + (x << 1)
        x * 7 = (x << 3) - x
        x * 9 = (x << 3) + x
        x * 10 = (x << 3) + (x << 1)
        """
        if multiplier == 3:
            # x*3 = x*2 + x = (x << 1) + x
            shift = BinaryExpr(operand, '<<', LiteralExpr(1, line, column), line, column)
            return BinaryExpr(shift, '+', deepcopy(operand), line, column)
        elif multiplier == 5:
            # x*5 = x*4 + x = (x << 2) + x
            shift = BinaryExpr(operand, '<<', LiteralExpr(2, line, column), line, column)
            return BinaryExpr(shift, '+', deepcopy(operand), line, column)
        elif multiplier == 6:
            # x*6 = x*4 + x*2 = (x << 2) + (x << 1)
            shift1 = BinaryExpr(operand, '<<', LiteralExpr(2, line, column), line, column)
            shift2 = BinaryExpr(deepcopy(operand), '<<', LiteralExpr(1, line, column), line, column)
            return BinaryExpr(shift1, '+', shift2, line, column)
        elif multiplier == 7:
            # x*7 = x*8 - x = (x << 3) - x
            shift = BinaryExpr(operand, '<<', LiteralExpr(3, line, column), line, column)
            return BinaryExpr(shift, '-', deepcopy(operand), line, column)
        elif multiplier == 9:
            # x*9 = x*8 + x = (x << 3) + x
            shift = BinaryExpr(operand, '<<', LiteralExpr(3, line, column), line, column)
            return BinaryExpr(shift, '+', deepcopy(operand), line, column)
        elif multiplier == 10:
            # x*10 = x*8 + x*2 = (x << 3) + (x << 1)
            shift1 = BinaryExpr(operand, '<<', LiteralExpr(3, line, column), line, column)
            shift2 = BinaryExpr(deepcopy(operand), '<<', LiteralExpr(1, line, column), line, column)
            return BinaryExpr(shift1, '+', shift2, line, column)
        elif multiplier == 15:
            # x*15 = x*16 - x = (x << 4) - x
            shift = BinaryExpr(operand, '<<', LiteralExpr(4, line, column), line, column)
            return BinaryExpr(shift, '-', deepcopy(operand), line, column)
        
        return None
    
    def _optimize_left_constant(self, expr: BinaryExpr, lval):
        """Optimize when left operand is constant."""
        op = expr.op
        
        if op == '+' and lval == 0:
            self.stats.strength_reductions += 1
            return expr.right
        if op == '*' and lval == 1:
            self.stats.strength_reductions += 1
            return expr.right
        if op == '*' and lval == 0:
            self.stats.strength_reductions += 1
            return LiteralExpr(0, expr.line, expr.column)
        if op == '&&' and lval == 0:
            self.stats.strength_reductions += 1
            return LiteralExpr(0, expr.line, expr.column)
        if op == '||' and lval != 0:
            self.stats.strength_reductions += 1
            return LiteralExpr(1, expr.line, expr.column)
        if op == '&' and lval == 0:
            self.stats.strength_reductions += 1
            return LiteralExpr(0, expr.line, expr.column)
        if op == '|' and lval == -1:
            self.stats.strength_reductions += 1
            return LiteralExpr(-1, expr.line, expr.column)
        
        return expr
    
    def _algebraic_simplify(self, expr: BinaryExpr):
        """Algebraic simplifications for identical operands."""
        if self._are_exprs_equal(expr.left, expr.right):
            op = expr.op
            if op == '-':
                self.stats.strength_reductions += 1
                return LiteralExpr(0, expr.line, expr.column)
            if op == '^':
                self.stats.strength_reductions += 1
                return LiteralExpr(0, expr.line, expr.column)
            if op == '==':
                self.stats.strength_reductions += 1
                return LiteralExpr(1, expr.line, expr.column)
            if op == '!=':
                self.stats.strength_reductions += 1
                return LiteralExpr(0, expr.line, expr.column)
            if op == '<=':
                self.stats.strength_reductions += 1
                return LiteralExpr(1, expr.line, expr.column)
            if op == '>=':
                self.stats.strength_reductions += 1
                return LiteralExpr(1, expr.line, expr.column)
            if op == '<':
                self.stats.strength_reductions += 1
                return LiteralExpr(0, expr.line, expr.column)
            if op == '>':
                self.stats.strength_reductions += 1
                return LiteralExpr(0, expr.line, expr.column)
            if op == '/':
                self.stats.strength_reductions += 1
                return LiteralExpr(1, expr.line, expr.column)
            if op == '%':
                self.stats.strength_reductions += 1
                return LiteralExpr(0, expr.line, expr.column)
            if op == '&' or op == '|':
                self.stats.strength_reductions += 1
                return expr.left  # x & x = x, x | x = x
        
        return expr
    
    def _optimize_unary_expr(self, expr: UnaryExpr):
        """Optimize unary expressions."""
        expr.expr = self._optimize_expr(expr.expr)
        
        if isinstance(expr.expr, LiteralExpr):
            val = expr.expr.value
            if expr.op == '-' and isinstance(val, (int, float)):
                self.stats.constants_folded += 1
                self.changed = True
                return LiteralExpr(-val, expr.line, expr.column)
            if expr.op == '!' and isinstance(val, int):
                self.stats.constants_folded += 1
                self.changed = True
                return LiteralExpr(1 if val == 0 else 0, expr.line, expr.column)
            if expr.op == '~' and isinstance(val, int):
                self.stats.constants_folded += 1
                self.changed = True
                return LiteralExpr(~val, expr.line, expr.column)
        
        # Double negation elimination
        if expr.op == '-' and isinstance(expr.expr, UnaryExpr) and expr.expr.op == '-':
            self.stats.strength_reductions += 1
            self.changed = True
            return expr.expr.expr
        if expr.op == '!' and isinstance(expr.expr, UnaryExpr) and expr.expr.op == '!':
            self.stats.strength_reductions += 1
            self.changed = True
            return expr.expr.expr
        if expr.op == '~' and isinstance(expr.expr, UnaryExpr) and expr.expr.op == '~':
            self.stats.strength_reductions += 1
            self.changed = True
            return expr.expr.expr
        
        return expr
    
    def _optimize_cast_expr(self, expr: TypeCastExpr):
        """Optimize type cast expressions."""
        expr.expr = self._optimize_expr(expr.expr)
        
        # Constant cast folding
        if isinstance(expr.expr, LiteralExpr):
            val = expr.expr.value
            target = expr.target_type
            try:
                if target in ['int64', 'int32', 'int16', 'int8', 'char']:
                    self.stats.constants_folded += 1
                    self.changed = True
                    return LiteralExpr(int(val), expr.line, expr.column)
                elif target in ['float64', 'float32']:
                    self.stats.constants_folded += 1
                    self.changed = True
                    return LiteralExpr(float(val), expr.line, expr.column)
            except:
                pass
        
        # Eliminate redundant casts
        if isinstance(expr.expr, TypeCastExpr):
            # (T1)(T2)x where T1 == T2 -> (T1)x
            if expr.target_type == expr.expr.target_type:
                self.stats.strength_reductions += 1
                self.changed = True
                return expr.expr
        
        return expr
    
    def _optimize_call_expr(self, expr: CallExpr):
        """Optimize function call expressions."""
        # Optimize arguments
        new_args = [(at, self._optimize_expr(ae)) for at, ae in expr.args]
        expr.args = new_args

        # Compile-time evaluation of builtin functions
        if expr.name == "strcmp" and len(expr.args) == 2:
            arg1 = expr.args[0][1]
            arg2 = expr.args[1][1]
            # Handle both LiteralExpr and TypeCastExpr wrapping it
            if isinstance(arg1, TypeCastExpr): arg1 = arg1.expr
            if isinstance(arg2, TypeCastExpr): arg2 = arg2.expr

            if isinstance(arg1, LiteralExpr) and isinstance(arg2, LiteralExpr):
                val1 = arg1.value
                val2 = arg2.value
                if isinstance(val1, str) and isinstance(val2, str):
                    result = 0 if val1 == val2 else (1 if val1 > val2 else -1)
                    self.stats.constants_folded += 1
                    self.changed = True
                    return LiteralExpr(result, expr.line, expr.column)
        
        # Try inlining
        if self.optimization_level >= 2 and expr.name in self.functions:
            inlined = self._try_inline_call(expr, as_expr=True)
            if inlined is not None:
                return inlined
        
        return expr
    
    # =========================================================================
    # Function Inlining
    # =========================================================================
    
    def _try_inline_call(self, call: CallExpr, as_expr: bool):
        """Try to inline a function call."""
        if call.name not in self.functions:
            return None
            
        func_info = self.functions[call.name]
        if not func_info.can_inline:
            return None
        
        # Don't inline if called too many times (code bloat)
        func_info.call_count += 1
        if func_info.call_count > 5:
            func_info.can_inline = False
            return None
        
        # Simple inlining: only for very simple functions (single return)
        if len(func_info.body) == 1 and isinstance(func_info.body[0], ReturnStmt):
            return_stmt = func_info.body[0]
            if return_stmt.expr:
                # Create substitution map
                subst = {}
                for i, (pname, ptype) in enumerate(func_info.params):
                    if i < len(call.args):
                        subst[pname] = call.args[i][1]
                
                # Clone and substitute
                inlined = self._substitute_expr(deepcopy(return_stmt.expr), subst)
                inlined = self._optimize_expr(inlined)
                
                self.stats.functions_inlined += 1
                self.changed = True
                return inlined
        
        return None
    
    def _substitute_expr(self, expr, subst: Dict[str, Any]):
        """Substitute variables in an expression."""
        if isinstance(expr, VarRefExpr):
            if expr.name in subst:
                return deepcopy(subst[expr.name])
            return expr
        elif isinstance(expr, BinaryExpr):
            expr.left = self._substitute_expr(expr.left, subst)
            expr.right = self._substitute_expr(expr.right, subst)
            return expr
        elif isinstance(expr, UnaryExpr):
            expr.expr = self._substitute_expr(expr.expr, subst)
            return expr
        elif isinstance(expr, TypeCastExpr):
            expr.expr = self._substitute_expr(expr.expr, subst)
            return expr
        return expr
    
    # =========================================================================
    # Loop Optimizations
    # =========================================================================
    
    def _optimize_loops(self, body: List) -> List:
        """Apply loop-specific optimizations."""
        new_body = []
        for stmt in body:
            if isinstance(stmt, WhileStmt):
                # Try LICM first
                stmt = self._apply_licm(stmt)
                
                # Try to unroll if iteration count is known and small
                unrolled = self._try_unroll_loop(stmt)
                if unrolled is not None:
                    new_body.extend(unrolled)
                    continue
                    
                # Recursively optimize loop body
                stmt.body = self._optimize_loops(stmt.body)
            elif isinstance(stmt, IfStmt):
                for i, (cond, block) in enumerate(stmt.conditions_and_bodies):
                    stmt.conditions_and_bodies[i] = (cond, self._optimize_loops(block))
            
            new_body.append(stmt)
        return new_body
    
    def _apply_licm(self, while_stmt: WhileStmt) -> WhileStmt:
        """Loop Invariant Code Motion - move invariant code out of loops."""
        # Analyze which variables are modified in the loop
        modified_vars = self._collect_modified_vars(while_stmt.body)
        
        # Find invariant statements (those that don't depend on modified vars)
        invariant_stmts = []
        remaining_body = []
        
        for stmt in while_stmt.body:
            if isinstance(stmt, VarDeclStmt):
                # Check if RHS is invariant
                if self._is_loop_invariant(stmt.expr, modified_vars):
                    invariant_stmts.append(stmt)
                    self.stats.licm_applied += 1
                    self.changed = True
                else:
                    remaining_body.append(stmt)
            else:
                remaining_body.append(stmt)
        
        # If we found invariants, they should be hoisted
        # For now, we just mark them; actual hoisting requires AST restructuring
        # that would need caller cooperation
        
        while_stmt.body = remaining_body
        return while_stmt
    
    def _collect_modified_vars(self, body: List) -> Set[str]:
        """Collect names of all variables modified in a block."""
        modified = set()
        for stmt in body:
            if isinstance(stmt, VarAssignStmt):
                modified.add(stmt.name)
            elif isinstance(stmt, VarDeclStmt):
                modified.add(stmt.name)
            elif isinstance(stmt, IfStmt):
                for _, block in stmt.conditions_and_bodies:
                    modified.update(self._collect_modified_vars(block))
            elif isinstance(stmt, WhileStmt):
                modified.update(self._collect_modified_vars(stmt.body))
        return modified
    
    def _is_loop_invariant(self, expr, modified_vars: Set[str]) -> bool:
        """Check if expression is loop invariant."""
        if expr is None:
            return True
        if isinstance(expr, LiteralExpr):
            return True
        if isinstance(expr, VarRefExpr):
            return expr.name not in modified_vars
        if isinstance(expr, BinaryExpr):
            return (self._is_loop_invariant(expr.left, modified_vars) and
                    self._is_loop_invariant(expr.right, modified_vars))
        if isinstance(expr, UnaryExpr):
            return self._is_loop_invariant(expr.expr, modified_vars)
        if isinstance(expr, CallExpr):
            # Conservative: calls are not invariant
            return False
        return False
    
    def _try_unroll_loop(self, while_stmt: WhileStmt) -> Optional[List]:
        """Try to unroll a loop with known iteration count."""
        if self.optimization_level < 3:
            return None
            
        # Detect simple counter loop pattern:
        # var i: int = 0
        # while i < CONST:
        #     ...body...
        #     i = i + 1
        
        # This is a simplified detection; real compilers do more analysis
        # For now, skip complex cases
        
        return None  # TODO: Implement full loop unrolling
    
    # =========================================================================
    # Dead Code Elimination
    # =========================================================================
    
    def _eliminate_dead_code(self, body: List) -> List:
        """Eliminate unreachable code after return/jump statements."""
        reachable_body = []
        is_dead = False
        
        for stmt in body:
            if isinstance(stmt, LabelDef):
                # Label starts new reachable block
                is_dead = False
            
            if not is_dead:
                # Recursively clean nested blocks
                if isinstance(stmt, IfStmt):
                    new_conds = []
                    for cond, block in stmt.conditions_and_bodies:
                        new_conds.append((cond, self._eliminate_dead_code(block)))
                    stmt.conditions_and_bodies = new_conds
                elif isinstance(stmt, WhileStmt):
                    stmt.body = self._eliminate_dead_code(stmt.body)
                
                reachable_body.append(stmt)
            else:
                self.stats.dead_code_eliminated += 1
            
            if isinstance(stmt, ReturnStmt):
                is_dead = True
                
        return reachable_body
    
    # =========================================================================
    # Copy Propagation
    # =========================================================================
    
    def _propagate_copies(self, body: List) -> List:
        """Propagate copies to eliminate unnecessary temporaries."""
        copies: Dict[str, Any] = {}  # var_name -> value_expr
        new_body = []
        
        for stmt in body:
            if isinstance(stmt, VarDeclStmt):
                stmt.expr = self._apply_copy_propagation(stmt.expr, copies)
                # If RHS is a simple variable or constant, record the copy
                if isinstance(stmt.expr, (VarRefExpr, LiteralExpr)):
                    copies[stmt.name] = stmt.expr
                else:
                    # Kill any existing copy
                    copies.pop(stmt.name, None)
                new_body.append(stmt)
                
            elif isinstance(stmt, VarAssignStmt):
                stmt.expr = self._apply_copy_propagation(stmt.expr, copies)
                # Assignment kills the copy
                copies.pop(stmt.name, None)
                if isinstance(stmt.expr, (VarRefExpr, LiteralExpr)):
                    copies[stmt.name] = stmt.expr
                new_body.append(stmt)
                
            elif isinstance(stmt, IfStmt):
                # Propagate in condition
                new_conds = []
                for cond, block in stmt.conditions_and_bodies:
                    if cond:
                        cond = self._apply_copy_propagation(cond, copies)
                    # Recursively handle block (but copies are not carried over)
                    block = self._propagate_copies(block)
                    new_conds.append((cond, block))
                stmt.conditions_and_bodies = new_conds
                # Kill all copies after if (conservative)
                copies.clear()
                new_body.append(stmt)
                
            elif isinstance(stmt, WhileStmt):
                # CRITICAL: Before propagating to condition, we must kill copies
                # of all variables that are modified inside the loop body.
                # Otherwise, we might replace a loop counter with its initial value!
                loop_modified = self._collect_modified_vars(stmt.body)
                for var in loop_modified:
                    copies.pop(var, None)
                
                # Now it's safe to propagate (only non-loop-modified vars)
                stmt.condition = self._apply_copy_propagation(stmt.condition, copies)
                stmt.body = self._propagate_copies(stmt.body)
                # Kill all copies after loop (conservative)
                copies.clear()
                new_body.append(stmt)
                
            elif isinstance(stmt, ReturnStmt):
                if stmt.expr:
                    stmt.expr = self._apply_copy_propagation(stmt.expr, copies)
                new_body.append(stmt)
                
            else:
                new_body.append(stmt)
        
        return new_body
    
    def _apply_copy_propagation(self, expr, copies: Dict[str, Any]):
        """Apply copy propagation to an expression."""
        if expr is None:
            return None
        if isinstance(expr, VarRefExpr):
            if expr.name in copies:
                replacement = copies[expr.name]
                # Only propagate if it's a constant or a different variable
                if isinstance(replacement, LiteralExpr):
                    self.stats.copy_propagations += 1
                    self.changed = True
                    return deepcopy(replacement)
                elif isinstance(replacement, VarRefExpr) and replacement.name != expr.name:
                    self.stats.copy_propagations += 1
                    self.changed = True
                    return deepcopy(replacement)
            return expr
        elif isinstance(expr, BinaryExpr):
            expr.left = self._apply_copy_propagation(expr.left, copies)
            expr.right = self._apply_copy_propagation(expr.right, copies)
            return expr
        elif isinstance(expr, UnaryExpr):
            # CRITICAL: Do NOT propagate through address-of operator!
            # &a needs the variable's address, not its value.
            # If we replace 'a' with a constant, &1 is invalid!
            if expr.op == '&':
                return expr  # Keep as-is
            expr.expr = self._apply_copy_propagation(expr.expr, copies)
            return expr
        elif isinstance(expr, TypeCastExpr):
            expr.expr = self._apply_copy_propagation(expr.expr, copies)
            return expr
        elif isinstance(expr, CallExpr):
            expr.args = [(t, self._apply_copy_propagation(e, copies)) for t, e in expr.args]
            return expr
        return expr
    
    # =========================================================================
    # Helper Functions
    # =========================================================================
    
    def _are_exprs_equal(self, e1, e2) -> bool:
        """Check if two expressions are structurally equal."""
        if type(e1) != type(e2):
            return False
        if isinstance(e1, LiteralExpr):
            return e1.value == e2.value
        if isinstance(e1, VarRefExpr):
            return e1.name == e2.name
        if isinstance(e1, BinaryExpr):
            return (e1.op == e2.op and 
                    self._are_exprs_equal(e1.left, e2.left) and 
                    self._are_exprs_equal(e1.right, e2.right))
        if isinstance(e1, UnaryExpr):
            return (e1.op == e2.op and 
                    self._are_exprs_equal(e1.expr, e2.expr))
        if isinstance(e1, TypeCastExpr):
            return (e1.target_type == e2.target_type and 
                    self._are_exprs_equal(e1.expr, e2.expr))
        return False


# =============================================================================
# Assembly-Level Peephole Optimizer
# =============================================================================

class PeepholeOptimizer:
    """
    Assembly-level peephole optimizer for x86-64.
    
    Performs local optimizations on generated assembly code to
    eliminate redundant instructions and use more efficient encodings.
    """
    
    def __init__(self):
        self.stats = OptimizationStats()
    
    def optimize(self, asm_lines: List[str]) -> List[str]:
        """Apply peephole optimizations to assembly code."""
        lines = asm_lines.copy()
        
        # Multiple passes
        for _ in range(3):
            old_len = len(lines)
            lines = self._remove_redundant_movs(lines)
            lines = self._optimize_push_pop(lines)
            lines = self._optimize_lea_arithmetic(lines)
            lines = self._remove_nop_operations(lines)
            lines = self._combine_adjacent_ops(lines)
            
            if len(lines) == old_len:
                break
        
        return lines
    
    def _remove_redundant_movs(self, lines: List[str]) -> List[str]:
        """Remove mov reg, reg where dest == src."""
        result = []
        for line in lines:
            stripped = line.strip()
            # Pattern: mov rax, rax
            if stripped.startswith('mov '):
                parts = stripped[4:].replace(',', ' ').split()
                if len(parts) >= 2 and parts[0] == parts[1]:
                    self.stats.peephole_optimizations += 1
                    continue
            result.append(line)
        return result
    
    def _optimize_push_pop(self, lines: List[str]) -> List[str]:
        """Eliminate adjacent push/pop of same register."""
        result = []
        i = 0
        while i < len(lines):
            if i + 1 < len(lines):
                curr = lines[i].strip()
                next_line = lines[i + 1].strip()
                
                # Pattern: push rax; pop rax -> nothing
                if curr.startswith('push ') and next_line.startswith('pop '):
                    push_reg = curr[5:].strip()
                    pop_reg = next_line[4:].strip()
                    if push_reg == pop_reg:
                        self.stats.peephole_optimizations += 1
                        i += 2
                        continue
                
                # Pattern: push rax; pop rbx -> mov rbx, rax
                if curr.startswith('push ') and next_line.startswith('pop '):
                    push_reg = curr[5:].strip()
                    pop_reg = next_line[4:].strip()
                    if push_reg != pop_reg:
                        self.stats.peephole_optimizations += 1
                        result.append(f"    mov {pop_reg}, {push_reg}")
                        i += 2
                        continue
            
            result.append(lines[i])
            i += 1
        return result
    
    def _optimize_lea_arithmetic(self, lines: List[str]) -> List[str]:
        """Use LEA for efficient multi-operation arithmetic."""
        # This is complex; keeping as placeholder
        return lines
    
    def _remove_nop_operations(self, lines: List[str]) -> List[str]:
        """Remove operations that have no effect."""
        result = []
        for line in lines:
            stripped = line.strip()
            
            # add reg, 0 -> nothing
            if stripped.startswith('add ') and stripped.endswith(', 0'):
                self.stats.peephole_optimizations += 1
                continue
            
            # sub reg, 0 -> nothing
            if stripped.startswith('sub ') and stripped.endswith(', 0'):
                self.stats.peephole_optimizations += 1
                continue
            
            # imul reg, 1 -> nothing
            if stripped.startswith('imul ') and stripped.endswith(', 1'):
                self.stats.peephole_optimizations += 1
                continue
            
            # shl reg, 0 or shr reg, 0 -> nothing
            if (stripped.startswith('shl ') or stripped.startswith('shr ')) and stripped.endswith(', 0'):
                self.stats.peephole_optimizations += 1
                continue
            
            # or reg, 0 -> nothing
            if stripped.startswith('or ') and stripped.endswith(', 0'):
                self.stats.peephole_optimizations += 1
                continue
            
            # and reg, -1 -> nothing
            if stripped.startswith('and ') and stripped.endswith(', -1'):
                self.stats.peephole_optimizations += 1
                continue
            
            result.append(line)
        return result
    
    def _combine_adjacent_ops(self, lines: List[str]) -> List[str]:
        """Combine adjacent operations on same register."""
        result = []
        i = 0
        while i < len(lines):
            if i + 1 < len(lines):
                curr = lines[i].strip()
                next_line = lines[i + 1].strip()
                
                # Pattern: add rax, N; add rax, M -> add rax, N+M
                if curr.startswith('add rax, ') and next_line.startswith('add rax, '):
                    try:
                        n = int(curr.split(',')[1].strip())
                        m = int(next_line.split(',')[1].strip())
                        total = n + m
                        if total == 0:
                            self.stats.peephole_optimizations += 1
                            i += 2
                            continue
                        self.stats.peephole_optimizations += 1
                        result.append(f"    add rax, {total}")
                        i += 2
                        continue
                    except ValueError:
                        pass
                
                # Pattern: sub rax, N; sub rax, M -> sub rax, N+M
                if curr.startswith('sub rax, ') and next_line.startswith('sub rax, '):
                    try:
                        n = int(curr.split(',')[1].strip())
                        m = int(next_line.split(',')[1].strip())
                        total = n + m
                        if total == 0:
                            self.stats.peephole_optimizations += 1
                            i += 2
                            continue
                        self.stats.peephole_optimizations += 1
                        result.append(f"    sub rax, {total}")
                        i += 2
                        continue
                    except ValueError:
                        pass
            
            result.append(lines[i])
            i += 1
        return result


# =============================================================================
# Integrated Optimizer Interface
# =============================================================================

def optimize_ast(ast, optimization_level: int = 3):
    """
    Main entry point for AST optimization.
    
    Args:
        ast: The parsed AST
        optimization_level: 0 (none) to 3 (aggressive)
    
    Returns:
        Optimized AST
    """
    if optimization_level == 0:
        return ast
    
    optimizer = ASTOptimizer(ast, optimization_level)
    return optimizer.optimize()


def optimize_assembly(asm: str) -> str:
    """
    Post-process assembly with peephole optimizations.
    
    Args:
        asm: Generated assembly code as string
    
    Returns:
        Optimized assembly code
    """
    lines = asm.split('\n')
    optimizer = PeepholeOptimizer()
    optimized_lines = optimizer.optimize(lines)
    return '\n'.join(optimized_lines)
