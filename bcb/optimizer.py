from .parser import *

class ASTOptimizer:
    def __init__(self, ast):
        self.ast = ast

    def optimize(self):
        if isinstance(self.ast, Program):
            for i, decl in enumerate(self.ast.declarations):
                if isinstance(decl, FunctionDef):
                    self.ast.declarations[i] = self.optimize_function(decl)
                elif isinstance(decl, GlobalVarDecl):
                    decl.expr = self.optimize_expr(decl.expr)
        return self.ast

    def optimize_function(self, func):
        new_body = []
        for stmt in func.body:
            optimized_stmt = self.optimize_stmt(stmt)
            if optimized_stmt:
                if isinstance(optimized_stmt, list):
                    new_body.extend(optimized_stmt)
                else:
                    new_body.append(optimized_stmt)
        
        # Dead code elimination
        # Safely eliminate code after ReturnStmt, but keep labels and code after them
        reachable_body = []
        is_dead = False
        for stmt in new_body:
            if isinstance(stmt, LabelDef):
                is_dead = False # A label starts a new potentially reachable block
            
            if not is_dead:
                reachable_body.append(stmt)
            
            # We only strip after ReturnStmt. JmpStmt is too risky with manual labels 
            # as it might be a jump to a label later in the function.
            if isinstance(stmt, ReturnStmt):
                is_dead = True
        
        func.body = reachable_body
        return func

    def optimize_stmt(self, stmt):
        if isinstance(stmt, VarDeclStmt):
            stmt.expr = self.optimize_expr(stmt.expr)
            return stmt
        elif isinstance(stmt, VarAssignStmt):
            stmt.expr = self.optimize_expr(stmt.expr)
            return stmt
        elif isinstance(stmt, ReturnStmt):
            if stmt.expr:
                stmt.expr = self.optimize_expr(stmt.expr)
            return stmt
        elif isinstance(stmt, IfStmt):
            new_cond_bodies = []
            for cond, body in stmt.conditions_and_bodies:
                optimized_cond = self.optimize_expr(cond) if cond else None
                
                # Check for constant condition
                if optimized_cond and isinstance(optimized_cond, LiteralExpr):
                    if optimized_cond.value:
                        # Truthy constant condition: this branch always executes
                        optimized_body = []
                        for s in body:
                            o_s = self.optimize_stmt(s)
                            if o_s:
                                if isinstance(o_s, list): optimized_body.extend(o_s)
                                else: optimized_body.append(o_s)
                        
                        # If this is the first branch, we can just return the body
                        if not new_cond_bodies:
                            return optimized_body
                        
                        # Else it's an 'elif' that is always true, so we keep it as 'else' effectively
                        new_cond_bodies.append((None, optimized_body))
                        break # No need for further branches
                    else:
                        # Falsy constant condition: this branch never executes
                        continue
                
                optimized_body = []
                for s in body:
                    o_s = self.optimize_stmt(s)
                    if o_s:
                        if isinstance(o_s, list): optimized_body.extend(o_s)
                        else: optimized_body.append(o_s)
                new_cond_bodies.append((optimized_cond, optimized_body))
            
            if not new_cond_bodies:
                return None
            
            stmt.conditions_and_bodies = new_cond_bodies
            return stmt
        
        elif isinstance(stmt, WhileStmt):
            stmt.condition = self.optimize_expr(stmt.condition)
            if isinstance(stmt.condition, LiteralExpr) and not stmt.condition.value:
                return None # While(0) is dead
            
            new_body = []
            for s in stmt.body:
                o_s = self.optimize_stmt(s)
                if o_s:
                    if isinstance(o_s, list): new_body.extend(o_s)
                    else: new_body.append(o_s)
            stmt.body = new_body
            return stmt
        
        elif isinstance(stmt, PushStmt):
            stmt.expr = self.optimize_expr(stmt.expr)
            return stmt
        
        elif isinstance(stmt, LabelDef):
            return stmt
        
        elif isinstance(stmt, JmpStmt):
            return stmt
            
        elif isinstance(stmt, CallExpr):
            # As a statement, call expression is also optimized
            new_args = []
            for at, ae in stmt.args:
                new_args.append((at, self.optimize_expr(ae)))
            stmt.args = new_args
            return stmt

        elif isinstance(stmt, ArrayAssignStmt):
            stmt.index = self.optimize_expr(stmt.index)
            stmt.expr = self.optimize_expr(stmt.expr)
            return stmt
            
        elif isinstance(stmt, FieldAssignStmt):
            stmt.expr = self.optimize_expr(stmt.expr)
            return stmt

        return stmt

    def optimize_expr(self, expr):
        if isinstance(expr, BinaryExpr):
            expr.left = self.optimize_expr(expr.left)
            expr.right = self.optimize_expr(expr.right)
            
            # Constant Folding
            if isinstance(expr.left, LiteralExpr) and isinstance(expr.right, LiteralExpr):
                lval = expr.left.value
                rval = expr.right.value
                try:
                    res = None
                    if expr.op == '+': res = lval + rval
                    elif expr.op == '-': res = lval - rval
                    elif expr.op == '*': res = lval * rval
                    elif expr.op == '/': 
                        if rval == 0: return expr # Don't fold div by zero, let it fail at runtime or analyzer catch it
                        # If both are ints, use floor division to match common C-like behavior
                        if isinstance(lval, int) and isinstance(rval, int):
                            res = lval // rval
                        else:
                            res = lval / rval
                    elif expr.op == '==': res = 1 if lval == rval else 0
                    elif expr.op == '!=': res = 1 if lval != rval else 0
                    elif expr.op == '<': res = 1 if lval < rval else 0
                    elif expr.op == '>': res = 1 if lval > rval else 0
                    elif expr.op == '<=': res = 1 if lval <= rval else 0
                    elif expr.op == '>=': res = 1 if lval >= rval else 0
                    elif expr.op == '&&': res = 1 if lval and rval else 0
                    elif expr.op == '||': res = 1 if lval or rval else 0
                    
                    if res is not None:
                        return LiteralExpr(res, expr.line, expr.column)
                except:
                    pass
            
            # Identity and Algebraic Simplification
            if isinstance(expr.right, LiteralExpr):
                rval = expr.right.value
                if expr.op == '+' and rval == 0: return expr.left
                if expr.op == '-' and rval == 0: return expr.left
                if expr.op == '*' and rval == 1: return expr.left
                if expr.op == '*' and rval == 0: return LiteralExpr(0, expr.line, expr.column)
                if expr.op == '/' and rval == 1: return expr.left
                if expr.op == '&&' and rval == 0: return LiteralExpr(0, expr.line, expr.column)
                if expr.op == '&&' and rval != 0: return expr.left
                if expr.op == '||' and rval != 0: return LiteralExpr(1, expr.line, expr.column)
                if expr.op == '||' and rval == 0: return expr.left
                if expr.op == '&' and rval == 0: return LiteralExpr(0, expr.line, expr.column)
                if expr.op == '&' and rval == -1: return expr.left
                if expr.op == '|' and rval == 0: return expr.left
                if expr.op == '|' and rval == -1: return LiteralExpr(-1, expr.line, expr.column)
                if expr.op == '^' and rval == 0: return expr.left
                if expr.op == '<<' and rval == 0: return expr.left
                if expr.op == '>>' and rval == 0: return expr.left

                # Strength Reduction (Powers of 2)
                if expr.op == '*' and rval > 0 and (rval & (rval - 1) == 0):
                    shift = rval.bit_length() - 1
                    return BinaryExpr(expr.left, '<<', LiteralExpr(shift, expr.line, expr.column), expr.line, expr.column)
            
            if isinstance(expr.left, LiteralExpr):
                lval = expr.left.value
                if expr.op == '+' and lval == 0: return expr.right
                if expr.op == '*' and lval == 1: return expr.right
                if expr.op == '*' and lval == 0: return LiteralExpr(0, expr.line, expr.column)
                if expr.op == '&&' and lval == 0: return LiteralExpr(0, expr.line, expr.column)
                if expr.op == '||' and lval != 0: return LiteralExpr(1, expr.line, expr.column)
                if expr.op == '&' and lval == 0: return LiteralExpr(0, expr.line, expr.column)
                if expr.op == '|' and lval == -1: return LiteralExpr(-1, expr.line, expr.column)
            
            # Complex Algebraic Simplifications
            if expr.op == '-' and self.are_exprs_equal(expr.left, expr.right):
                return LiteralExpr(0, expr.line, expr.column)
            if expr.op == '^' and self.are_exprs_equal(expr.left, expr.right):
                return LiteralExpr(0, expr.line, expr.column)
            if expr.op == '==' and self.are_exprs_equal(expr.left, expr.right):
                return LiteralExpr(1, expr.line, expr.column)
            if expr.op == '!=' and self.are_exprs_equal(expr.left, expr.right):
                return LiteralExpr(0, expr.line, expr.column)
            if expr.op == '<=' and self.are_exprs_equal(expr.left, expr.right):
                return LiteralExpr(1, expr.line, expr.column)
            if expr.op == '>=' and self.are_exprs_equal(expr.left, expr.right):
                return LiteralExpr(1, expr.line, expr.column)

        elif isinstance(expr, UnaryExpr):
            expr.expr = self.optimize_expr(expr.expr)
            if expr.op == '-' and isinstance(expr.expr, LiteralExpr) and isinstance(expr.expr.value, (int, float)):
                return LiteralExpr(-expr.expr.value, expr.line, expr.column)
            if expr.op == '!' and isinstance(expr.expr, LiteralExpr):
                return LiteralExpr(1 if not expr.expr.value else 0, expr.line, expr.column)

        elif isinstance(expr, TypeCastExpr):
            expr.expr = self.optimize_expr(expr.expr)
            if isinstance(expr.expr, LiteralExpr):
                if expr.target_type in ['int64', 'int32', 'int16', 'int8', 'char']:
                    try:
                        expr.expr.value = int(expr.expr.value)
                        return expr.expr
                    except: pass
                elif expr.target_type in ['float64', 'float32']:
                    try:
                        expr.expr.value = float(expr.expr.value)
                        return expr.expr
                    except: pass
        
        elif isinstance(expr, CallExpr):
            new_args = []
            for at, ae in expr.args:
                new_args.append((at, self.optimize_expr(ae)))
            expr.args = new_args
            
        elif isinstance(expr, ArrayAccessExpr):
            expr.arr = self.optimize_expr(expr.arr)
            if expr.index:
                expr.index = self.optimize_expr(expr.index)
        
        elif isinstance(expr, FieldAccessExpr):
            expr.obj = self.optimize_expr(expr.obj)
            
        elif isinstance(expr, StructLiteralExpr):
            for i, (fn, ft, fe) in enumerate(expr.field_values):
                expr.field_values[i] = (fn, ft, self.optimize_expr(fe))
                
        elif isinstance(expr, ArrayLiteralExpr):
            expr.values = [self.optimize_expr(v) for v in expr.values]

        return expr

    def are_exprs_equal(self, e1, e2):
        """Simplistic equality check for AST nodes."""
        if type(e1) != type(e2): return False
        if isinstance(e1, LiteralExpr): return e1.value == e2.value
        if isinstance(e1, VarRefExpr): return e1.name == e2.name
        if isinstance(e1, BinaryExpr):
            return e1.op == e2.op and self.are_exprs_equal(e1.left, e2.left) and self.are_exprs_equal(e1.right, e2.right)
        if isinstance(e1, TypeCastExpr):
            return e1.target_type == e2.target_type and self.are_exprs_equal(e1.expr, e2.expr)
        return False
