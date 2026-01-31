from .parser import *
from .parser import NoValueExpr
from .errors import DiagnosticLevel

class SemanticAnalyzer:
    def __init__(self, ast, error_manager):
        self.ast = ast
        self.errors = error_manager
        self.scopes = [] # Stack of scopes (dicts)
        self.global_scope = {}
        self.structs = {}
        self.enums = {}
        self.current_function = None
        self.function_stack = [] # Virtual stack for tracking push/pop types

    def enter_scope(self):
        self.scopes.append({})

    def exit_scope(self):
        self.scopes.pop()

    def declare(self, name, type_info, node=None):
        if not self.scopes:
            self.global_scope[name] = type_info
        else:
            if name in self.scopes[-1]:
                self.errors.error(f"Redeclaration of variable '{name}'", node.line, node.column, f"Variable '{name}' is already defined in this scope.")
            self.scopes[-1][name] = type_info

    def lookup(self, name):
        # Check local scopes in reverse
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        # Check global scope (functions, data)
        if name in self.global_scope:
            return self.global_scope[name]
        return None
    
    def is_type_compatible(self, target, actual):
        if target == actual: return True
        # Allow implicit int conversions (Warning or PRE?)
        if target in ['int64', 'int32', 'int16', 'int8', 'char'] and actual in ['int64', 'int32', 'int16', 'int8', 'char']:
             return True # Compatibleish
        if target in ['float64', 'float32'] and actual in ['float64', 'float32']:
             return True
        if actual == "no_value": return True
        return False

    def check_type_compatibility(self, target, actual, node, context="assignment", expr_node=None):
        if actual == "struct_literal":
             # We assume if it's a struct literal being assigned to a struct type, it's valid for now.
             # Ideally we check fields.
             if target in self.structs:
                  return # Optimistic pass for struct literal
        
        # Enums are basically ints
        if target in self.enums and actual in ['int64', 'int32', 'int16', 'int8']:
             # Warning?
             return
        if actual in self.enums and target in ['int64', 'int32', 'int16', 'int8']:
             return

        # Pointers: allow void*? or specific checks
        if isinstance(target, str) and target.endswith('*') and isinstance(actual, str) and actual.endswith('*'):
             # Allow ptr mismatch if one is void*? For now strict
             if target != actual:
                 self.errors.warning(f"Pointer type mismatch in {context}: expected '{target}', got '{actual}'", node.line, node.column)
             return

        if not self.is_type_compatible(target, actual):
            self.errors.error(f"Type mismatch in {context}: expected '{target}', got '{actual}'", node.line, node.column)
        elif target != actual:
            # implicit conversion warning/PRE
            if target == "char" and actual in ["int32", "int64"]:
                 # Check if literal value fits char range (0-255 or -128-127)
                 if expr_node and isinstance(expr_node, LiteralExpr) and isinstance(expr_node.value, int):
                      val = expr_node.value
                      if 0 <= val <= 255: # standard ascii/byte
                           return
                 
                 self.errors.pre(f"Implicit conversion from '{actual}' to 'char' may truncate value", node.line, node.column, "Use explicit cast if intended.")
            elif target in ["int8", "int16", "int32"] and actual == "int64":
                 # Check if literal value fits
                 if expr_node and isinstance(expr_node, LiteralExpr) and isinstance(expr_node.value, int):
                      val = expr_node.value
                      fits = False
                      if target == "int32" and -(2**31) <= val <= (2**31)-1: fits = True
                      elif target == "int16" and -(2**15) <= val <= (2**15)-1: fits = True
                      elif target == "int8" and -(2**7) <= val <= (2**7)-1: fits = True
                      
                      if fits: return

                 self.errors.pre(f"Implicit conversion from 'int64' to '{target}' may truncate value", node.line, node.column)

    
    def analyze(self):
        # 1. Register Structs and Enums
        if self.ast.data_block:
            for s in self.ast.data_block.structs:
                self.structs[s.name] = s
            for e in self.ast.data_block.enums:
                self.enums[e.name] = e
            
            for type_name, name, value in self.ast.data_block.entries:
                self.declare(name, type_name)

        # 2. Register Functions
        for decl in self.ast.declarations:
            if isinstance(decl, (FunctionDef, FunctionDecl)):
                self.global_scope[decl.name] = ("function", decl.return_type, decl.params)

        # 3. Analyze Function Bodies
        for decl in self.ast.declarations:
            if isinstance(decl, FunctionDef):
                self.analyze_function(decl)

    def analyze_function(self, func):
        self.current_function = func
        self.function_stack = [] # Reset virtual stack
        self.enter_scope()
        
        for name, type_name in func.params:
            self.declare(name, type_name, func)
        
        for stmt in func.body:
            self.analyze_stmt(stmt)
            
        self.exit_scope()
        self.current_function = None

    def analyze_stmt(self, stmt):
        if isinstance(stmt, VarDeclStmt):
            expr_type = self.analyze_expr(stmt.expr)
            self.declare(stmt.name, stmt.type_name, stmt)
            self.check_type_compatibility(stmt.type_name, expr_type, stmt, f"declaration of '{stmt.name}'", expr_node=stmt.expr)
            
        elif isinstance(stmt, VarAssignStmt):
            var_type = self.analyze_lvalue(stmt.name, stmt) # Simple check
            expr_type = self.analyze_expr(stmt.expr)
            if var_type:
                # Handle pointer deref assignment
                target_type = var_type
                if isinstance(stmt.type_name, str) and stmt.type_name.endswith('*'):
                     # md int32* ptr = val; means *ptr = val
                     # var_type should be int32*
                     # target of assignment is int32 (what ptr points to)
                     target_type = var_type[:-1] # strip *
                
                self.check_type_compatibility(target_type, expr_type, stmt, f"assignment to '{stmt.name}'", expr_node=stmt.expr)

        elif isinstance(stmt, ReturnStmt):
            if stmt.expr:
                expr_type = self.analyze_expr(stmt.expr)
                if self.current_function:
                    self.check_type_compatibility(self.current_function.return_type, expr_type, stmt, "return statement", expr_node=stmt.expr)
            else:
                 if self.current_function and self.current_function.return_type != "void":
                      self.errors.error(f"Function local '{self.current_function.name}' expects return type '{self.current_function.return_type}', got void", stmt.line, stmt.column)

        elif isinstance(stmt, CallExpr):
            self.analyze_call(stmt)
            
        elif isinstance(stmt, IfStmt):
             for cond, body in stmt.conditions_and_bodies:
                 if cond:
                     c_type = self.analyze_expr(cond)
                 self.enter_scope() # Blocks have scope
                 for s in body:
                     self.analyze_stmt(s)
                 self.exit_scope()
        
        elif isinstance(stmt, WhileStmt):
             self.analyze_expr(stmt.condition)
             self.enter_scope()
             # Warning: Stack effects inside loops are not tracked across iterations currently
             stack_depth_before = len(self.function_stack)
             for s in stmt.body:
                 self.analyze_stmt(s)
             if len(self.function_stack) != stack_depth_before:
                  self.errors.warning("Stack modification inside loop may cause overflow/underflow if not balanced", stmt.line, stmt.column)
             self.exit_scope()

        elif isinstance(stmt, PushStmt):
            expr_type = self.analyze_expr(stmt.expr)
            # Check compatibility between expr and push type
            self.check_type_compatibility(stmt.type_name, expr_type, stmt, "push statement", expr_node=stmt.expr)
            self.function_stack.append((stmt.type_name, stmt))

        elif isinstance(stmt, PopStmt):
            if not self.function_stack:
                self.errors.error("Stack underflow: popping from empty stack", stmt.line, stmt.column)
                return
            
            pushed_type, pushed_node = self.function_stack.pop()
            
            # Check if popped type matches pushed type
            # We enforce strict matching or safe conversion here?
            # User example: push int16, pop int32.
            # If implementation pushes 64-bit, this is fine visually if types compatible.
            if not self.is_type_compatible(stmt.type_name, pushed_type) and not self.is_type_compatible(pushed_type, stmt.type_name):
                 self.errors.warning(f"Popped type '{stmt.type_name}' does not match pushed type '{pushed_type}'", stmt.line, stmt.column)

            # Check variable assignment
            var_type = self.analyze_lvalue(stmt.var_name, stmt)
            if var_type:
                self.check_type_compatibility(var_type, stmt.type_name, stmt, f"pop to '{stmt.var_name}'")

    def analyze_lvalue(self, name, node):
        t = self.lookup(name)
        if not t:
            self.errors.error(f"Undefined variable '{name}'", node.line, node.column)
            return None
        return t

    def analyze_expr(self, expr):
        if isinstance(expr, LiteralExpr):
            if isinstance(expr.value, int): return "int64" # Default
            if isinstance(expr.value, float): return "float64"
            if isinstance(expr.value, str): return "string"
            return "unknown"
            
        elif isinstance(expr, VarRefExpr):
            t = self.lookup(expr.name)
            if not t:
                self.errors.error(f"Undefined variable '{expr.name}'", expr.line, expr.column)
                return "unknown"
            return t
            
        elif isinstance(expr, BinaryExpr):
            l_type = self.analyze_expr(expr.left)
            r_type = self.analyze_expr(expr.right)
            
            if expr.op in ["==", "!=", "<", ">", "<=", ">="]:
                return "int64" # bools are ints here
            
            if l_type == "unknown" or r_type == "unknown": return "unknown"
            
            # Simple interaction rules
            if l_type == r_type: return l_type
            if "float" in l_type or "float" in r_type:
                 return "float64" # promote to float
            return "int64" # default for mixed ints

        elif isinstance(expr, CallExpr):
            return self.analyze_call(expr)
            
        elif isinstance(expr, TypeCastExpr):
            self.analyze_expr(expr.expr) # check inner
            return expr.target_type

        elif isinstance(expr, UnaryExpr):
            t = self.analyze_expr(expr.expr)
            if expr.op == '&':
                 return f"{t}*"
            if expr.op == '*':
                 if isinstance(t, str) and t.endswith('*'):
                      return t[:-1]
                 else:
                      self.errors.error(f"Cannot dereference non-pointer type '{t}'", expr.line, expr.column)
                      return "unknown"
            return t
        
        elif isinstance(expr, EnumValueExpr):
            # Verify enum exists and value exists
            if expr.enum_name not in self.enums:
                 self.errors.error(f"Undefined enum '{expr.enum_name}'", expr.line, expr.column)
                 return "unknown"
            e = self.enums[expr.enum_name]
            if expr.value_name not in e.values:
                 self.errors.error(f"Enum '{expr.enum_name}' has no member '{expr.value_name}'", expr.line, expr.column)
                 return "unknown"
            return expr.enum_name # Return the Enum type name

        elif isinstance(expr, StructLiteralExpr):
            # For now, we don't infer struct type well from AST unless passed down, 
            # but usually it's assigned to a variable of known type.
            # We can try to guess from fields or return "struct_literal" and let check_type_compatibility handle it if target is struct.
            return "struct_literal"
            
        elif isinstance(expr, FieldAccessExpr):
            obj_type = self.analyze_expr(expr.obj)
            if obj_type in ["unknown", "struct_literal"]: return "unknown"
            
            # Check if obj_type is a struct
            if obj_type in self.structs:
                 s = self.structs[obj_type]
                 # Find field
                 field_type = None
                 for ft, fn in s.fields:
                      if fn == expr.field_name:
                           field_type = ft
                           break
                 if not field_type:
                      self.errors.error(f"Struct '{s.name}' has no field '{expr.field_name}'", expr.line, expr.column)
                      return "unknown"
                 return field_type
            else:
                 self.errors.error(f"Cannot access field '{expr.field_name}' on non-struct type '{obj_type}'", expr.line, expr.column)
                 return "unknown"

        elif isinstance(expr, NoValueExpr):
            return "no_value"

        return "unknown"

    def analyze_call(self, call_node):
        func_info = self.lookup(call_node.name)
        if not func_info:
            # We no longer implicitly allow printf. It must be declared.
            self.errors.error(
                f"Undefined function '{call_node.name}'", 
                call_node.line, 
                call_node.column,
                hint=f"Did you forget to 'define {call_node.name}(...)' or import it? If it's an external C function like printf, you must declare it first."
            )
            return "unknown"
        
        # ("function", ret, params)
        kind, ret_type, params = func_info
        if kind != "function":
             self.errors.error(f"'{call_node.name}' is not a function", call_node.line, call_node.column)
             return "unknown"

        # Check arg count
        # Handle varargs like printf (not strictly defined in AST as varags yet, but parser supports ...args)
        # Note: func_info params is list of (name, type)
        # Check if last param is varargs
        
        is_varargs = False
        if params and params[-1][1].startswith("..."):
            is_varargs = True
        
        if is_varargs:
            min_args = len(params) - 1
            if len(call_node.args) < min_args:
                 self.errors.error(f"Function '{call_node.name}' expects at least {min_args} arguments, got {len(call_node.args)}", call_node.line, call_node.column)
        else:
            if len(call_node.args) != len(params):
                self.errors.error(f"Function '{call_node.name}' expects {len(params)} arguments, got {len(call_node.args)}", call_node.line, call_node.column)
        
        # Check arg types
        for i, (arg_type, arg_expr) in enumerate(call_node.args):
            if is_varargs and i >= len(params) - 1:
                break # Skip type check for varargs for now
            
            if i < len(params):
                expected_type = params[i][1]
                # We can verify explicitly passed type against expected, OR expression type
                # The node has arg_type (string from source call(explicit_type expr))
                
                # Check explicit cast in call
                self.check_type_compatibility(expected_type, arg_type, call_node, f"argument {i+1} of '{call_node.name}'")
                
                # Analyze inner expression
                actual_expr_type = self.analyze_expr(arg_expr)
                
                # If explicit type provided in call (arg_type), we treat it as a Cast or Assertion.
                # If parsed as "int32* ptr", arg_type is int32*. actual_expr_type is int32* (if ptr is int32*).
                # check(int32*, int32*) -> OK.
                
                # If parsed as "int32 ptr", arg_type is int32. actual_expr_type is int32*.
                # check(int32, int32*) -> ERROR.
                
                self.check_type_compatibility(arg_type, actual_expr_type, call_node, f"argument {i+1} value", expr_node=arg_expr)

        if call_node.name == "printf":
             # PRE: Check format string?
             pass

        return ret_type
