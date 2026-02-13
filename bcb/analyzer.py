from .parser import *
from .parser import NoValueExpr, LengthExpr, GetTypeExpr, ArgsAccessExpr, ArrayAccessExpr, ArrayLiteralExpr, ArrayAssignStmt
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
        self.declared_vars = [] # List of (name, scope_depth, node)
        self.used_vars = set()      # (name, scope_depth)
        self.used_functions = {"printf", "malloc", "free", "exit"} # name
        self.declared_params = []   # (func_name, param_name, node)
        self.used_params = set()    # (func_name, param_name)
        self.in_condition = False

    def enter_scope(self):
        self.scopes.append({})

    def exit_scope(self):
        self.scopes.pop()

    def declare(self, name, type_info, node=None):
        if not self.scopes:
            if name in self.global_scope:
                existing = self.global_scope[name]
                if isinstance(existing, tuple) and existing[0] == "function":
                     self.error(f"Global name '{name}' conflicts with an existing function", node)
                else:
                     self.error(f"Redeclaration of global variable '{name}'", node)
            self.global_scope[name] = type_info
        else:
            if name in self.scopes[-1]:
                self.error(f"Redeclaration of variable '{name}'", node)
            
            # Check for shadowing
            shadowed = self.lookup(name, track=False)
            if shadowed:
                 self.warning(f"Variable '{name}' shadows an existing definition", node)

            self.scopes[-1][name] = type_info
            
            # Track for unused variable warning
            scope_depth = len(self.scopes) - 1
            self.declared_vars.append((name, scope_depth, node))

    def lookup(self, name, track=True):
        # Check local scopes in reverse
        for i, scope in enumerate(reversed(self.scopes)):
            if name in scope:
                if track:
                    # Track usage
                    scope_depth = len(self.scopes) - 1 - i
                    self.used_vars.add((name, scope_depth))
                    
                    # Track parameter usage
                    if self.current_function:
                        self.used_params.add((self.current_function.name, name))
                return scope[name]
        
        # Check global scope (functions, data)
        if name in self.global_scope:
            if track:
                self.used_functions.add(name)
                # Track for parameter usage even if global (to suppress false unused warnings)
                if self.current_function:
                    self.used_params.add((self.current_function.name, name))
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
        if actual == "struct_literal" and target in self.structs:
             return True
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

        # Array compatibility (e.g. int32[] = int64[] from literal or int32[10])
        if isinstance(target, str) and '[' in target and isinstance(actual, str) and '[' in actual:
             t_bracket = target.find('[')
             a_bracket = actual.find('[')
             target_elem = target[:t_bracket]
             actual_elem = actual[:a_bracket]
             if self.is_type_compatible(target_elem, actual_elem):
                  return

        # Pointers: allow void*? or specific checks
        if isinstance(target, str) and target.endswith('*') and isinstance(actual, str) and actual.endswith('*'):
             # Allow ptr mismatch if one is void*? For now strict
             if target != actual:
                 self.warning(f"Pointer type mismatch in {context}: expected '{target}', got '{actual}'", node)
             return

        if not self.is_type_compatible(target, actual):
            self.error(f"Type mismatch in {context}: expected '{target}', got '{actual}'", node)
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

        # 2. Register Global Variables and Functions
        for decl in self.ast.declarations:
            if isinstance(decl, GlobalVarDecl):
                # Analyze expression in global scope (empty scopes)
                expr_type = self.analyze_expr(decl.expr)
                declared_type = decl.type_name
                if decl.is_array:
                    declared_type += "[]"
                    if decl.array_size <= 0:
                         self.error(f"Invalid array size {decl.array_size} for global '{decl.name}'", decl)
                self.declare(decl.name, declared_type, decl)
                self.check_type_compatibility(declared_type, expr_type, decl, f"declaration of global variable '{decl.name}'", expr_node=decl.expr)
            elif isinstance(decl, (FunctionDef, FunctionDecl)):
                self.global_scope[decl.name] = ("function", decl.return_type, decl.params)

        # 3. Analyze Function Bodies
        for decl in self.ast.declarations:
            if isinstance(decl, FunctionDef):
                self.analyze_function(decl)
        
        # 4. Check for unused functions
        for name, info in self.global_scope.items():
            if info[0] == "function" and name not in self.used_functions:
                 # Find the function node to get location
                 func_node = None
                 for d in self.ast.declarations:
                      if (isinstance(d, (FunctionDef, FunctionDecl))) and d.name == name:
                           func_node = d
                           break
                 if func_node and name != "main":
                      self.warning(f"Unused function '{name}'", func_node)

    def analyze_function(self, func):
        self.current_function = func
        self.function_stack = [] # Reset virtual stack
        self.enter_scope()
        
        for name, type_name in func.params:
            self.declare(name, type_name, func)
            self.declared_params.append((func.name, name, func))
        
        # Track initial declared_vars count to check only this function's locals
        start_declared = len(self.declared_vars)
        
        is_reachable = True
        for stmt in func.body:
            if not is_reachable:
                 self.warning("Unreachable code detected", stmt)
                 # We still analyze it to find errors, but only once.
                 # To avoid spamming warnings, we could break, but let's keep analyzing.
            
            self.analyze_stmt(stmt)
            
            if isinstance(stmt, (ReturnStmt, JmpStmt)):
                 is_reachable = False
            
        # Check for unused local variables in this function
        # Locals are those at scope_depth >= 1
        for i in range(start_declared, len(self.declared_vars)):
            name, depth, node = self.declared_vars[i]
            if depth >= 1 and (name, depth) not in self.used_vars:
                 # Ignore if it starts with underscore
                 if not name.startswith('_'):
                      self.warning(f"Unused variable '{name}'", node)
        
        # Check for unused parameters
        for fname, pname, node in self.declared_params:
            if fname == func.name and (fname, pname) not in self.used_params:
                if not pname.startswith('_'):
                    self.warning(f"Unused parameter '{pname}' in function '{fname}'", node)

        self.exit_scope()
        self.current_function = None

    def analyze_stmt(self, stmt):
        if isinstance(stmt, VarDeclStmt):
            expr_type = self.analyze_expr(stmt.expr)
            declared_type = stmt.type_name
            if stmt.is_array:
                declared_type += "[]"
            self.declare(stmt.name, declared_type, stmt)
            
            # For array init with literal, we might need relaxed check or "array_literal" handling
            # If expr_type matches declared_type (e.g. int32[] vs int32[]) -> OK.
            
            self.check_type_compatibility(declared_type, expr_type, stmt, f"declaration of '{stmt.name}'", expr_node=stmt.expr)
            
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
                       self.error(f"Function '{self.current_function.name}' expects return type '{self.current_function.return_type}', got void", stmt)

        elif isinstance(stmt, CallExpr):
            self.analyze_call(stmt)
            
        elif isinstance(stmt, IfStmt):
             for cond, body in stmt.conditions_and_bodies:
                 if cond:
                     old_in_cond = self.in_condition
                     self.in_condition = True
                     t = self.analyze_expr(cond)
                     self.in_condition = old_in_cond
                     if t == "no_value":
                         self.error("Condition cannot be 'no_value'", cond)
                 
                 self.enter_scope() # Blocks have scope
                 if not body:
                      self.tip("Empty block in $if statement.", cond if cond else stmt)
                 
                 for s in body:
                     self.analyze_stmt(s)
                 self.exit_scope()
        
        elif isinstance(stmt, WhileStmt):
             old_in_cond = self.in_condition
             self.in_condition = True
             t = self.analyze_expr(stmt.condition)
             self.in_condition = old_in_cond
             
             if t == "no_value":
                 self.error("Condition cannot be 'no_value'", stmt.condition)
             
             # PRE: Check for infinite/dead loops
             if isinstance(stmt.condition, LiteralExpr):
                  if stmt.condition.value:
                       self.pre("Potential infinite loop detected", stmt, hint="Ensure the loop has a 'jmp' or 'return' that breaks out, or the condition changes.")
                  else:
                       self.warning("Loop body is unreachable (condition is constant false)", stmt)

             self.enter_scope()
             if not stmt.body:
                  self.warning("Empty $while loop detected", stmt, hint="Consider if this was intended to wait for a side effect.")
             
             # Warning: Stack effects inside loops are not tracked across iterations currently
             stack_depth_before = len(self.function_stack)
             for s in stmt.body:
                 self.analyze_stmt(s)
             if len(self.function_stack) != stack_depth_before:
                  self.warning("Stack modification inside loop may cause overflow/underflow if not balanced", stmt)
             self.exit_scope()

        elif isinstance(stmt, PushStmt):
            expr_type = self.analyze_expr(stmt.expr)
            # Check compatibility between expr and push type
            self.check_type_compatibility(stmt.type_name, expr_type, stmt, "push statement", expr_node=stmt.expr)
            self.function_stack.append((stmt.type_name, stmt))

        elif isinstance(stmt, PopStmt):
            if not self.function_stack:
                self.error("Stack underflow: popping from empty stack", stmt)
                return
            
            pushed_type, pushed_node = self.function_stack.pop()
            
            # Check if popped type matches pushed type
            # We enforce strict matching or safe conversion here?
            # User example: push int16, pop int32.
            # If implementation pushes 64-bit, this is fine visually if types compatible.
            if not self.is_type_compatible(stmt.type_name, pushed_type) and not self.is_type_compatible(pushed_type, stmt.type_name):
                 self.warning(f"Popped type '{stmt.type_name}' does not match pushed type '{pushed_type}'", stmt)

            # Check variable assignment
            var_type = self.analyze_lvalue(stmt.var_name, stmt)
            if var_type:
                self.check_type_compatibility(var_type, stmt.type_name, stmt, f"pop to '{stmt.var_name}'")

        elif isinstance(stmt, SwapStmt):
            if len(self.function_stack) < 2:
                self.error("Stack underflow: swap requires at least 2 elements on stack", stmt)
                return
            
            # Swap top two elements
            top = self.function_stack.pop()
            second = self.function_stack.pop()
            self.function_stack.append(top)
            self.function_stack.append(second)
            
            # Optionally check if swapped types match explicit type?
            # For now we assume 'swap T' is just an instruction.

        elif isinstance(stmt, DupStmt):
            if not self.function_stack:
                self.error("Stack underflow: dup requires at least 1 element on stack", stmt)
                return
            
            top_type, top_node = self.function_stack[-1]
            self.function_stack.append((top_type, stmt))
            
            if not self.is_type_compatible(stmt.type_name, top_type):
                 self.warning(f"Dup type '{stmt.type_name}' does not match stack top type '{top_type}'", stmt)

    def analyze_lvalue(self, name, node):
        t = self.lookup(name)
        if not t:
            self.error(f"Undefined variable '{name}'", node)
            return None
        return t

    def analyze_expr(self, expr):
        if isinstance(expr, LiteralExpr):
            if isinstance(expr.value, int): return "int32" # Changed from int64 to int32 by default
            if isinstance(expr.value, float): return "float64"
            if isinstance(expr.value, str): return "string"
            return "unknown"
            
        elif isinstance(expr, VarRefExpr):
            t = self.lookup(expr.name)
            if not t:
                self.error(f"Undefined variable '{expr.name}'", expr)
                return "unknown"
            return t
            
        elif isinstance(expr, BinaryExpr):
            l_type = self.analyze_expr(expr.left)
            r_type = self.analyze_expr(expr.right)
            
            if expr.op in ["==", "!=", "<", ">", "<=", ">="]:
                # PRE: Check for identical expressions on both sides
                if self.are_expressions_identical(expr.left, expr.right):
                    # Suppress warning if either is an explicit cast
                    if not (isinstance(expr.left, TypeCastExpr) or isinstance(expr.right, TypeCastExpr)):
                        self.warning(f"Comparison of identical expressions '{expr.op}' will always be constant", expr)
                
                expr.inferred_type = "int32"
                return "int32" # bools are ints here
            
            if l_type == "unknown" or r_type == "unknown":
                 expr.inferred_type = "unknown"
                 return "unknown"
            
             # Constant evaluation for specific pre-checks
            if isinstance(expr.left, LiteralExpr) and isinstance(expr.right, LiteralExpr):
                 if expr.op == '/' and expr.right.value == 0:
                      self.errors.pre("Division by zero detected", expr.line, expr.column, hint="Ensure the divisor is non-zero at runtime.")
                 elif expr.op == '%' and expr.right.value == 0:
                      self.errors.pre("Modulo by zero detected", expr.line, expr.column)
                 elif expr.op == '/' and expr.right.value == 0.0:
                      self.errors.pre("Division by zero (float) detected", expr.line, expr.column, hint="Floating point division by zero results in 'inf' or 'nan'.")

            # Simple interaction rules
            res_type = "int32" # Default to int32 math
            if l_type == r_type: res_type = l_type
            elif "float" in l_type or "float" in r_type:
                 res_type = "float64" # promote to float
            elif "int64" == l_type or "int64" == r_type:
                 res_type = "int64" # promote to int64
            
            # PRE: Check for always true/false comparisons
            if expr.op in ["==", "!="] and ("float" in l_type or "float" in r_type):
                 self.pre("Direct equality comparison of floating point numbers", expr, hint="Use an epsilon-based comparison (abs(a-b) < 0.0001) for more reliability.")
            
            expr.inferred_type = res_type
            return res_type

        elif isinstance(expr, CallExpr):
            ret = self.analyze_call(expr)
            if ret == "void":
                 # PRE: Using void in expression?
                 self.error(f"Function '{expr.name}' returns void and cannot be used in an expression", expr)
            return ret
            
        elif isinstance(expr, TypeCastExpr):
            inner_type = self.analyze_expr(expr.expr)
            if inner_type == expr.target_type:
                 # Suppress tip for LiteralExpr as it's the standard way to denote types
                 # Also suppress if we are in a condition (explicit type comparisons)
                 if not isinstance(expr.expr, LiteralExpr) and not self.in_condition:
                      self.tip(f"Redundant cast: expression is already of type '{expr.target_type}'", expr)
            return expr.target_type

        elif isinstance(expr, UnaryExpr):
            t = self.analyze_expr(expr.expr)
            if expr.op == '&':
                 return f"{t}*"
            if expr.op == '*':
                 if isinstance(t, str) and t.endswith('*'):
                      return t[:-1]
                 else:
                      self.error(f"Cannot dereference non-pointer type '{t}'", expr)
                      return "unknown"
            return t
        
        elif isinstance(expr, EnumValueExpr):
            # Verify enum exists and value exists
            if expr.enum_name not in self.enums:
                 self.error(f"Undefined enum '{expr.enum_name}'", expr)
                 return "unknown"
            e = self.enums[expr.enum_name]
            if expr.value_name not in e.values:
                 self.error(f"Enum '{expr.enum_name}' has no member '{expr.value_name}'", expr)
                 return "unknown"
            return expr.enum_name # Return the Enum type name

        elif isinstance(expr, LengthExpr):
             self.analyze_expr(expr.expr)
             return "int32"

        elif isinstance(expr, GetTypeExpr):
             t = self.analyze_expr(expr.expr)
             # Store the inferred type name in the node for codegen/optimization
             expr.inferred_type_name = t
             return "string"

        elif isinstance(expr, ArrayAccessExpr):
             arr_type = self.analyze_expr(expr.arr)
             if expr.index is not None:
                 self.analyze_expr(expr.index)
                 # PRE: OOB check for constant index
                 if isinstance(expr.index, LiteralExpr) and isinstance(expr.index.value, int):
                      idx = expr.index.value
                      # We need to know array size. If expr.arr is VarRef, we might find it.
                      if isinstance(expr.arr, VarRefExpr):
                           # Try to find declaration
                           decl_node = self.find_decl_node(expr.arr.name)
                           if decl_node and hasattr(decl_node, 'array_size') and decl_node.array_size is not None:
                                if idx < 0 or idx >= decl_node.array_size:
                                     self.pre(f"Array access out of bounds: index {idx} on array of size {decl_node.array_size}", expr, hint="Accessing memory outside array boundaries can lead to segments faults or memory corruption.")

                 if '[' in arr_type:
                     bracket_pos = arr_type.find('[')
                     return arr_type[:bracket_pos]
                 elif arr_type.endswith('*'):
                     return arr_type[:-1]
                 return "unknown"
             else:
                 return arr_type

        elif isinstance(expr, ArrayLiteralExpr):
             if not expr.values: return "unknown"
             t = self.analyze_expr(expr.values[0])
             return t + "[]"

        elif isinstance(expr, ArgsAccessExpr):
             self.analyze_expr(expr.index)
             var_type = self.lookup(expr.name)
             if not var_type or not "..." in var_type:
                  self.error(f"'{expr.name}' is not a variadic parameter and cannot be accessed with (index)", expr)
             return "dynamic_arg"

        elif isinstance(expr, StructLiteralExpr):
            # For now, we don't infer struct type well from AST unless passed down, 
            # but usually it's assigned to a variable of known type.
            # We can try to guess from fields or return "struct_literal" and let check_type_compatibility handle it if target is struct.
            return "struct_literal"
            
        elif isinstance(expr, FieldAccessExpr):
            # Check for .amount on variadic params
            if expr.field_name == "amount" and isinstance(expr.obj, VarRefExpr):
                var_type = self.lookup(expr.obj.name)
                if var_type and var_type.startswith("..."):
                    return "int32"

            # Check if this is an Enum access (Type.MEMBER)
            # Logic: If expr.obj is a VarRef, check if it shadows a variable. 
            # If not a variable, check if it's an Enum.
            if isinstance(expr.obj, VarRefExpr):
                var_type = self.lookup(expr.obj.name)
                if not var_type and expr.obj.name in self.enums:
                     # It is an Enum access
                     e = self.enums[expr.obj.name]
                     if expr.field_name not in e.values:
                          self.error(f"Enum '{expr.obj.name}' has no member '{expr.field_name}'", expr)
                          return "unknown"
                     return expr.obj.name

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
                      self.error(f"Struct '{s.name}' has no field '{expr.field_name}'", expr)
                      return "unknown"
                 return field_type
            else:
                 self.error(f"Cannot access field '{expr.field_name}' on non-struct type '{obj_type}'", expr)
                 return "unknown"

        elif isinstance(expr, NoValueExpr):
            return "no_value"

        return "unknown"

    def get_hint_for_error(self, message):
        """Returns a helpful tip based on the error message keyword."""
        msg = message.lower()
        if "undefined variable" in msg: return "Check for typos or ensure the variable is declared in the current or outer scope."
        if "undefined function" in msg: return "Ensure the function is defined or 'define'd as an external call."
        if "type mismatch" in msg: return "Use an explicit cast like 'int32(expression)' or 'float64(expression)' to resolve type differences."
        if "redeclaration" in msg: return "Use 'md' (Modify) to change the value of an existing variable instead of declaring it again."
        if "modify" in msg or "md" in msg: return "The 'md' keyword is mandatory for all variable and field assignments."
        if "struct" in msg and "field" in msg: return "Struct fields must be declared in the 'data' block before usage."
        if "pointer" in msg: return "Use '&' to get an address and '*' to dereference it."
        if "array" in msg: return "Arrays are zero-indexed and their size must be a positive constant integer."
        if "main" in msg: return "Your program entry point should be 'export main(void) -> int32'."
        if "float" in msg: return "Floating point math defaults to float64; use float32(expr) for single precision."
        if "control flow" in msg or "if" in msg or "while" in msg: return "BCB control flow keywords start with '$' (e.g., $if, $while)."
        if "variadic" in msg: return "For variadic functions like printf, provide an explicit format string as the first argument."
        if "conversion" in msg: return "BCB is strict about types. Explicit casts prevent unexpected truncation or precision loss."
        if "null" in msg: return "Initialize pointers with 'no_value' to safely represent null."
        if "length" in msg: return "Use the built-in 'length(array)' function to get the element count."
        if "stack" in msg: return "Each function call manages its own stack frame; ensure 'push' and 'pop' are balanced."
        if "data" in msg: return "The 'data' block is reserved for global strings, structs, and enums."
        if "call" in msg: return "Always use 'call function_name(args)' to execute logic."
        if "not a function" in msg: return "Check if the name is shadowed by a variable or if there is a typo in the function name."
        if "shadows" in msg: return "Avoid using the same name for local variables and outer scope variables/functions to prevent confusion."
        if "unused" in msg: return "Unused code can be safely removed or marked with an underscore prefix (e.g. _unused) to suppress this warning."
        if "redeclaration" in msg or "conflicts" in msg:
             if "global" in msg: return "Global names must be unique. Check for duplicate definitions or name clashes with functions."
             return "Use 'md' (Modify) to change the value of an existing variable instead of declaring it again."
        if "redundant cast" in msg: return "Casting an expression to its own type is unnecessary; BCB already correctly infers types in most cases."
        return "Refer to the BCB documentation for syntax and best practices."

    def error(self, message, node=None, line=None, column=None, hint=None):
         # Helper to wrap error with context hint
         l = line if line is not None else (node.line if node else 0)
         c = column if column is not None else (node.column if node else 0)
         if hint is None:
             hint = self.get_hint_for_error(message)
         self.errors.add(DiagnosticLevel.ERROR, message, l, c, hint=hint)

    def warning(self, message, node=None, line=None, column=None, hint=None):
        l = line if line is not None else (node.line if node else 0)
        c = column if column is not None else (node.column if node else 0)
        if hint is None:
             hint = self.get_hint_for_error(message)
        self.errors.add(DiagnosticLevel.WARNING, message, l, c, hint=hint)

    def tip(self, message, node=None, line=None, column=None, hint=None):
        l = line if line is not None else (node.line if node else 0)
        c = column if column is not None else (node.column if node else 0)
        if hint is None:
             hint = self.get_hint_for_error(message)
        self.errors.add(DiagnosticLevel.TIP, message, l, c, hint=hint)

    def pre(self, message, node=None, line=None, column=None, hint=None):
        l = line if line is not None else (node.line if node else 0)
        c = column if column is not None else (node.column if node else 0)
        if hint is None:
             hint = self.get_hint_for_error(message)
        self.errors.add(DiagnosticLevel.PRE, message, l, c, hint=hint)

    def are_expressions_identical(self, left, right):
        """Heuristic to check if two expressions are exactly the same."""
        if type(left) != type(right): return False
        if isinstance(left, LiteralExpr): return left.value == right.value
        if isinstance(left, VarRefExpr): return left.name == right.name
        # Recursively check binary
        if isinstance(left, BinaryExpr):
             return left.op == right.op and self.are_expressions_identical(left.left, right.left) and self.are_expressions_identical(left.right, right.right)
        if isinstance(left, TypeCastExpr):
             return left.target_type == right.target_type and self.are_expressions_identical(left.expr, right.expr)
        return False

    def find_decl_node(self, name):
         # Search trackable declared vars
         for n, d, node in reversed(self.declared_vars):
              if n == name: return node
         return None

    def analyze_call(self, call_node):
        func_info = self.lookup(call_node.name)
        
        # Always analyze arguments to track usage and find errors within them
        for arg_type, arg_expr in call_node.args:
            self.analyze_expr(arg_expr)

        if not func_info:
            self.error(f"Undefined function '{call_node.name}'", call_node)
            return "unknown"
        
        # Validate it's actually a function
        if not isinstance(func_info, tuple) or func_info[0] != "function":
             self.error(f"'{call_node.name}' is not a function", call_node)
             return "unknown"
        
        # ("function", ret, params)
        kind, ret_type, params = func_info

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
                 self.error(f"Function '{call_node.name}' expects at least {min_args} arguments, got {len(call_node.args)}", call_node)
        else:
            if len(call_node.args) != len(params):
                self.error(f"Function '{call_node.name}' expects {len(params)} arguments, got {len(call_node.args)}", call_node)
        
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
