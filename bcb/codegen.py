from .parser import Program, DataBlock, FunctionDecl, FunctionDef, CallExpr, ReturnStmt, VarDeclStmt, VarAssignStmt, BinaryExpr, LiteralExpr, VarRefExpr, IfStmt, WhileStmt, LabelDef, JmpStmt, IfnStmt, CmpTStmt, TypeCastExpr, UnaryExpr, StructDef, StructLiteralExpr, FieldAccessExpr, FieldAssignStmt, EnumDef, EnumValueExpr

class CodeGen:
    def __init__(self, ast):
        self.ast = ast
        self.output = []
        self.data_labels = {}
        self.float_literals = {}  # (value, type) -> label
        self.locals = {}  # name -> (offset, type)
        self.next_local_offset = 0
        self.label_count = 0
        self.structs = {}  # struct_name -> list of (field_type, field_name)
        self.struct_field_offsets = {}  # struct_name -> {field_name: offset}
        self.enums = {}  # enum_name -> {value_name: int_value}
        self.current_func_end_label = None
        self.functions = {} # name -> return_type
        self.function_params = {} # name -> params list
        self.string_literals = {} # value -> label
        # Target configuration
        self.outtype = getattr(ast, "outtype", None) or "win64"
        self.is_linux = self.outtype.lower() == "linux64"
        self.is_windows = self.outtype.lower() == "win64"

    def new_label(self, prefix="L"):
        self.label_count += 1
        return f"{prefix}_{self.label_count}"

    def register_struct(self, struct_def):
        """Register a struct definition and calculate field offsets."""
        self.structs[struct_def.name] = struct_def.fields
        offsets = {}
        current_offset = 0
        for field_type, field_name in struct_def.fields:
            offsets[field_name] = current_offset
            current_offset += self.get_type_size(field_type)
        self.struct_field_offsets[struct_def.name] = offsets

    def get_type_size(self, type_name):
        """Returns the size in bytes of a given type."""
        if type_name in ['int32', 'float32']:
            return 4
        elif type_name in ['int64', 'float64', 'string']:
            return 8
        elif type_name in ['char', 'int8']:
            return 1
        elif type_name == 'int16':
            return 2
        elif type_name in self.structs:
            # Calculate struct size
            total = 0
            for ft, _ in self.structs[type_name]:
                total += self.get_type_size(ft)
            return total
        return 8  # Default fallback

    def register_enum(self, enum_def):
        """Register an enum definition with its values mapped to integers."""
        value_map = {}
        for i, value_name in enumerate(enum_def.values):
            value_map[value_name] = i
        self.enums[enum_def.name] = value_map

    def generate(self):
        self.output.append(".intel_syntax noprefix")

        # 0. Process function declarations and definitions
        for decl in self.ast.declarations:
            if isinstance(decl, (FunctionDef, FunctionDecl)):
                self.functions[decl.name] = decl.return_type
                self.function_params[decl.name] = decl.params
        
        # 0. Process struct definitions
        if self.ast.data_block and self.ast.data_block.structs:
            for struct_def in self.ast.data_block.structs:
                self.register_struct(struct_def)
        
        # 0.5. Process enum definitions
        if self.ast.data_block and self.ast.data_block.enums:
            for enum_def in self.ast.data_block.enums:
                self.register_enum(enum_def)
        
        # 1. Populate data labels from the data block
        if self.ast.data_block:
            for type_name, name, value in self.ast.data_block.entries:
                if type_name == 'string':
                    self.data_labels[name] = f"L_{name}"

        # 2. Pre-generate functions to discover float literals
        functions_output = []
        original_output = self.output
        self.output = functions_output
        
        for decl in self.ast.declarations:
            if isinstance(decl, FunctionDef):
                self.gen_function(decl)
            elif isinstance(decl, FunctionDecl):
                pass
        
        # 3. Revert to original output and construct the final assembly
        self.output = original_output
        
        # Output the data section (strings and discovered float literals)
        if self.ast.data_block or self.float_literals or self.string_literals:
            # Windows uses .rdata, Linux typically uses .rodata
            if self.is_linux:
                self.output.append(".section .rodata")
            else:
                self.output.append(".section .rdata,\"dr\"")
            if self.ast.data_block:
                for type_name, name, value in self.ast.data_block.entries:
                    if type_name == 'string':
                        label = self.data_labels[name]
                        self.output.append(f"{label}:")
                        escaped_value = value.replace('"', '\\"')
                        self.output.append(f"    .asciz \"{escaped_value}\"")
                
            for (val, t), label in self.float_literals.items():
                self.output.append(f"{label}:")
                if t == 'float32':
                    self.output.append(f"    .float {val}")
                else:
                    self.output.append(f"    .double {val}")

            for val, label in self.string_literals.items():
                self.output.append(f"{label}:")
                escaped_value = val.replace('"', '\\"')
                self.output.append(f"    .asciz \"{escaped_value}\"")
            
        # Output the text section
        self.output.append(".text")
        self.output.extend(functions_output)

        # On Linux, emit a non-executable stack note to silence linker warnings
        if self.is_linux:
            self.output.append(".section .note.GNU-stack,\"\",@progbits")

        return "\n".join(self.output) + "\n"

    def get_struct_size(self, struct_name):
        """Returns the total size of a struct in bytes."""
        if struct_name not in self.structs:
            return 8
        total = 0
        for ft, _ in self.structs[struct_name]:
            total += self.get_type_size(ft)
        return total

    def get_field_type(self, struct_name, field_name):
        """Returns the type of a field in a struct."""
        if struct_name not in self.structs:
            return 'int32'
        for ft, fn in self.structs[struct_name]:
            if fn == field_name:
                return ft
        return 'int32'

    def gen_struct_init(self, var_name, struct_type, struct_literal):
        """Generate code to initialize a struct variable with a struct literal."""
        if var_name not in self.locals:
            return
        
        var_offset, _ = self.locals[var_name]
        field_offsets = self.struct_field_offsets.get(struct_type, {})
        
        for field_name, field_type, field_expr in struct_literal.field_values:
            if field_name not in field_offsets:
                continue
            
            field_offset = field_offsets[field_name]
            # Calculate memory location: var is at [rbp - var_offset]
            # Struct starts at that address, field is at struct_start + field_offset
            # Since var_offset is from rbp downward, and struct grows upward from the base:
            # field_addr = rbp - var_offset + field_offset
            # But actually, we stored offset as distance from rbp, so struct base is at rbp - var_offset
            # and field is at rbp - var_offset + field_offset (but we need to adjust for how we store)
            # Actually, the var_offset points to the END of the struct space (top of struct)
            # So field at offset 0 should be at rbp - var_offset + (struct_size - 4) for first int32
            # This is getting complex. Let's simplify: store struct starting at lowest address.
            
            # Simpler approach: struct starts at [rbp - var_offset] and grows upward
            # So field at offset 0 is at [rbp - var_offset]
            # field at offset 4 is at [rbp - var_offset + 4]
            struct_size = self.get_struct_size(struct_type)
            actual_field_mem_offset = var_offset - struct_size + field_offset
            
            self.gen_expression(field_expr, expected_type=field_type)
            
            if field_type == 'int32':
                self.output.append(f"    mov dword ptr [rbp - {var_offset - field_offset}], eax")
            elif field_type == 'int64':
                self.output.append(f"    mov qword ptr [rbp - {var_offset - field_offset}], rax")
            elif field_type == 'float32':
                self.output.append(f"    movss [rbp - {var_offset - field_offset}], xmm0")
            elif field_type == 'float64':
                self.output.append(f"    movsd [rbp - {var_offset - field_offset}], xmm0")
            elif field_type == 'char':
                self.output.append(f"    mov byte ptr [rbp - {var_offset - field_offset}], al")
            elif field_type == 'int8':
                self.output.append(f"    mov byte ptr [rbp - {var_offset - field_offset}], al")
            elif field_type == 'int16':
                self.output.append(f"    mov word ptr [rbp - {var_offset - field_offset}], ax")

    def gen_function(self, func):
        self.locals = {}
        self.next_local_offset = 0
        self.current_func_end_label = self.new_label(f"E_{func.name}")
        self.current_func_return_type = func.return_type
        
        if func.is_exported:
            self.output.append(f".globl {func.name}")
        
        self.output.append(f"{func.name}:")
        self.output.append("    push rbp")
        self.output.append("    mov rbp, rsp")
        self.output.append("    sub rsp, 64")

        # Calling convention: choose argument registers by target
        if self.is_linux:
            # System V AMD64: rdi, rsi, rdx, rcx, r8, r9 for ints
            # xmm0-xmm7 for floats
            int_arg_regs = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
            float_arg_regs = [f"xmm{i}" for i in range(8)]
            int_arg_idx = 0
            float_arg_idx = 0
        else:
            # Windows x64: rcx, rdx, r8, r9 shared
            arg_regs = ["rcx", "rdx", "r8", "r9"]
            float_arg_regs = ["xmm0", "xmm1", "xmm2", "xmm3"]
        
        for i, (pname, ptype) in enumerate(func.params):
            self.next_local_offset += 8
            self.locals[pname] = (self.next_local_offset, ptype)
            
            is_float = ptype in ['float32', 'float64']
            
            if self.is_linux:
                if is_float:
                    if float_arg_idx < len(float_arg_regs):
                        reg = float_arg_regs[float_arg_idx]
                        if ptype == 'float32':
                            self.output.append(f"    movss [rbp - {self.next_local_offset}], {reg}")
                        else:
                            self.output.append(f"    movsd [rbp - {self.next_local_offset}], {reg}")
                        float_arg_idx += 1
                    else:
                        pass # TODO: Implement Linux stack args
                else:
                    if int_arg_idx < len(int_arg_regs):
                        self.output.append(f"    mov [rbp - {self.next_local_offset}], {int_arg_regs[int_arg_idx]}")
                        int_arg_idx += 1
                    else:
                        pass # TODO: Stack args
            else:
                # Windows
                if i < len(arg_regs):
                    if is_float:
                         if ptype == 'float32':
                             self.output.append(f"    movss [rbp - {self.next_local_offset}], {float_arg_regs[i]}")
                         else:
                             self.output.append(f"    movsd [rbp - {self.next_local_offset}], {float_arg_regs[i]}")
                    else:
                        self.output.append(f"    mov [rbp - {self.next_local_offset}], {arg_regs[i]}")
                else:
                    stack_offset = 48 + (i - 4) * 8
                    self.output.append(f"    mov rax, [rbp + {stack_offset}]")
                    self.output.append(f"    mov [rbp - {self.next_local_offset}], rax")

        for stmt in func.body:
            self.gen_statement(stmt)

        self.output.append(f"{self.current_func_end_label}:")
        self.output.append("    add rsp, 64")
        self.output.append("    pop rbp")
        self.output.append("    ret")

    def gen_statement(self, stmt):
        if isinstance(stmt, CallExpr):
            self.gen_call(stmt)
        elif isinstance(stmt, ReturnStmt):
            if stmt.return_type != 'void':
                actual_type = self.gen_expression(stmt.expr, expected_type=stmt.return_type)
                # Ensure it matches the function's declared return type
                if hasattr(self, 'current_func_return_type') and self.current_func_return_type != actual_type:
                    self.gen_conversion(actual_type, self.current_func_return_type)
            if self.current_func_end_label:
                self.output.append(f"    jmp {self.current_func_end_label}")

        elif isinstance(stmt, VarDeclStmt):
            # Check if this is a struct type
            if stmt.type_name in self.structs:
                struct_size = self.get_struct_size(stmt.type_name)
                # Align to 8 bytes
                aligned_size = ((struct_size + 7) // 8) * 8
                self.next_local_offset += aligned_size
                self.locals[stmt.name] = (self.next_local_offset, stmt.type_name)
                
                # Generate struct literal initialization
                if isinstance(stmt.expr, StructLiteralExpr):
                    self.gen_struct_init(stmt.name, stmt.type_name, stmt.expr)
            else:
                actual_type = self.gen_expression(stmt.expr, expected_type=None)
                self.next_local_offset += 8
                self.locals[stmt.name] = (self.next_local_offset, stmt.type_name)
                
                # Helper to store result based on actual vs target type
                if stmt.type_name in ['float32', 'float64']:
                    self.gen_conversion(actual_type, stmt.type_name)
                    if stmt.type_name == 'float32':
                        self.output.append(f"    movss [rbp - {self.next_local_offset}], xmm0")
                    else:
                        self.output.append(f"    movsd [rbp - {self.next_local_offset}], xmm0")
                else:
                    self.gen_conversion(actual_type, stmt.type_name)
                    self.output.append(f"    mov [rbp - {self.next_local_offset}], rax")

        elif isinstance(stmt, VarAssignStmt):
            # Pointer mutation via: md int32* ptr = value;  =>  *ptr = value
            if isinstance(stmt.type_name, str) and stmt.type_name.endswith('*'):
                base_type = stmt.type_name[:-1]
                if stmt.name in self.locals:
                    ptr_offset, ptr_type = self.locals[stmt.name]
                    # Load pointer value from local variable (use caller-saved register)
                    self.output.append(f"    mov rdx, [rbp - {ptr_offset}]")
                    actual_type = self.gen_expression(stmt.expr, expected_type=base_type if base_type else None)

                    # Store into *ptr according to base_type
                    if base_type == 'float32':
                        self.gen_conversion(actual_type, 'float32')
                        self.output.append("    movss [rdx], xmm0")
                    elif base_type == 'float64':
                        self.gen_conversion(actual_type, 'float64')
                        self.output.append("    movsd [rdx], xmm0")
                    elif base_type == 'int32':
                        # Convert to int64 then store lower 32 bits
                        self.gen_conversion(actual_type, 'int64')
                        self.output.append("    mov dword ptr [rdx], eax")
                    elif base_type == 'char':
                        self.gen_conversion(actual_type, 'int64') # char is int-like
                        self.output.append("    mov byte ptr [rdx], al")
                    elif base_type == 'int8':
                        self.gen_conversion(actual_type, 'int64')
                        self.output.append("    mov byte ptr [rdx], al")
                    elif base_type == 'int16':
                        self.gen_conversion(actual_type, 'int64')
                        self.output.append("    mov word ptr [rdx], ax")
                    else:
                        # Default to 64-bit store for int64, pointers, or unknown base
                        self.gen_conversion(actual_type, 'int64')
                        self.output.append("    mov qword ptr [rdx], rax")
                return

            # Regular variable assignment
            expected_type = None  # Default logic as requested
            actual_type = self.gen_expression(stmt.expr, expected_type=None)

            if stmt.name in self.locals:
                offset, t = self.locals[stmt.name]
                if t in ['float32', 'float64']:
                    self.gen_conversion(actual_type, t)
                    if t == 'float32':
                        self.output.append(f"    movss [rbp - {offset}], xmm0")
                    else:
                        self.output.append(f"    movsd [rbp - {offset}], xmm0")
                else:
                    self.gen_conversion(actual_type, t)
                    self.output.append(f"    mov [rbp - {offset}], rax")
        elif isinstance(stmt, IfStmt):
            self.gen_if_stmt(stmt)
        elif isinstance(stmt, WhileStmt):
            self.gen_while_stmt(stmt)
        elif isinstance(stmt, LabelDef):
            self.output.append(f"{stmt.name[1:]}:")
        elif isinstance(stmt, JmpStmt):
            self.output.append(f"    jmp {stmt.target[1:]}")
        elif isinstance(stmt, IfnStmt):
            if stmt.condition in self.locals:
                offset, t = self.locals[stmt.condition]
                self.output.append(f"    cmp qword ptr [rbp - {offset}], 0")
                self.output.append(f"    je {stmt.target[1:]}")
        elif isinstance(stmt, CmpTStmt):
            if stmt.condition in self.locals:
                offset, t = self.locals[stmt.condition]
                self.output.append(f"    cmp qword ptr [rbp - {offset}], 0")
                self.output.append(f"    jne {stmt.target[1:]}")
        elif isinstance(stmt, FieldAssignStmt):
            # Handle field assignment: md type var.field = expr
            if stmt.var_name in self.locals:
                var_offset, struct_type = self.locals[stmt.var_name]
                
                if struct_type in self.structs:
                    field_offsets = self.struct_field_offsets.get(struct_type, {})
                    field_offset = field_offsets.get(stmt.field_name, 0)
                    field_type = self.get_field_type(struct_type, stmt.field_name)
                    
                    # Generate the expression value
                    self.gen_expression(stmt.expr, expected_type=field_type)
                    
                    # Store the value in the field
                    if field_type == 'int32':
                        self.output.append(f"    mov dword ptr [rbp - {var_offset - field_offset}], eax")
                    elif field_type == 'int64':
                        self.output.append(f"    mov qword ptr [rbp - {var_offset - field_offset}], rax")
                    elif field_type == 'float32':
                        self.output.append(f"    movss [rbp - {var_offset - field_offset}], xmm0")
                    elif field_type == 'float64':
                        self.output.append(f"    movsd [rbp - {var_offset - field_offset}], xmm0")
                    elif field_type == 'char':
                         self.output.append(f"    mov byte ptr [rbp - {var_offset - field_offset}], al")
                    elif field_type == 'int8':
                         self.output.append(f"    mov byte ptr [rbp - {var_offset - field_offset}], al")
                    elif field_type == 'int16':
                         self.output.append(f"    mov word ptr [rbp - {var_offset - field_offset}], ax")
    def gen_if_stmt(self, stmt):
        end_label = self.new_label("if_end")
        
        for i, (condition, body) in enumerate(stmt.conditions_and_bodies):
            next_part_label = self.new_label(f"if_part_{i+1}")
            
            if condition:
                self.gen_expression(condition)
                # Comparison operators should be handled in gen_expression
                # and leave a result in RAX or flags. 
                # For simplicity, let's assume gen_expression for comparisons 
                # leaves result in RAX and we'll check it.
                # Actually, it's better to make BinaryExpr for comparisons 
                # use CMP and we handle the jump here.
                
                # If the last thing in gen_expression was a comparison, 
                # we can use the flags.
                # Let's adjust gen_binary_expr to not just 'add/sub' but also 'cmp'.
                
                # Handle condition
                if isinstance(condition, BinaryExpr) and condition.op in ['>', '<', '==', '!=']:
                    # We'll generate the cmp in gen_expression
                    # but we need to know WHICH jump to use.
                    # This is tricky without a better AST or state.
                    # Let's just have gen_expression return the operator if it was a comparison.
                    pass
                
                # Simple implementation: gen_expression results in 1 or 0 in RAX
                self.output.append("    cmp rax, 0")
                self.output.append(f"    je {next_part_label}")
            
            for s in body:
                self.gen_statement(s)
            
            self.output.append(f"    jmp {end_label}")
            self.output.append(f"{next_part_label}:")
            
        self.output.append(f"{end_label}:")

    def gen_while_stmt(self, stmt):
        start_label = self.new_label("while_start")
        end_label = self.new_label("while_end")
        
        self.output.append(f"{start_label}:")
        self.gen_expression(stmt.condition)
        self.output.append("    cmp rax, 0")
        self.output.append(f"    je {end_label}")
        
        for s in stmt.body:
            self.gen_statement(s)
            
        self.output.append(f"    jmp {start_label}")
        self.output.append(f"{end_label}:")

    def gen_expression(self, expr, expected_type=None):
        # 1. Handle UnaryExpr
        if isinstance(expr, UnaryExpr):
            # Address-of: &var  -> pointer to local variable
            if expr.op == '&':
                if isinstance(expr.expr, VarRefExpr) and expr.expr.name in self.locals:
                    offset, t = self.locals[expr.expr.name]
                    self.output.append(f"    lea rax, [rbp - {offset}]")
                    # Represent pointer type as "<base>*" so size logic still treats it as int-like
                    return f"{t}*"
                else:
                    raise RuntimeError("& operator is only supported on local variables")

            # Dereference: *ptr  -> load value from address in rax
            if expr.op == '*':
                ptr_type = self.gen_expression(expr.expr, expected_type=None)
                # Expect a pointer-like type string, e.g., "int32*"
                base_type = None
                if isinstance(ptr_type, str) and ptr_type.endswith('*'):
                    base_type = ptr_type[:-1] or None

                # Decide the target type to load as
                target_type = expected_type or base_type or 'int64'

                if target_type == 'float32':
                    self.output.append("    movss xmm0, [rax]")
                elif target_type == 'float64':
                    self.output.append("    movsd xmm0, [rax]")
                elif target_type == 'int32':
                    # Sign-extend 32-bit int to 64-bit
                    self.output.append("    movsxd rax, dword ptr [rax]")
                elif target_type == 'char':
                    self.output.append("    movzx rax, byte ptr [rax]")
                elif target_type == 'int8':
                    self.output.append("    movsx rax, byte ptr [rax]")
                elif target_type == 'int16':
                    self.output.append("    movsx rax, word ptr [rax]")
                else:
                    # Default 64-bit load
                    self.output.append("    mov rax, [rax]")
                return target_type

            # Other unary ops usually preserve type or imply int. 
            # For simplicity, let's defer to inner, currently only '~' (int)
            # If expected is None, assume int.
            req_type = expected_type if expected_type else 'int64'
            self.gen_expression(expr.expr, expected_type=req_type)
            if expr.op == '~':
                self.output.append("    not rax")
            return req_type

        # 2. Handle TypeCastExpr
        if isinstance(expr, TypeCastExpr):
            target = expr.target_type
            # Generate inner with target type
            self.gen_expression(expr.expr, expected_type=target)
            
            # Result is now in target type (xmm0 or rax)
            # If expected_type is specified and different, convert.
            # If expected_type is None, return as target type.
            
            if expected_type is not None and expected_type != target:
                self.gen_conversion(target, expected_type)
                return expected_type
            return target

        # 3. Handle Literals
        if isinstance(expr, LiteralExpr):
            if isinstance(expr.value, float):
                # Native float
                final_type = 'float64' # Default for literal float
                if expected_type == 'float32': final_type = 'float32'
                
                # If caller wants int, we have to load as float then convert?
                # Or load int representation? LiteralExpr logic handles loading.
                
                # Let's rely on existing logic but adapted
                t = final_type
                if (expr.value, t) not in self.float_literals:
                    label = self.new_label("float_lit")
                    self.float_literals[(expr.value, t)] = label
                label = self.float_literals[(expr.value, t)]
                
                if t == 'float32':
                    self.output.append(f"    movss xmm0, [rip + {label}]")
                else:
                    self.output.append(f"    movsd xmm0, [rip + {label}]")
                
                if expected_type is not None and expected_type not in ['float32', 'float64']:
                    # Caller wants int
                    self.gen_conversion(t, expected_type)
                    return expected_type
                return t
            elif isinstance(expr.value, int):
                # Native int
                # Use hex to ensure assembler handles 64-bit immediates correctly
                val_hex = hex(expr.value)
                self.output.append(f"    mov rax, {val_hex}")
                
                if expected_type in ['float32', 'float64']:
                    self.gen_conversion('int64', expected_type)
                    return expected_type
                return 'int64'
            elif isinstance(expr.value, str):
                # String literal
                if expr.value not in self.string_literals:
                    label = self.new_label("str_lit")
                    self.string_literals[expr.value] = label
                label = self.string_literals[expr.value]
                self.output.append(f"    lea rax, [rip + {label}]")
                return 'string'

        # 4. Handle VarRef
        elif isinstance(expr, VarRefExpr):
            if expr.name in self.locals:
                offset, t = self.locals[expr.name]
                native_type = t
                
                # Load variable
                if t == 'float32':
                    self.output.append(f"    movss xmm0, [rbp - {offset}]")
                elif t == 'float64':
                    self.output.append(f"    movsd xmm0, [rbp - {offset}]")
                else:
                    self.output.append(f"    mov rax, [rbp - {offset}]")
                
                # Convert if needed
                if expected_type is not None and expected_type != t:
                    self.gen_conversion(t, expected_type)
                    return expected_type
                return t
            elif expr.name in self.data_labels:
                label = self.data_labels[expr.name]
                self.output.append(f"    lea rax, [rip + {label}]")
                return 'int64'  # pointers are ints

        # 4.5. Handle FieldAccessExpr (e.g., p.x)
        elif isinstance(expr, FieldAccessExpr):
            # Get the struct variable and its type
            if isinstance(expr.obj, VarRefExpr):
                var_name = expr.obj.name
                if var_name in self.locals:
                    var_offset, struct_type = self.locals[var_name]
                    
                    if struct_type in self.structs:
                        field_offsets = self.struct_field_offsets.get(struct_type, {})
                        field_offset = field_offsets.get(expr.field_name, 0)
                        field_type = self.get_field_type(struct_type, expr.field_name)
                        
                        # Load the field value
                        if field_type == 'int32':
                            self.output.append(f"    movsxd rax, dword ptr [rbp - {var_offset - field_offset}]")
                        elif field_type == 'int64':
                            self.output.append(f"    mov rax, [rbp - {var_offset - field_offset}]")
                        elif field_type == 'float32':
                            self.output.append(f"    movss xmm0, [rbp - {var_offset - field_offset}]")
                        elif field_type == 'float64':
                            self.output.append(f"    movsd xmm0, [rbp - {var_offset - field_offset}]")
                        elif field_type == 'char':
                            self.output.append(f"    movzx rax, byte ptr [rbp - {var_offset - field_offset}]")
                        elif field_type == 'int8':
                            self.output.append(f"    movsx rax, byte ptr [rbp - {var_offset - field_offset}]")
                        elif field_type == 'int16':
                            self.output.append(f"    movsx rax, word ptr [rbp - {var_offset - field_offset}]")
                        
                        # Convert if needed
                        if expected_type is not None and expected_type != field_type:
                            self.gen_conversion(field_type, expected_type)
                            return expected_type
                        return field_type
        # 4.6. Handle EnumValueExpr (e.g., Color.RED)
        elif isinstance(expr, EnumValueExpr):
            if expr.enum_name in self.enums:
                val = self.enums[expr.enum_name].get(expr.value_name, 0)
                self.output.append(f"    mov rax, {val}")
                
                if expected_type in ['float32', 'float64']:
                    self.gen_conversion('int64', expected_type)
                    return expected_type
                return 'int64'
            return 'int64'

        # 5. Handle Calls
        elif isinstance(expr, CallExpr):
            self.gen_call(expr)
            ret_type = self.functions.get(expr.name, 'int64')
            
            if expected_type is not None and expected_type != ret_type:
                 self.gen_conversion(ret_type, expected_type)
                 return expected_type
            return ret_type

        # 6. Handle BinaryExpr
        elif isinstance(expr, BinaryExpr):
            is_comparison = expr.op in ['==', '!=', '<', '>', '<=', '>=']
            
            # Determine expected type for LHS
            # For comparisons, we do not impose the result type (int/bool) on the operands.
            # For arithmetic, we propagate expected_type if available.
            lhs_expected = None if is_comparison else expected_type

            # Generate LHS
            # This returns the actual type of the left operand
            left_type = self.gen_expression(expr.left, expected_type=lhs_expected)
            
            # Determine if we are doing float operation based on LHS or if explicitly requested
            is_float_op = left_type in ['float32', 'float64']
            
            # RHS Expected Type
            # If float op, we want RHS to match float type. 
            # If int op, we default to int64 (or lhs_expected if it was an int type, but defaulting to int64 covers all)
            rhs_expected = left_type if is_float_op else (lhs_expected if lhs_expected else 'int64')
            
            if is_float_op:
                # Float Op
                self.output.append("    sub rsp, 16")
                if left_type == 'float32':
                    self.output.append("    movss [rsp], xmm0")
                else:
                    self.output.append("    movsd [rsp], xmm0")
                
                # Generate RHS
                self.gen_expression(expr.right, expected_type=rhs_expected)
                
                # Move RHS to xmm1, restore LHS to xmm0
                if left_type == 'float32':
                    self.output.append("    movss xmm1, xmm0")
                    self.output.append("    movss xmm0, [rsp]")
                else:
                    self.output.append("    movsd xmm1, xmm0")
                    self.output.append("    movsd xmm0, [rsp]")
                
                self.output.append("    add rsp, 16")
                
                if is_comparison:
                    if left_type == 'float32':
                        self.output.append("    ucomiss xmm0, xmm1")
                    else:
                        self.output.append("    ucomisd xmm0, xmm1")
                    
                    # Set AL based on flags
                    # ucomiss sets ZF,PF,CF.
                    # Equal: ZF=1, PF=0 (Unordered sets PF=1)
                    
                    if expr.op == '==':
                        # Check (ZF=1 AND PF=0)
                        self.output.append("    setnp al") # PF=0
                        self.output.append("    sete bl")  # ZF=1
                        self.output.append("    and al, bl")
                        
                    elif expr.op == '!=':
                        # Check (ZF=0 OR PF=1)
                        self.output.append("    setp al")
                        self.output.append("    setne bl")
                        self.output.append("    or al, bl")
                        
                    elif expr.op == '<':
                        self.output.append("    seta al") # Wait, xmm0 < xmm1 -> CF=1 (below). 'seta' is above.
                        # Intel syntax: ucomiss src, dest ?? No, ucomiss op1, op2.
                        # cmp op1, op2. op1 < op2 -> carry (below).
                        # ucomiss xmm0, xmm1.
                        # if xmm0 < xmm1 -> CF=1.
                        self.output.append("    setb al")
                        
                    elif expr.op == '>':
                        self.output.append("    seta al")
                        
                    elif expr.op == '<=':
                        self.output.append("    setbe al")
                        
                    elif expr.op == '>=':
                        self.output.append("    setae al")
                        
                    self.output.append("    movzx rax, al")
                    return 'int64'
                
                else:
                    # Arithmetic
                    if expr.op == '+':
                        if left_type == 'float32': self.output.append("    addss xmm0, xmm1")
                        else: self.output.append("    addsd xmm0, xmm1")
                    elif expr.op == '-':
                        if left_type == 'float32': self.output.append("    subss xmm0, xmm1")
                        else: self.output.append("    subsd xmm0, xmm1")
                    elif expr.op == '*':
                        if left_type == 'float32': self.output.append("    mulss xmm0, xmm1")
                        else: self.output.append("    mulsd xmm0, xmm1")
                    elif expr.op == '/':
                        if left_type == 'float32': self.output.append("    divss xmm0, xmm1")
                        else: self.output.append("    divsd xmm0, xmm1")
                    
                    return left_type
            
            else:
                # Int Math
                self.output.append("    push rax")
                self.gen_expression(expr.right, expected_type='int64')
                self.output.append("    mov rbx, rax")
                self.output.append("    pop rax")
                
                if expr.op == '+':
                    self.output.append("    add rax, rbx")
                elif expr.op == '-':
                    self.output.append("    sub rax, rbx")
                elif expr.op == '*':
                    self.output.append("    imul rax, rbx")
                elif expr.op == '/':
                    self.output.append("    cqo") 
                    self.output.append("    idiv rbx")
                elif expr.op == '%':
                    self.output.append("    cqo") 
                    self.output.append("    idiv rbx")
                    self.output.append("    mov rax, rdx")
                elif expr.op == '&':
                    self.output.append("    and rax, rbx")
                elif expr.op == '|':
                    self.output.append("    or rax, rbx")
                elif expr.op == '^':
                    self.output.append("    xor rax, rbx")
                elif expr.op == '<<':
                    self.output.append("    mov rcx, rbx")
                    self.output.append("    shl rax, cl")
                elif expr.op == '>>':
                    self.output.append("    mov rcx, rbx")
                    self.output.append("    sar rax, cl")
                elif expr.op in ['==', '!=', '<', '>', '<=', '>=']:
                    self.output.append("    cmp rax, rbx")
                    if expr.op == '==': self.output.append("    sete al")
                    elif expr.op == '!=': self.output.append("    setne al")
                    elif expr.op == '<': self.output.append("    setl al")
                    elif expr.op == '>': self.output.append("    setg al")
                    elif expr.op == '<=': self.output.append("    setle al")
                    elif expr.op == '>=': self.output.append("    setge al")
                    self.output.append("    movzx rax, al")
                elif expr.op == '&&':
                    self.output.append("    cmp rax, 0")
                    self.output.append("    setne al")
                    self.output.append("    cmp rbx, 0")
                    self.output.append("    setne bl")
                    self.output.append("    and al, bl")
                    self.output.append("    movzx rax, al")
                elif expr.op == '||':
                    self.output.append("    cmp rax, 0")
                    self.output.append("    setne al")
                    self.output.append("    cmp rbx, 0")
                    self.output.append("    setne bl")
                    self.output.append("    or al, bl")
                    self.output.append("    movzx rax, al")
                
                return 'int64'

    def gen_conversion(self, from_type, to_type):
        if from_type == to_type: return

        # Treat char as int64 for conversion purposes
        if from_type == 'char': from_type = 'int64'
        if to_type == 'char': to_type = 'int64'
        if from_type in ['int8', 'int16']: from_type = 'int64'
        if to_type in ['int8', 'int16']: to_type = 'int64'
        
        if from_type == to_type: return
        
        # Float to Int
        if from_type in ['float32', 'float64'] and to_type not in ['float32', 'float64']:
            if from_type == 'float32':
                self.output.append("    cvttss2si rax, xmm0")
            else:
                self.output.append("    cvttsd2si rax, xmm0")
        
        # Int to Float
        elif from_type not in ['float32', 'float64'] and to_type in ['float32', 'float64']:
            if to_type == 'float32':
                self.output.append("    cvtsi2ss xmm0, rax")
            else:
                self.output.append("    cvtsi2sd xmm0, rax")
                
        # Float to Float
        elif from_type in ['float32', 'float64'] and to_type in ['float32', 'float64']:
            if from_type == 'float32' and to_type == 'float64':
                self.output.append("    cvtss2sd xmm0, xmm0")
            elif from_type == 'float64' and to_type == 'float32':
                self.output.append("    cvtsd2ss xmm0, xmm0")

    def gen_call(self, expr):
        func_params = self.function_params.get(expr.name)

        # Select integer argument registers based on target calling convention
        if self.is_linux:
            # System V AMD64
            int_arg_regs = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
            float_arg_regs = [f"xmm{i}" for i in range(8)]
            int_arg_idx = 0
            float_arg_idx = 0
            
            for i, (arg_type, arg_expr) in enumerate(expr.args):
                is_float = arg_type in ['float32', 'float64']
                
                # Determine promotion for float32
                should_promote = False
                if arg_type == 'float32':
                    if func_params is None:
                        should_promote = True
                    elif i >= len(func_params):
                        should_promote = True
                    else:
                        _, param_type = func_params[i]
                        if param_type.startswith('...'):
                            should_promote = True
                
                if is_float:
                    self.gen_expression(arg_expr, expected_type=arg_type)
                    
                    if should_promote:
                        self.output.append("    cvtss2sd xmm0, xmm0")
                        
                    # Pass in XMM register
                    if float_arg_idx < len(float_arg_regs):
                        reg = float_arg_regs[float_arg_idx]
                        if arg_type == 'float32' and not should_promote:
                            self.output.append(f"    movss {reg}, xmm0")
                        else:
                            # float64 or promoted float32
                            self.output.append(f"    movsd {reg}, xmm0")
                        float_arg_idx += 1
                else:
                    self.gen_expression(arg_expr, expected_type=arg_type)
                    # Pass in Integer register
                    if int_arg_idx < len(int_arg_regs):
                        reg = int_arg_regs[int_arg_idx]
                        if arg_type == 'string' and isinstance(arg_expr, VarRefExpr) and arg_expr.name in self.data_labels:
                            label = self.data_labels[arg_expr.name]
                            self.output.append(f"    lea {reg}, [rip + {label}]")
                        else:
                            self.output.append(f"    mov {reg}, rax")
                        int_arg_idx += 1
                        
            # For varargs (like printf), AL must contain number of vector registers used (Linux ABI)
            self.output.append(f"    mov al, {float_arg_idx}")
            
        else:
            # Windows x64
            arg_regs = ["rcx", "rdx", "r8", "r9"]
            float_arg_regs = ["xmm0", "xmm1", "xmm2", "xmm3"]
            
            for i, (arg_type, arg_expr) in enumerate(expr.args):
                is_float = arg_type in ['float32', 'float64']
                
                # Determine promotion for float32
                should_promote = False
                if arg_type == 'float32':
                    if func_params is None:
                        should_promote = True
                    elif i >= len(func_params):
                        should_promote = True
                    else:
                        _, param_type = func_params[i]
                        if param_type.startswith('...'):
                            should_promote = True

                if is_float:
                    self.gen_expression(arg_expr, expected_type=arg_type)
                    
                    if should_promote:
                        self.output.append("    cvtss2sd xmm0, xmm0")

                    if i < len(float_arg_regs):
                        if arg_type == 'float32' and not should_promote:
                             self.output.append(f"    movss {float_arg_regs[i]}, xmm0")
                        else:
                             self.output.append(f"    movsd {float_arg_regs[i]}, xmm0")
                        
                        # Windows Variadic: Mirror float in GPR
                        self.output.append(f"    movq {arg_regs[i]}, xmm0")
                else:
                    self.gen_expression(arg_expr, expected_type=arg_type)
                    if i < len(arg_regs):
                        if arg_type == 'string' and isinstance(arg_expr, VarRefExpr) and arg_expr.name in self.data_labels:
                            label = self.data_labels[arg_expr.name]
                            self.output.append(f"    lea {arg_regs[i]}, [rip + {label}]")
                        else:
                            self.output.append(f"    mov {arg_regs[i]}, rax")

        # Allocate shadow space (32 bytes) only for Windows x64 ABI
        if self.is_windows:
            self.output.append("    sub rsp, 32")
        self.output.append(f"    call {expr.name}")
        if self.is_windows:
            self.output.append("    add rsp, 32")
