from .parser import Program, DataBlock, FunctionDecl, FunctionDef, GlobalVarDecl, CallExpr, ReturnStmt, VarDeclStmt, VarAssignStmt, BinaryExpr, LiteralExpr, VarRefExpr, IfStmt, WhileStmt, LabelDef, JmpStmt, IfnStmt, CmpTStmt, TypeCastExpr, UnaryExpr, StructDef, StructLiteralExpr, FieldAccessExpr, FieldAssignStmt, EnumDef, EnumValueExpr, PushStmt, PopStmt, SwapStmt, DupStmt, NoValueExpr, ArrayAccessExpr, ArrayLiteralExpr, ArrayAssignStmt, LengthExpr, GetTypeExpr, ArgsAccessExpr

class CodeGen:
    def __init__(self, ast, optimization_level: int = 3):
        self.ast = ast
        self.optimization_level = optimization_level  # 0-3, higher = more aggressive
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
        self.globals = {} # name -> {'label': str, 'type': str, 'init': ASTNode}
        self.internal_functions = set()
        self.variadic_info = {} # func_name -> (param_name, start_index)
        self.pending_type_arrays = {} # label -> list of type strings
        # Target configuration
        self.outtype = getattr(ast, "outtype", None) or "win64"
        self.is_linux = self.outtype.lower() == "linux64"
        self.is_windows = self.outtype.lower() == "win64"
        # Optimization tracking
        self._const_fold_cache = {}  # Cache for constant folding
        self._reg_hints = {}  # Register allocation hints

    def new_label(self, prefix="L"):
        self.label_count += 1
        return f"{prefix}_{self.label_count}"

    def resolve_array_size(self, array_size):
        """Resolve array size to an integer value or generate runtime code.
        
        Args:
            array_size: Either an int (literal size) or str (variable name)
            
        Returns:
            int: The resolved array size (for compile-time constants)
            tuple: (None, var_name) for runtime resolution
            
        Raises:
            RuntimeError: If the variable name cannot be resolved
        """
        if isinstance(array_size, int):
            return array_size
        elif isinstance(array_size, str):
            # It's a variable name - look it up in locals or globals
            if array_size in self.locals:
                # Return a tuple indicating runtime resolution is needed
                return (None, array_size)
            elif array_size in self.globals:
                # Check if the global has a constant value
                global_info = self.globals[array_size]
                if isinstance(global_info.get('init'), LiteralExpr) and isinstance(global_info['init'].value, int):
                    return global_info['init'].value
                # For non-constant globals, we need runtime resolution
                return (None, array_size)
            else:
                raise RuntimeError(f"Unknown variable '{array_size}' used as array size")
        else:
            raise RuntimeError(f"Invalid array size type: {type(array_size)}")

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
        elif type_name.endswith(']'):
            # Array type: type[size]
            open_bracket = type_name.rfind('[')
            base_type = type_name[:open_bracket]
            size_str = type_name[open_bracket+1:-1]
            try:
                size = int(size_str)
            except ValueError:
                return 8 # pointer? or dynamic?
            return self.get_type_size(base_type) * size
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
                if isinstance(decl, FunctionDef):
                    self.internal_functions.add(decl.name)
                    # Check for variadic parameter
                    for i, (pname, ptype) in enumerate(decl.params):
                        if ptype.startswith("..."):
                            self.variadic_info[decl.name] = (pname, i)
                            break
        
        # 0. Process struct definitions
        if self.ast.data_block and self.ast.data_block.structs:
            for struct_def in self.ast.data_block.structs:
                self.register_struct(struct_def)
        
        # 0.5. Process enum definitions
        if self.ast.data_block and self.ast.data_block.enums:
            for enum_def in self.ast.data_block.enums:
                self.register_enum(enum_def)
        
        # 0.6. Process Global Variables
        for decl in self.ast.declarations:
            if isinstance(decl, GlobalVarDecl):
                label = f"G_{decl.name}"
                type_name = decl.type_name
                # Arrays are handled differently in bcb? 
                # For now let's assume they are similar to local but in .data
                if decl.is_array:
                    # Resolve array size (could be int or variable name)
                    resolved_size = self.resolve_array_size(decl.array_size)
                    type_name += f"[{resolved_size}]"
                self.globals[decl.name] = {'label': label, 'type': type_name, 'init': decl.expr}
        
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
        
        # First, collect all type strings from pending type arrays into string_literals
        # This ensures they are emitted in .rodata before we reference them in .data
        for label, types in self.pending_type_arrays.items():
            for t in types:
                if t not in self.string_literals:
                    sl_label = self.new_label("type_str")
                    self.string_literals[t] = sl_label

        # Output the data section (strings and discovered float literals)
        if self.ast.data_block or self.float_literals or self.string_literals:
            # Windows uses .rdata, Linux typically uses .rodata
            if self.is_linux:
                self.output.append(".section .rodata")
            else:
                self.output.append(".section .rdata")
            if self.ast.data_block:
                for type_name, name, value in self.ast.data_block.entries:
                    if type_name == 'string':
                        label = self.data_labels[name]
                        self.output.append(f"{label}:")
                        escaped_value = value.replace('"', '\\"').replace('\n', '\\n')
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

        # Output type arrays in .data section (they need runtime relocations for PIE)
        if self.pending_type_arrays:
            self.output.append(".section .data")
            for label, types in self.pending_type_arrays.items():
                self.output.append(f"{label}:")
                for t in types:
                    sl_label = self.string_literals[t]
                    self.output.append(f"    .quad {sl_label}")

        # Output global variables in .data section
        if self.globals:
            self.output.append(".section .data")
            
            for name, info in self.globals.items():
                label = info['label']
                t = info['type']
                expr = info['init']
                self.output.append(f"{label}:")
                self.emit_global_constant(t, expr)
            
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
        
        # Find actual field types from definition
        real_field_types = {fn: ft for ft, fn in self.structs.get(struct_type, [])}
        
        for field_name, field_type_unused, field_expr in struct_literal.field_values:
            if field_name not in field_offsets:
                continue
            
            field_type = real_field_types.get(field_name, field_type_unused)
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
            
            if field_type.endswith(']') and isinstance(field_expr, ArrayLiteralExpr):
                 # Array initialization in struct
                 open_bracket = field_type.rfind('[')
                 base_type = field_type[:open_bracket]
                 element_size = self.get_type_size(base_type)
                 
                 for i, val_expr in enumerate(field_expr.values):
                     # Calculate offset for this element
                     # field starts at [rbp - (var_offset - field_offset)]
                     # element i is at + (i * element_size)
                     # So address is rbp - var_offset + field_offset + i*sz
                     # or rbp - (var_offset - field_offset - i*sz)
                     
                     current_offset_from_rbp = var_offset - field_offset - (i * element_size)
                     mem_loc = f"[rbp - {current_offset_from_rbp}]"
                     
                     self.gen_expression(val_expr, expected_type=base_type)
                     
                     if base_type == 'float32':
                         self.output.append(f"    movss {mem_loc}, xmm0")
                     elif base_type == 'float64':
                         self.output.append(f"    movsd {mem_loc}, xmm0")
                     elif base_type == 'int32':
                         self.output.append(f"    mov dword ptr {mem_loc}, eax")
                     elif base_type == 'int64' or base_type.endswith('*') or base_type == 'string' or base_type in self.enums:
                         self.output.append(f"    mov qword ptr {mem_loc}, rax")
                     elif base_type == 'char' or base_type == 'int8':
                         self.output.append(f"    mov byte ptr {mem_loc}, al")
                     elif base_type == 'int16':
                         self.output.append(f"    mov word ptr {mem_loc}, ax")
                     elif base_type in self.structs:
                         # Struct copy inside array
                         sz = self.get_struct_size(base_type)
                         self.output.append("    mov rsi, rax")
                         self.output.append(f"    lea rdi, {mem_loc}")
                         self.output.append(f"    mov rcx, {sz}")
                         self.output.append("    rep movsb")
                 continue
            
            # Variables or other expressions assigned to fixed-size array field
            # If target is fixed-size array, we should copy contents if it's an array expression
            if '[' in field_type and not field_type.endswith('[]'):
                field_size = self.get_type_size(field_type)
                
                # Evaluate expression (returns pointer to the array data)
                self.gen_expression(field_expr, expected_type=field_type)
                
                # Copy from rax (source address) to field address
                self.output.append("    mov rsi, rax")
                self.output.append(f"    lea rdi, [rbp - {var_offset - field_offset}]")
                self.output.append(f"    mov rcx, {field_size}")
                self.output.append("    rep movsb")
                continue

            self.gen_expression(field_expr, expected_type=field_type)
            
            mem_loc = f"[rbp - {var_offset - field_offset}]"
            
            if field_type == 'int32':
                self.output.append(f"    mov dword ptr {mem_loc}, eax")
            elif field_type == 'int64' or field_type == 'string' or field_type in self.enums or field_type.endswith('*'):
                self.output.append(f"    mov qword ptr {mem_loc}, rax")
            elif field_type == 'float32':
                self.output.append(f"    movss {mem_loc}, xmm0")
            elif field_type == 'float64':
                self.output.append(f"    movsd {mem_loc}, xmm0")
            elif field_type == 'char' or field_type == 'int8':
                self.output.append(f"    mov byte ptr {mem_loc}, al")
            elif field_type == 'int16':
                self.output.append(f"    mov word ptr {mem_loc}, ax")
            elif field_type in self.structs:
                # Struct copy (non-array field)
                sz = self.get_struct_size(field_type)
                self.output.append("    mov rsi, rax")
                self.output.append(f"    lea rdi, {mem_loc}")
                self.output.append(f"    mov rcx, {sz}")
                self.output.append("    rep movsb")

    def emit_global_constant(self, type_name, expr):
        """Emits a constant value for an initializer in the .data section."""
        # Strip potential pointer suffix for size check but keep for logic
        base_size = self.get_type_size(type_name)
        
        if isinstance(expr, LiteralExpr):
            val = expr.value
            if type_name in ['int8', 'char']:
                self.output.append(f"    .byte {val}")
            elif type_name == 'int16':
                self.output.append(f"    .word {val}")
            elif type_name == 'int32':
                self.output.append(f"    .long {val}")
            elif type_name == 'int64' or type_name.endswith('*'):
                self.output.append(f"    .quad {val}")
            elif type_name == 'float32':
                self.output.append(f"    .float {val}")
            elif type_name == 'float64':
                self.output.append(f"    .double {val}")
            elif type_name == 'string':
                if val not in self.string_literals:
                    str_label = self.new_label("str_glob")
                    self.string_literals[val] = str_label
                else:
                    str_label = self.string_literals[val]
                self.output.append(f"    .quad {str_label}")
            else:
                self.output.append(f"    .quad {val}")
        elif isinstance(expr, EnumValueExpr):
            if expr.enum_name in self.enums:
                val = self.enums[expr.enum_name].get(expr.value_name, 0)
                if base_size == 1: self.output.append(f"    .byte {val}")
                elif base_size == 2: self.output.append(f"    .word {val}")
                elif base_size == 4: self.output.append(f"    .long {val}")
                else: self.output.append(f"    .quad {val}")
            else:
                self.output.append(f"    .zero {base_size}")
        elif type_name in self.structs and isinstance(expr, StructLiteralExpr):
            # Recurse for struct fields
            struct_fields = self.structs[type_name]
            # Map field name to (value, type) from literal
            lit_fields = {fn: (fv, ft) for fn, ft, fv in expr.field_values}
            
            for f_type, f_name in struct_fields:
                if f_name in lit_fields:
                    fv, ft = lit_fields[f_name]
                    self.emit_global_constant(f_type, fv)
                else:
                    # Pad with zero
                    self.output.append(f"    .zero {self.get_type_size(f_type)}")
        elif type_name.endswith(']') and isinstance(expr, ArrayLiteralExpr):
            open_bracket = type_name.rfind('[')
            inner_type = type_name[:open_bracket]
            size_str = type_name[open_bracket+1:-1]
            try:
                size = int(size_str)
            except ValueError:
                size = len(expr.values)
            
            for i in range(size):
                if i < len(expr.values):
                    self.emit_global_constant(inner_type, expr.values[i])
                else:
                    self.output.append(f"    .zero {self.get_type_size(inner_type)}")
        else:
            # Fallback to zero
            self.output.append(f"    .zero {base_size}")


    def gen_function(self, func):
        self.current_function = func
        self.locals = {}
        # Count non-volatile registers to save
        pushed_count = 1 # rbx is always saved
        if not self.is_linux:
            pushed_count += 2 # rsi, rdi on Windows
        
        self.next_local_offset = pushed_count * 8
        self.max_local_offset = self.next_local_offset
        self.current_func_end_label = self.new_label(f"E_{func.name}")
        self.current_func_return_type = func.return_type
        
        if func.is_exported:
            self.output.append(f".globl {func.name}")
        
        self.output.append(f"{func.name}:")
        self.output.append("    push rbp")
        self.output.append("    mov rbp, rsp")
        
        # Save non-volatile registers
        if not self.is_linux:
            self.output.append("    push rsi")
            self.output.append("    push rdi")
        self.output.append("    push rbx")
        
        # Placeholder for stack allocation
        sub_rsp_idx = len(self.output)
        self.output.append("    sub rsp, STACK_SIZE")

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

        # Variadic Setup
        if func.name in self.variadic_info:
            vname, vstart = self.variadic_info[func.name]
            if self.is_windows:
                 # Spill registers to shadow space for contiguous access
                 self.output.append("    mov [rbp + 16], rcx")
                 self.output.append("    mov [rbp + 24], rdx")
                 self.output.append("    mov [rbp + 32], r8")
                 self.output.append("    mov [rbp + 40], r9")
                 
                 # Save count (rax) and types array (r10)
                 amt_off = self.alloc_temp("int32")
                 self.output.append(f"    mov [rbp - {amt_off}], rax")
                 self.locals[f"__amount_{vname}"] = (amt_off, "int32")
                 
                 types_off = self.alloc_temp("string*")
                 self.output.append(f"    mov [rbp - {types_off}], r10")
                 self.locals[f"__types_{vname}"] = (types_off, "string*")
            else:
                 # Linux: Spill registers to stack for contiguous access
                 # System V passes args in rdi, rsi, rdx, rcx, r8, r9 for integers
                 # and xmm0-xmm7 for floats
                 # Count is passed in R11 (to avoid conflict with AL for varargs)
                 # Types array is in R10
                 
                 # First, allocate and save count (r11) and types array (r10)
                 amt_off = self.alloc_temp("int32")
                 self.output.append(f"    mov [rbp - {amt_off}], r11")
                 self.locals[f"__amount_{vname}"] = (amt_off, "int32")
                 
                 types_off = self.alloc_temp("string*")
                 self.output.append(f"    mov [rbp - {types_off}], r10")
                 self.locals[f"__types_{vname}"] = (types_off, "string*")
                 
                 # Now allocate spill area for registers:
                 # 6 integer regs (48 bytes) + 8 XMM regs (64 bytes) = 112 bytes
                 self.next_local_offset += 112
                 self.max_local_offset = max(self.max_local_offset, self.next_local_offset)
                 spill_base = self.next_local_offset
                 
                 # Spill the integer registers (at spill_base - 0 to spill_base - 48)
                 self.output.append(f"    mov [rbp - {spill_base - 0}], rdi")
                 self.output.append(f"    mov [rbp - {spill_base - 8}], rsi")
                 self.output.append(f"    mov [rbp - {spill_base - 16}], rdx")
                 self.output.append(f"    mov [rbp - {spill_base - 24}], rcx")
                 self.output.append(f"    mov [rbp - {spill_base - 32}], r8")
                 self.output.append(f"    mov [rbp - {spill_base - 40}], r9")
                 
                 # Spill the XMM registers (at spill_base - 48 to spill_base - 112)
                 # We use movq to store the lower 64 bits (enough for float64)
                 self.output.append(f"    movq [rbp - {spill_base - 48}], xmm0")
                 self.output.append(f"    movq [rbp - {spill_base - 56}], xmm1")
                 self.output.append(f"    movq [rbp - {spill_base - 64}], xmm2")
                 self.output.append(f"    movq [rbp - {spill_base - 72}], xmm3")
                 self.output.append(f"    movq [rbp - {spill_base - 80}], xmm4")
                 self.output.append(f"    movq [rbp - {spill_base - 88}], xmm5")
                 self.output.append(f"    movq [rbp - {spill_base - 96}], xmm6")
                 self.output.append(f"    movq [rbp - {spill_base - 104}], xmm7")
                 
                 # Store spill base for ArgsAccessExpr
                 self.locals[f"__linux_spill_base_{vname}"] = (spill_base, "int64")
        
        for i, (pname, ptype) in enumerate(func.params):
            self.next_local_offset += 8
            self.max_local_offset = max(self.max_local_offset, self.next_local_offset)
            # If param is array or struct, treat as pointer for code generation
            local_type = ptype
            if ptype.endswith('[]'):
                 local_type = ptype[:-2] + "*"
            elif ptype in self.structs:
                 local_type = ptype + "*"
            self.locals[pname] = (self.next_local_offset, local_type)
            
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
        
        # Calculate final stack size
        # We need (max_local_offset + 8 [pushed rbp] + 8 [ret addr] + padding) % 16 == 0
        # This simplifies to: (max_local_offset + padding) % 16 == 0
        aligned_total_depth = (self.max_local_offset + 15) // 16 * 16
        stack_size = aligned_total_depth - (pushed_count * 8)
        
        self.output[sub_rsp_idx] = f"    sub rsp, {stack_size}"
        
        if stack_size > 0:
            self.output.append(f"    add rsp, {stack_size}")
            
        # Restore non-volatile registers
        self.output.append("    pop rbx")
        if not self.is_linux:
            self.output.append("    pop rdi")
            self.output.append("    pop rsi")
            
        self.output.append("    pop rbp")
        self.output.append("    ret")

    def alloc_temp(self, type_name):
        """Allocate a temporary local variable."""
        size = self.get_type_size(type_name)
        # Align to 8 bytes for simplicity on stack
        aligned_size = ((size + 7) // 8) * 8
        self.next_local_offset += aligned_size
        self.max_local_offset = max(self.max_local_offset, self.next_local_offset)
        return self.next_local_offset

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
            if stmt.type_name in self.structs and not stmt.is_array:
                struct_size = self.get_struct_size(stmt.type_name)
                # Align to 8 bytes
                aligned_size = ((struct_size + 7) // 8) * 8
                self.next_local_offset += aligned_size
                self.max_local_offset = max(self.max_local_offset, self.next_local_offset)
                self.locals[stmt.name] = (self.next_local_offset, stmt.type_name)
                
                # Generate struct literal initialization
                if isinstance(stmt.expr, StructLiteralExpr):
                    self.gen_struct_init(stmt.name, stmt.type_name, stmt.expr)
            else:
                if stmt.is_array:
                     # Array Declaration
                     element_type = stmt.type_name
                     element_size = self.get_type_size(element_type)
                     # Resolve array size (could be int or variable name)
                     resolved_array_size = self.resolve_array_size(stmt.array_size)
                     
                     # Check if it's a runtime resolution (tuple) or compile-time (int)
                     is_runtime_size = isinstance(resolved_array_size, tuple)
                     if is_runtime_size:
                         # Runtime VLA - we need to allocate on stack dynamically
                         _, size_var_name = resolved_array_size
                         
                         # Load the size variable value into rcx
                         if size_var_name in self.locals:
                             offset, var_type = self.locals[size_var_name]
                             self.output.append(f"    mov rcx, [rbp - {offset}]")
                         elif size_var_name in self.globals:
                             global_info = self.globals[size_var_name]
                             self.output.append(f"    mov rcx, [rel {global_info['label']}]")
                         else:
                             raise RuntimeError(f"Unknown variable '{size_var_name}' used as array size")
                         
                         # Calculate total size: size * element_size + 8 (for length header)
                         # rcx = array_size, we need rcx = array_size * element_size + 8
                         self.output.append(f"    imul rcx, {element_size}")
                         self.output.append(f"    add rcx, 8")
                         # Align to 8 bytes
                         self.output.append(f"    add rcx, 7")
                         self.output.append(f"    and rcx, -8")  # Align down to 8 bytes
                         
                         # Allocate on stack
                         self.output.append(f"    sub rsp, rcx")
                         self.output.append(f"    mov rax, rsp")
                         self.output.append(f"    add rax, 8")  # Point to data start
                         
                         # Store the array pointer and update tracking
                         # We need to track this VLA differently
                         self.next_local_offset += 8  # Just for the pointer
                         self.max_local_offset = max(self.max_local_offset, self.next_local_offset)
                         
                         # Store the pointer to the VLA
                         ptr_offset = self.next_local_offset
                         self.output.append(f"    mov [rbp - {ptr_offset}], rax")
                         
                         # Store the length in the length header
                         if size_var_name in self.locals:
                             offset, var_type = self.locals[size_var_name]
                             self.output.append(f"    mov rax, [rbp - {offset}]")
                         elif size_var_name in self.globals:
                             global_info = self.globals[size_var_name]
                             self.output.append(f"    mov rax, [rel {global_info['label']}]")
                         self.output.append(f"    mov [rsp], rax")  # Store length at rsp (start of block)
                         
                         # Store as VLA type (with '[]' to indicate dynamic)
                         self.locals[stmt.name] = (ptr_offset, element_type + '[]')
                         
                         # For VLAs, skip array literal initialization for now
                         # (would need loop-based initialization)
                         if isinstance(stmt.expr, ArrayLiteralExpr):
                             # TODO: Implement VLA initialization from array literal
                             pass
                         elif isinstance(stmt.expr, LiteralExpr) and stmt.expr.value == 0:
                             # Zero init - could call memset
                             pass
                     else:
                         # Compile-time constant size
                         # Add 8 bytes for length header
                         total_size = resolved_array_size * element_size + 8
                         # Align to 8 bytes
                         aligned_size = ((total_size + 7) // 8) * 8
                         self.next_local_offset += aligned_size
                         self.max_local_offset = max(self.max_local_offset, self.next_local_offset)
                         
                         # Calculate offsets
                         # Block start: [rbp - self.next_local_offset]
                         # Length header: [rbp - self.next_local_offset]
                         # Data start: [rbp - self.next_local_offset + 8]
                         
                         # Store length
                         length_loc = f"[rbp - {self.next_local_offset}]"
                         self.output.append(f"    mov qword ptr {length_loc}, {resolved_array_size}")
                         
                         # Mark as array type in locals
                         # Point locals to DATA start.
                         # self.next_local_offset points to bottom of block. 
                         # data_offset = self.next_local_offset - 8
                         var_offset = self.next_local_offset - 8
                         # Store full type with size for fixed-size arrays (e.g., int32[10])
                         self.locals[stmt.name] = (var_offset, element_type + f'[{resolved_array_size}]')
                         
                         # Initialization
                         if isinstance(stmt.expr, ArrayLiteralExpr):
                             for i, val_expr in enumerate(stmt.expr.values):
                                 if i >= resolved_array_size: break
                                 
                                 offset = i * element_size
                                 # Values are generated; expected type matches element
                                 self.gen_expression(val_expr, expected_type=element_type)
                                 
                                 # Store at [rbp - var_offset + offset]
                                 # var_offset is data start (distance from rbp). 
                                 # [rbp - var_offset] is data[0].
                                 # [rbp - (var_offset - offset)] -> [rbp - var_offset + offset]
                                 
                                 mem_loc = f"[rbp - {var_offset - offset}]"
                                 
                                 if element_type == 'float32':
                                     self.output.append(f"    movss {mem_loc}, xmm0")
                                 elif element_type == 'float64':
                                     self.output.append(f"    movsd {mem_loc}, xmm0")
                                 elif element_type == 'int32':
                                     self.output.append(f"    mov dword ptr {mem_loc}, eax")
                                 elif element_type == 'int64' or element_type.endswith('*') or element_type == 'string':
                                     self.output.append(f"    mov qword ptr {mem_loc}, rax")
                                 elif element_type == 'char' or element_type == 'int8':
                                     self.output.append(f"    mov byte ptr {mem_loc}, al")
                                 elif element_type == 'int16':
                                     self.output.append(f"    mov word ptr {mem_loc}, ax")
                                 elif element_type in self.enums:
                                     self.output.append(f"    mov qword ptr {mem_loc}, rax")
                                 elif element_type in self.structs:
                                     # Struct by value in array initialization
                                     sz = self.get_struct_size(element_type)
                                     # ...
                                     self.gen_expression(val_expr, expected_type=element_type)
                                     
                                     self.output.append("    mov rsi, rax")
                                     self.output.append(f"    lea rdi, {mem_loc}")
                                     self.output.append(f"    mov rcx, {sz}")
                                     self.output.append("    rep movsb")
                                 else:
                                     pass
                             
                             # TODO: Zero fill if values < size?
                             if len(stmt.expr.values) < resolved_array_size and len(stmt.expr.values) > 0:
                                 # Basic zero fill support
                                 pass
                         elif isinstance(stmt.expr, LiteralExpr) and stmt.expr.value == 0:
                             # Zero init whole array?
                             pass
                         else:
                            # Generic expression for array init (e.g. CallExpr returning an array pointer)
                            actual_type = self.gen_expression(stmt.expr, expected_type=element_type + '[]')
                            
                            # Assume RAX points to data start of returned array.
                            # Copy data to our local buffer.
                            sz = resolved_array_size * element_size
                            self.output.append("    mov rsi, rax")
                            self.output.append(f"    lea rdi, [rbp - {var_offset}]")
                            self.output.append(f"    mov rcx, {sz}")
                            self.output.append("    rep movsb")

                else:
                    actual_type = self.gen_expression(stmt.expr, expected_type=None)
                    self.next_local_offset += 8
                    self.max_local_offset = max(self.max_local_offset, self.next_local_offset)
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
            # Handle dotted variable names (e.g., math.x for SonOf variables or p.x for struct fields)
            if '.' in stmt.name:
                parts = stmt.name.split('.', 1)
                parent_name = parts[0]
                child_name = parts[1]
                
                # Check if parent is a local struct variable (struct field assignment)
                if parent_name in self.locals:
                    var_offset, struct_type = self.locals[parent_name]
                    
                    if struct_type in self.structs:
                        # This is a struct field assignment
                        field_offsets = self.struct_field_offsets.get(struct_type, {})
                        field_offset = field_offsets.get(child_name, 0)
                        field_type = self.get_field_type(struct_type, child_name)
                        
                        # Generate the expression value
                        self.gen_expression(stmt.expr, expected_type=field_type)
                        
                        # Store the value in the field
                        if field_type == 'int32':
                            self.output.append(f"    mov dword ptr [rbp - {var_offset - field_offset}], eax")
                        elif field_type == 'int64' or field_type == 'string' or field_type in self.enums or field_type.endswith('*'):
                            self.output.append(f"    mov qword ptr [rbp - {var_offset - field_offset}], rax")
                        elif field_type == 'float32':
                            self.output.append(f"    movss [rbp - {var_offset - field_offset}], xmm0")
                        elif field_type == 'float64':
                            self.output.append(f"    movsd [rbp - {var_offset - field_offset}], xmm0")
                        elif field_type == 'char' or field_type == 'int8':
                             self.output.append(f"    mov byte ptr [rbp - {var_offset - field_offset}], al")
                        elif field_type == 'int16':
                             self.output.append(f"    mov word ptr [rbp - {var_offset - field_offset}], ax")
                        return
                
                # Otherwise, treat as SonOf variable (use the child name)
                var_name = child_name
            else:
                var_name = stmt.name
            
            # Pointer mutation via: md int32* ptr = value;  =>  *ptr = value
            if isinstance(stmt.type_name, str) and stmt.type_name.endswith('*'):
                base_type = stmt.type_name[:-1]
                if var_name in self.locals:
                    ptr_offset, ptr_type = self.locals[var_name]
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

            if var_name in self.locals:
                offset, t = self.locals[var_name]
                if t in ['float32', 'float64']:
                    self.gen_conversion(actual_type, t)
                    if t == 'float32':
                        self.output.append(f"    movss [rbp - {offset}], xmm0")
                    else:
                        self.output.append(f"    movsd [rbp - {offset}], xmm0")
                else:
                    self.gen_conversion(actual_type, t)
                    self.output.append(f"    mov [rbp - {offset}], rax")
            elif var_name in self.globals:
                info = self.globals[var_name]
                label = info['label']
                t = info['type']
                
                if t in ['float32', 'float64']:
                    self.gen_conversion(actual_type, t)
                    if t == 'float32':
                        self.output.append(f"    movss [rip + {label}], xmm0")
                    else:
                        self.output.append(f"    movsd [rip + {label}], xmm0")
                else:
                    self.gen_conversion(actual_type, t)
                    size = self.get_type_size(t)
                    if size == 1:
                        self.output.append(f"    mov byte ptr [rip + {label}], al")
                    elif size == 2:
                        self.output.append(f"    mov word ptr [rip + {label}], ax")
                    elif size == 4:
                        self.output.append(f"    mov dword ptr [rip + {label}], eax")
                    else:
                        self.output.append(f"    mov qword ptr [rip + {label}], rax")
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
                    elif field_type == 'int64' or field_type == 'string' or field_type in self.enums or field_type.endswith('*'):
                        self.output.append(f"    mov qword ptr [rbp - {var_offset - field_offset}], rax")
                    elif field_type == 'float32':
                        self.output.append(f"    movss [rbp - {var_offset - field_offset}], xmm0")
                    elif field_type == 'float64':
                        self.output.append(f"    movsd [rbp - {var_offset - field_offset}], xmm0")
                    elif field_type == 'char' or field_type == 'int8':
                         self.output.append(f"    mov byte ptr [rbp - {var_offset - field_offset}], al")
                    elif field_type == 'int16':
                         self.output.append(f"    mov word ptr [rbp - {var_offset - field_offset}], ax")
                    elif field_type in self.structs:
                        sz = self.get_struct_size(field_type)
                        self.output.append("    mov rsi, rax")
                        self.output.append(f"    lea rdi, [rbp - {var_offset - field_offset}]")
                        self.output.append(f"    mov rcx, {sz}")
                        self.output.append("    rep movsb")
            elif stmt.var_name in self.globals:
                info = self.globals[stmt.var_name]
                label = info['label']
                struct_type = info['type']
                
                if struct_type in self.structs:
                    field_offsets = self.struct_field_offsets.get(struct_type, {})
                    field_offset = field_offsets.get(stmt.field_name, 0)
                    field_type = self.get_field_type(struct_type, stmt.field_name)
                    
                    self.gen_expression(stmt.expr, expected_type=field_type)
                    
                    # Store logic
                    if field_type == 'int32':
                        self.output.append(f"    mov dword ptr [rip + {label} + {field_offset}], eax")
                    elif field_type == 'int64' or field_type.endswith('*'):
                        self.output.append(f"    mov qword ptr [rip + {label} + {field_offset}], rax")
                    elif field_type == 'float32':
                        self.output.append(f"    movss [rip + {label} + {field_offset}], xmm0")
                    elif field_type == 'float64':
                        self.output.append(f"    movsd [rip + {label} + {field_offset}], xmm0")
                    elif field_type == 'char' or field_type == 'int8':
                        self.output.append(f"    mov byte ptr [rip + {label} + {field_offset}], al")
                    elif field_type == 'int16':
                        self.output.append(f"    mov word ptr [rip + {label} + {field_offset}], ax")
        elif isinstance(stmt, ArrayAssignStmt):
             # 1. Calc address of element
             if stmt.arr_name in self.locals:
                  offset, t = self.locals[stmt.arr_name]
                  if t.endswith('[]'):
                       # Dynamic array (no size): load the pointer value
                       self.output.append(f"    mov rax, [rbp - {offset}]")
                       base_type = t[:-2]
                  elif '[' in t:
                       # Fixed-size array: get address of inline data
                       self.output.append(f"    lea rax, [rbp - {offset}]")
                       base_type = t[:t.rfind('[')]
                  else:
                       # Pointer
                       self.output.append(f"    mov rax, [rbp - {offset}]")
                       base_type = t[:-1] if t.endswith('*') else 'int64'
                  
                  self.output.append("    push rax")
             elif stmt.arr_name in self.globals:
                  info = self.globals[stmt.arr_name]
                  label = info['label']
                  t = info['type']
                  
                  self.output.append(f"    lea rax, [rip + {label}]")
                  base_type = t[:t.rfind('[')] if t.endswith(']') else t
                  self.output.append("    push rax")
             else:
                  # Fallback/Error?
                  self.output.append("    xor rax, rax")
                  self.output.append("    push rax")
                  base_type = 'int64'
             
             # Index
             self.gen_expression(stmt.index, expected_type='int64')
             self.output.append("    mov rcx, rax")
             self.output.append("    pop rax")
             
             elm_size = self.get_type_size(base_type)
             self.output.append(f"    imul rcx, {elm_size}")
             self.output.append("    add rax, rcx")
             self.output.append("    push rax") # Save address
             
             # Value
             self.gen_expression(stmt.expr, expected_type=base_type)
             self.output.append("    pop rdx") # Address in rdx
             
             # Store
             if base_type == 'float32':
                 self.output.append("    movss [rdx], xmm0")
             elif base_type == 'float64':
                 self.output.append("    movsd [rdx], xmm0")
             elif base_type == 'int32':
                 self.output.append("    mov dword ptr [rdx], eax")
             elif base_type == 'int64' or base_type == 'string' or base_type in self.enums or base_type.endswith('*'):
                 self.output.append("    mov qword ptr [rdx], rax")
             elif base_type == 'char' or base_type == 'int8':
                 self.output.append("    mov byte ptr [rdx], al")
             elif base_type == 'int16':
                 self.output.append("    mov word ptr [rdx], ax")
             elif base_type in self.structs:
                 # Struct copy to array element
                 sz = self.get_struct_size(base_type)
                 self.output.append("    mov rsi, rax")
                 self.output.append("    mov rdi, rdx")
                 self.output.append(f"    mov rcx, {sz}")
                 self.output.append("    rep movsb")
             else:
                 self.output.append("    mov qword ptr [rdx], rax")

        elif isinstance(stmt, PushStmt):
            actual_type = self.gen_expression(stmt.expr, expected_type=stmt.type_name)
            
            # If explicit type provided, ensure we convert to it before pushing
            if actual_type != stmt.type_name:
                self.gen_conversion(actual_type, stmt.type_name)
                
            if stmt.type_name == 'float32':
                self.output.append("    sub rsp, 8")
                self.output.append("    movss [rsp], xmm0")
            elif stmt.type_name == 'float64':
                self.output.append("    sub rsp, 8")
                self.output.append("    movsd [rsp], xmm0")
            else:
                # Integer types (and pointers)
                # Ensure we represent smaller ints correctly in RAX if needed?
                # RAX should hold the value. 'push' pushes 64 bits.
                self.output.append("    push rax")

        elif isinstance(stmt, PopStmt):
             # Check if var exists
             if stmt.var_name in self.locals:
                 offset, var_type = self.locals[stmt.var_name]
                 
                 # Pop into register based on stmt.type_name (how we interpret the stack slot)
                 if stmt.type_name == 'float32':
                     self.output.append("    movss xmm0, [rsp]")
                     self.output.append("    add rsp, 8")
                     # Convert to var_type if needed? PopStmt implies simple pop, but let's be safe
                     if var_type != 'float32':
                         self.gen_conversion('float32', var_type)
                 elif stmt.type_name == 'float64':
                     self.output.append("    movsd xmm0, [rsp]")
                     self.output.append("    add rsp, 8")
                     if var_type != 'float64':
                         self.gen_conversion('float64', var_type)
                 else:
                     self.output.append("    pop rax")
                     if var_type not in ['float32', 'float64'] and stmt.type_name != var_type:
                         pass # conversions for ints usually implicit or done via store size
                 
                 # Store to variable
                 if var_type == 'float32':
                     self.output.append(f"    movss [rbp - {offset}], xmm0")
                 elif var_type == 'float64':
                     self.output.append(f"    movsd [rbp - {offset}], xmm0")
                 elif var_type == 'int32':
                     self.output.append(f"    mov dword ptr [rbp - {offset}], eax")
                 elif var_type == 'int64':
                     self.output.append(f"    mov [rbp - {offset}], rax")
                 elif var_type == 'char':
                     self.output.append(f"    mov byte ptr [rbp - {offset}], al")
                 elif var_type == 'int8':
                     self.output.append(f"    mov byte ptr [rbp - {offset}], al")
                 elif var_type == 'int16':
                     self.output.append(f"    mov word ptr [rbp - {offset}], ax")
                 elif var_type in self.structs:
                     pass # popping struct? Not supported yet simple copy
                 else:
                     self.output.append(f"    mov [rbp - {offset}], rax")

        elif isinstance(stmt, SwapStmt):
            # Swap top two elements on stack.
            # Since our stack implementation uses 8-byte slots for EVERYTHING (ints, floats, pointers),
            # we can just swap the two 8-byte words at the top of the stack using registers.
            self.output.append("    pop rax") # Top element (A)
            self.output.append("    pop rdx") # Second element (B)
            self.output.append("    push rax") # Push A (now deep)
            self.output.append("    push rdx") # Push B (now top)
            
        elif isinstance(stmt, DupStmt):
            # Duplicate top element.
            # Since our stack implementation uses 8-byte slots, we just copy the top 8 bytes.
            self.output.append("    mov rax, [rsp]")
            self.output.append("    push rax")
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
                # Native int - optimized loading
                val = expr.value
                
                # Optimization: Use XOR for zero (faster, smaller encoding)
                if val == 0:
                    self.output.append("    xor eax, eax")  # Implicitly zeros RAX
                # Optimization: Small positive values fit in 32-bit mov (smaller encoding)
                elif 0 < val <= 0x7FFFFFFF:
                    self.output.append(f"    mov eax, {val}")  # Zero-extends to RAX
                # Optimization: -1 can use XOR + NOT or just mov
                elif val == -1:
                    self.output.append("    mov rax, -1")  # Or: xor eax, eax; not rax
                else:
                    # Full 64-bit immediate
                    val_hex = hex(val)
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
                else:
                    label = self.string_literals[expr.value]
                self.output.append(f"    lea rax, [rip + {label}]")
                return 'string'

        # 4. Handle VarRef
        elif isinstance(expr, VarRefExpr):
            # Handle dotted variable names (e.g., math.x for SonOf variables)
            var_name = expr.name
            if '.' in var_name:
                var_name = var_name.split('.', 1)[1]
            
            if var_name in self.locals:
                offset, t = self.locals[var_name]
                native_type = t
                
                # Load variable
                if t == 'float32':
                    self.output.append(f"    movss xmm0, [rbp - {offset}]")
                elif t == 'float64':
                    self.output.append(f"    movsd xmm0, [rbp - {offset}]")
                elif t.endswith('[]'):
                    # Dynamic array (no size): load the pointer value
                    self.output.append(f"    mov rax, [rbp - {offset}]")
                    # Return pointer type
                    t = t[:-2] + "*"
                elif '[' in t:
                    # Fixed-size array (e.g., int32[10]): get address of inline data
                    self.output.append(f"    lea rax, [rbp - {offset}]")
                    # Return pointer type
                    t = t[:t.rfind('[')] + "*"
                elif t in self.structs:
                    # Struct decay to pointer
                    self.output.append(f"    lea rax, [rbp - {offset}]")
                    t = t + "*"
                else:
                    self.output.append(f"    mov rax, [rbp - {offset}]")
                
                # Convert if needed
                if expected_type is not None and expected_type != t:
                    self.gen_conversion(t, expected_type)
                    return expected_type
                return t
            elif var_name in self.data_labels:
                label = self.data_labels[var_name]
                self.output.append(f"    lea rax, [rip + {label}]")
                return 'string'
            elif var_name in self.globals:
                info = self.globals[var_name]
                label = info['label']
                t = info['type']
                
                if t == 'float32':
                    self.output.append(f"    movss xmm0, [rip + {label}]")
                elif t == 'float64':
                    self.output.append(f"    movsd xmm0, [rip + {label}]")
                elif t.endswith(']') or t in self.structs:
                    self.output.append(f"    lea rax, [rip + {label}]")
                    # Return pointer type
                    if t.endswith(']'):
                        t = t[:t.rfind('[')] + "*"
                    else:
                        t = t + "*"
                else:
                    size = self.get_type_size(t)
                    if size == 1:
                        self.output.append(f"    movsx rax, byte ptr [rip + {label}]")
                    elif size == 2:
                        self.output.append(f"    movsx rax, word ptr [rip + {label}]")
                    elif size == 4:
                        self.output.append(f"    movsxd rax, dword ptr [rip + {label}]")
                    else:
                        self.output.append(f"    mov rax, [rip + {label}]")
                
                if expected_type is not None and expected_type != t:
                    self.gen_conversion(t, expected_type)
                    return expected_type
                return t

        # 4.5. Handle FieldAccessExpr (e.g., p.x)
        elif isinstance(expr, FieldAccessExpr):
            # Check for variadic amount
            if expr.field_name == "amount" and isinstance(expr.obj, VarRefExpr):
                pname = expr.obj.name
                if f"__amount_{pname}" in self.locals:
                    off, _ = self.locals[f"__amount_{pname}"]
                    self.output.append(f"    mov rax, [rbp - {off}]")
                    return "int32"
            
            # Check if this is a SonOf variable access (parent.child)
            if isinstance(expr.obj, VarRefExpr):
                parent_name = expr.obj.name
                child_name = expr.field_name
                
                # Check if the child is a global variable (SonOf)
                if child_name in self.globals:
                    info = self.globals[child_name]
                    label = info['label']
                    t = info['type']
                    
                    if t == 'float32':
                        self.output.append(f"    movss xmm0, [rip + {label}]")
                    elif t == 'float64':
                        self.output.append(f"    movsd xmm0, [rip + {label}]")
                    else:
                        size = self.get_type_size(t)
                        if size == 1:
                            self.output.append(f"    movsx rax, byte ptr [rip + {label}]")
                        elif size == 2:
                            self.output.append(f"    movsx rax, word ptr [rip + {label}]")
                        elif size == 4:
                            self.output.append(f"    movsxd rax, dword ptr [rip + {label}]")
                        else:
                            self.output.append(f"    mov rax, [rip + {label}]")
                    
                    if expected_type is not None and expected_type != t:
                        self.gen_conversion(t, expected_type)
                        return expected_type
                    return t
            
            # 1. Get base address and struct type
            if isinstance(expr.obj, VarRefExpr) and expr.obj.name in self.locals:
                var_offset, struct_type = self.locals[expr.obj.name]
                # If it's a pointer to struct, load it
                if struct_type.endswith('*'):
                    self.output.append(f"    mov rax, [rbp - {var_offset}]")
                    struct_type = struct_type[:-1]
                else:
                    self.output.append(f"    lea rax, [rbp - {var_offset}]")
            else:
                struct_type = self.gen_expression(expr.obj)
                # For non-VarRefExpr, we assume gen_expression left the address in RAX
            
            if isinstance(struct_type, str) and struct_type.endswith('*'):
                struct_type = struct_type[:-1]

            if struct_type in self.structs:
                field_offsets = self.struct_field_offsets.get(struct_type, {})
                field_offset = field_offsets.get(expr.field_name, 0)
                field_type = self.get_field_type(struct_type, expr.field_name)
                
                # Base address in RAX. Add field offset.
                if field_offset != 0:
                    self.output.append(f"    add rax, {field_offset}")
                
                # Load the field value
                if field_type == 'int32':
                    self.output.append("    movsxd rax, dword ptr [rax]")
                elif field_type == 'int64' or field_type.endswith('*') or field_type == 'string':
                    self.output.append("    mov rax, [rax]")
                elif field_type == 'float32':
                    self.output.append("    movss xmm0, [rax]")
                elif field_type == 'float64':
                    self.output.append("    movsd xmm0, [rax]")
                elif field_type == 'char':
                    self.output.append("    movzx rax, byte ptr [rax]")
                elif field_type == 'int8':
                    self.output.append("    movsx rax, byte ptr [rax]")
                elif field_type == 'int16':
                    self.output.append("    movsx rax, word ptr [rax]")
                elif field_type in self.structs or '[' in field_type:
                    # Nested struct or Array? return address which is already in rax
                    pass
                else:
                    # Enums etc
                    self.output.append("    mov rax, [rax]")
                
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
        
        elif isinstance(expr, ArrayAccessExpr):
             # 1. Base Address (decayed pointer)
             if isinstance(expr.arr, VarRefExpr) and expr.arr.name in self.locals:
                  offset, t = self.locals[expr.arr.name]
                  # Check if it's a dynamic array (int32[]) vs fixed-size (int32[10])
                  # Dynamic arrays have '[]' suffix (no size), fixed-size arrays have '[N]'
                  if t.endswith('[]'):
                       # Dynamic array: load the pointer value
                       self.output.append(f"    mov rax, [rbp - {offset}]")
                       arr_type = t
                  elif '[' in t:
                       # Fixed-size array: get address of inline data
                       self.output.append(f"    lea rax, [rbp - {offset}]")
                       arr_type = t
                  else:
                       # Should be pointer
                       arr_type = self.gen_expression(expr.arr)
             else:
                  arr_type = self.gen_expression(expr.arr)
             
             if expr.index is None:
                  # Equivalent to array decay (pointer)
                  if arr_type.endswith('[]'):
                       return arr_type[:-2] + "*"
                  return arr_type

             self.output.append("    push rax")
             
             # Index
             self.gen_expression(expr.index, expected_type='int64')
             self.output.append("    mov rcx, rax")
             self.output.append("    pop rax")
             
             # Calculate offset
             if '[' in arr_type:
                  bracket_pos = arr_type.find('[')
                  element_type = arr_type[:bracket_pos]
             else:
                  element_type = arr_type.replace('[]', '').replace('*', '') if arr_type else 'int64'
             
             elm_size = self.get_type_size(element_type)
             
             self.output.append(f"    imul rcx, {elm_size}")
             self.output.append("    add rax, rcx")
             
             # Load value
             if element_type == 'float32':
                 self.output.append("    movss xmm0, [rax]")
             elif element_type == 'float64':
                 self.output.append("    movsd xmm0, [rax]")
             elif element_type == 'int32':
                 self.output.append("    movsxd rax, dword ptr [rax]")
             elif element_type == 'char':
                 self.output.append("    movzx rax, byte ptr [rax]")
             elif element_type == 'int8':
                 self.output.append("    movsx rax, byte ptr [rax]")
             elif element_type == 'int16':
                 self.output.append("    movsx rax, word ptr [rax]")
             elif element_type in self.structs:
                 # Struct access: return pointer to the element
                 # (In bcb, struct variables often act as pointers to their storage)
                 pass
             else:
                 self.output.append("    mov rax, [rax]")
             
             return element_type

        elif isinstance(expr, StructLiteralExpr):
             struct_name = expected_type
             if not struct_name or struct_name not in self.structs:
                 # If no expected type, try the one from the expr itself if it has it
                 if hasattr(expr, 'struct_name'):
                     struct_name = expr.struct_name
                 else:
                     return 'unknown'
            
             sz = self.get_struct_size(struct_name)
             temp_offset = self.alloc_temp(struct_name)
             
             field_offsets = self.struct_field_offsets.get(struct_name, {})
             real_field_types = {fn: ft for ft, fn in self.structs.get(struct_name, [])}
             
             for field_name, field_type_unused, field_expr in expr.field_values:
                  if field_name not in field_offsets: continue
                  offset = field_offsets[field_name]
                  field_type = real_field_types.get(field_name, field_type_unused)
                  
                  if '[' in field_type and not field_type.endswith('[]') and not isinstance(field_expr, ArrayLiteralExpr):
                       # Expression assigned to fixed-size array field
                       sz = self.get_type_size(field_type)
                       self.gen_expression(field_expr, expected_type=field_type) # Should return pointer to array data
                       self.output.append("    mov rsi, rax")
                       self.output.append(f"    lea rdi, [rbp - {temp_offset - offset}]")
                       self.output.append(f"    mov rcx, {sz}")
                       self.output.append("    rep movsb")
                       continue
                  
                  self.gen_expression(field_expr, expected_type=field_type)
                  
                  mem_loc = f"[rbp - {temp_offset - offset}]"
                  
                  if field_type == 'int32':
                      self.output.append(f"    mov dword ptr {mem_loc}, eax")
                  elif field_type == 'int64' or field_type.endswith('*') or field_type == 'string':
                      self.output.append(f"    mov qword ptr {mem_loc}, rax")
                  elif field_type == 'float32':
                      self.output.append(f"    movss {mem_loc}, xmm0")
                  elif field_type == 'float64':
                      self.output.append(f"    movsd {mem_loc}, xmm0")
                  elif field_type == 'char' or field_type == 'int8':
                      self.output.append(f"    mov byte ptr {mem_loc}, al")
                  elif field_type == 'int16':
                      self.output.append(f"    mov word ptr {mem_loc}, ax")
             
             self.output.append(f"    lea rax, [rbp - {temp_offset}]")
             return struct_name

        elif isinstance(expr, ArrayLiteralExpr):
             # Determine element type
             element_type = 'int64'
             if expected_type:
                 element_type = expected_type.replace('[]', '').replace('*', '')
             elif expr.values:
                 # Try to infer from first element
                 pass # We use default int64 if unknown
             
             count = len(expr.values)
             element_size = self.get_type_size(element_type)
             total_size = count * element_size + 8
             # Allocation on stack for literal
             aligned_size = ((total_size + 7) // 8) * 8
             self.next_local_offset += aligned_size
             self.max_local_offset = max(self.max_local_offset, self.next_local_offset)
             temp_block_bottom = self.next_local_offset
             
             # Store length header
             self.output.append(f"    mov qword ptr [rbp - {temp_block_bottom}], {count}")
             
             # Data starts at [rbp - (temp_block_bottom - 8)]
             data_offset = temp_block_bottom - 8
             
             for i, val_expr in enumerate(expr.values):
                 actual_v_type = self.gen_expression(val_expr, expected_type=element_type)
                 self.gen_conversion(actual_v_type, element_type)
                 
                 mem_loc = f"[rbp - {data_offset - i * element_size}]"
                 
                 if element_type == 'float32':
                     self.output.append(f"    movss {mem_loc}, xmm0")
                 elif element_type == 'float64':
                     self.output.append(f"    movsd {mem_loc}, xmm0")
                 elif element_type == 'int32':
                     self.output.append(f"    mov dword ptr {mem_loc}, eax")
                 elif element_type == 'int64' or element_type.endswith('*') or element_type == 'string':
                     self.output.append(f"    mov qword ptr {mem_loc}, rax")
                 elif element_type == 'char' or element_type == 'int8':
                     self.output.append(f"    mov byte ptr {mem_loc}, al")
                 elif element_type == 'int16':
                     self.output.append(f"    mov word ptr {mem_loc}, ax")
                 else:
                     self.output.append(f"    mov qword ptr {mem_loc}, rax")
             
             # Return data start address in RAX
             self.output.append(f"    lea rax, [rbp - {data_offset}]")
             return element_type + '[]'
        
        elif isinstance(expr, LengthExpr):
             # For fixed-size arrays (type[size]), return the size as a constant
             # For dynamic arrays, read from the length header
             
             # First, try to get the original array type directly from the variable
             if isinstance(expr.expr, VarRefExpr) and expr.expr.name in self.locals:
                  offset, t = self.locals[expr.expr.name]
                  if '[' in t and not t.endswith('[]'):
                      # Fixed-size array: extract size from type string "type[size]"
                      open_bracket = t.rfind('[')
                      size_str = t[open_bracket+1:-1]
                      try:
                          size = int(size_str)
                          self.output.append(f"    mov rax, {size}")
                          return 'int64'
                      except ValueError:
                          pass
                  elif t.endswith('[]'):
                      # Dynamic array: load pointer and read length header
                      self.output.append(f"    mov rax, [rbp - {offset}]")
                      self.output.append("    mov rax, [rax - 8]")
                      return 'int64'
             
             # Fallback: generate expression and check type
             t = self.gen_expression(expr.expr)
             
             if isinstance(t, str) and '[' in t and not t.endswith('[]'):
                 # Fixed-size array: extract size from type string "type[size]"
                 open_bracket = t.rfind('[')
                 size_str = t[open_bracket+1:-1]
                 try:
                     size = int(size_str)
                     self.output.append(f"    mov rax, {size}")
                     return 'int64'
                 except ValueError:
                     pass
             
             # Header-based array (dynamic/pointer)
             self.output.append("    mov rax, [rax - 8]")
             return 'int64'

        elif isinstance(expr, GetTypeExpr):
             if isinstance(expr.expr, ArgsAccessExpr):
                 # Runtime type tag lookup for variadic arguments
                 self.gen_expression(expr.expr.index, expected_type='int32')
                 pname = expr.expr.name
                 if f"__types_{pname}" in self.locals:
                      off, _ = self.locals[f"__types_{pname}"]
                      self.output.append(f"    mov rcx, [rbp - {off}]")
                      self.output.append("    mov rax, [rcx + rax*8]")
                      return "string"
             
             # Inferred type name was stored by analyzer for static cases
             type_name = getattr(expr, 'inferred_type_name', 'unknown')
             
             if type_name not in self.string_literals:
                 label = self.new_label("type_str")
                 self.string_literals[type_name] = label
             else:
                 label = self.string_literals[type_name]
             
             self.output.append(f"    lea rax, [rip + {label}]")
             return "string"

        elif isinstance(expr, ArgsAccessExpr):
             # Accessing the value of a variadic argument: myargs(index)
             self.gen_expression(expr.index, expected_type='int32')
             pname = expr.name
             # Find which function we are in
             v_info = self.variadic_info.get(self.current_function.name)
             if v_info and v_info[0] == pname:
                  start_idx = v_info[1]
                  if self.is_windows:
                      # Contiguous in shadow space + stack: [rbp + 16 + (start + idx)*8]
                      self.output.append(f"    add rax, {start_idx}")
                      self.output.append("    shl rax, 3") # rax * 8
                      self.output.append("    lea rcx, [rbp + 16]")
                      self.output.append("    add rax, rcx")
                      self.output.append("    mov rax, [rax]")
                      # If expected_type is float, move bits to xmm0
                      if expected_type in ['float32', 'float64']:
                          self.output.append("    movq xmm0, rax")
                          return expected_type
                      # If expected_type is array, return pointer type
                      if expected_type and expected_type.endswith('[]'):
                          return expected_type
                      return "int64" # Treat as generic 64-bit value
                  else:
                      # Linux: args are spilled to [rbp - spill_base + offset]
                      # Integer args: rdi, rsi, rdx, rcx, r8, r9 at offsets 0-40
                      # Float args: xmm0-xmm7 at offsets 48-104
                      # We need to check the type to know which spill area to read from
                      spill_base_key = f"__linux_spill_base_{pname}"
                      if spill_base_key in self.locals:
                          spill_base, _ = self.locals[spill_base_key]
                          
                          # Check if this argument is a float by looking at the types array
                          # Types array is at __types_{pname}
                          types_key = f"__types_{pname}"
                          if types_key in self.locals:
                              types_off, _ = self.locals[types_key]
                              # Save the index in r11
                              self.output.append("    mov r11, rax")  # Save index
                              # Load type string pointer: types[idx]
                              self.output.append("    mov rax, r11")
                              self.output.append("    shl rax, 3")  # idx * 8
                              self.output.append(f"    add rax, [rbp - {types_off}]")  # rax = types + idx*8
                              self.output.append("    mov rax, [rax]")  # rax = type string pointer
                              
                              # Check if type string is exactly "float32" or "float64"
                              # First check if it starts with 'f' (for "float...")
                              # Then check if byte 7 is null (not an array like "float32[]")
                              # This distinguishes:
                              # - "float32" (7 chars, starts with 'f', byte 7 is null) -> float
                              # - "float64" (7 chars, starts with 'f', byte 7 is null) -> float
                              # - "int32[]" (7 chars, starts with 'i', byte 7 is null) -> NOT float
                              # - "float32[]" (9 chars, starts with 'f', byte 7 is '[') -> NOT float
                              float_label = self.new_label("arg_float")
                              end_label = self.new_label("arg_end")
                              
                              self.output.append("    cmp byte ptr [rax], 102")  # Check if first char is 'f' (ASCII 102)
                              self.output.append(f"    jne {float_label}")  # Not a float if doesn't start with 'f', use int path (labeled as float for fall-through)
                              self.output.append("    cmp byte ptr [rax + 7], 0")  # Check if 8th byte is null (not an array)
                              self.output.append(f"    jne {float_label}")  # Not a scalar float if byte 7 is not null
                              
                              # Float path: read from XMM spill area (offsets 48-104)
                              self.output.append("    mov rax, r11")  # Restore index
                              self.output.append(f"    add rax, {start_idx}")
                              self.output.append("    shl rax, 3")  # rax = (idx + start_idx) * 8
                              # XMM spill area starts at spill_base - 48
                              # So offset from spill_base is 48 + idx*8
                              self.output.append("    add rax, 48")  # Adjust for XMM area
                              self.output.append(f"    mov r11, {spill_base}")
                              self.output.append("    sub r11, rax")
                              self.output.append("    neg r11")
                              self.output.append("    lea rax, [rbp + r11]")
                              self.output.append("    mov rax, [rax]")
                              self.output.append(f"    jmp {end_label}")
                              
                              # Integer path: read from GPR spill area (offsets 0-40)
                              self.output.append(f"{float_label}:")
                              self.output.append("    mov rax, r11")  # Restore index
                              self.output.append(f"    add rax, {start_idx}")
                              self.output.append("    shl rax, 3")  # rax = (idx + start_idx) * 8
                              self.output.append(f"    mov r11, {spill_base}")
                              self.output.append("    sub r11, rax")  # r11 = spill_base - offset
                              self.output.append("    neg r11")  # r11 = offset - spill_base
                              self.output.append("    lea rax, [rbp + r11]")
                              self.output.append("    mov rax, [rax]")
                              
                              self.output.append(f"{end_label}:")
                              
                              # If expected_type is float, move bits to xmm0
                              if expected_type in ['float32', 'float64']:
                                  self.output.append("    movq xmm0, rax")
                                  return expected_type
                              # If expected_type is array, return pointer type
                              if expected_type and expected_type.endswith('[]'):
                                  return expected_type
                              return "int64" # Treat as generic 64-bit value
             return "unknown"

        elif isinstance(expr, NoValueExpr):
             # Initialize to 0
             self.output.append("    xor rax, rax")
             
             if expected_type in ['float32', 'float64']:
                 if expected_type == 'float32':
                      self.output.append("    xorps xmm0, xmm0")
                 else:
                      self.output.append("    xorpd xmm0, xmm0")
                 return expected_type
             return expected_type if expected_type else 'int64'

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
                    # Optimization: Use LEA for add if both operands are in registers
                    # LEA is faster on some microarchitectures and doesn't affect flags
                    self.output.append("    lea rax, [rax + rbx]")
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
        # Handle dotted function names (e.g., math.add -> add)
        func_name = expr.name
        if '.' in func_name:
            func_name = func_name.split('.', 1)[1]
        
        func_params = self.function_params.get(func_name)
        
        # 1. Evaluate all arguments and store them in temporary locals
        # This prevents register clobbering during evaluation of subsequent arguments
        arg_temps = []
        actual_arg_types = []
        original_offset = self.next_local_offset
        
        for i, (arg_type, arg_expr) in enumerate(expr.args):
             # Determine promotion/conversion needs
             target_type = arg_type
             if arg_type == 'float32':
                  # Check if promotion to float64 is needed (varargs)
                  should_promote = False
                  if func_params is None:
                      should_promote = True
                  elif i >= len(func_params):
                      should_promote = True
                  else:
                      _, param_type = func_params[i]
                      if param_type.startswith('...'):
                          should_promote = True
                  
                  if should_promote:
                      target_type = 'float64'
             
             # Allocate temp
             temp_offset = self.alloc_temp(target_type)
             
             # Generate expression
             actual_type = self.gen_expression(arg_expr, expected_type=arg_type)
             # For variadic type info, use the declared arg_type, not the actual evaluated type
             # This ensures type tags match what the caller specified
             actual_arg_types.append(arg_type)
             
             # Convert if needed (e.g. float32 -> float64 promotion or int conversions)
             if target_type == 'float64' and actual_type == 'float32':
                  self.output.append("    cvtss2sd xmm0, xmm0")
             elif target_type != actual_type:
                  # General conversion
                  self.gen_conversion(actual_type, target_type)
             
             # Store result to temp
             if target_type == 'float32':
                  self.output.append(f"    movss [rbp - {temp_offset}], xmm0")
             elif target_type == 'float64':
                  self.output.append(f"    movsd [rbp - {temp_offset}], xmm0")
             else:
                  # For all integer types
                  self.output.append(f"    mov [rbp - {temp_offset}], rax")
                  
             arg_temps.append((temp_offset, target_type))



        # Robust Call Sequence: Align stack to 16 bytes (Save RSP to RBX)
        self.output.append("    mov rbx, rsp")
        self.output.append("    and rsp, -16")

        # 2. Populate registers/stack from temps
        if self.is_linux:
            # System V AMD64
            int_arg_regs = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
            float_arg_regs = [f"xmm{i}" for i in range(8)]
            int_arg_idx = 0
            float_arg_idx = 0
            
            for i, (temp_offset, arg_type) in enumerate(arg_temps):
                is_float = arg_type in ['float32', 'float64']
                
                if is_float:
                    if float_arg_idx < len(float_arg_regs):
                        reg = float_arg_regs[float_arg_idx]
                        if arg_type == 'float32':
                            self.output.append(f"    movss {reg}, [rbp - {temp_offset}]")
                        else:
                            self.output.append(f"    movsd {reg}, [rbp - {temp_offset}]")
                        float_arg_idx += 1
                    else:
                        pass # TODO: Stack args
                else:
                     if int_arg_idx < len(int_arg_regs):
                        reg = int_arg_regs[int_arg_idx]
                        if arg_type == 'string' and isinstance(expr.args[i][1], VarRefExpr) and expr.args[i][1].name in self.data_labels:
                             # Special case for string labels? No, they are evaluated to addresses in gen_expression
                             # So [rbp - temp_offset] contains the address.
                             self.output.append(f"    mov {reg}, [rbp - {temp_offset}]")
                        else:
                             self.output.append(f"    mov {reg}, [rbp - {temp_offset}]")
                        int_arg_idx += 1
                     else:
                        pass # TODO: Stack args

            # For varargs, AL must contain number of vector registers used
            # NOTE: We set this AFTER variadic info to avoid being overwritten
            saved_float_arg_idx = float_arg_idx
            
        else:
            # Windows x64
            arg_regs = ["rcx", "rdx", "r8", "r9"]
            float_arg_regs = ["xmm0", "xmm1", "xmm2", "xmm3"]
            
            # Handle stack args first (if any)
            # Windows stack args: Arg 4 at [rsp + 32], Arg 5 at [rsp + 40]...
            # We must use 'mov' to place them.
            # But we haven't allocated shadow space yet.
            # If we allocate shadow space now, rsp changes.
            
            # Allocate shadow space
            self.output.append("    sub rsp, 32")
            
            # Check for extra args
            if len(arg_temps) > 4:
                 # Need more stack space?
                 # Shadow space covers 4 args (sort of, but callee uses it).
                 # Args > 4 must be pushed or moved to stack ABOVE shadow space?
                 # NO.
                 # Shadow space is 32 bytes.
                 # Arg 4 is at rsp + 32.
                 # Arg 5 is at rsp + 40.
                 # Caller (us) must allocate this space.
                 # 'sub rsp, 32' only allocates shadow space.
                 # If we have 5 args, we need 32 + 8 = 40 bytes?
                 # Windows ABI: "The caller is responsible for allocating space for parameters... 
                 # and must always allocate at least 32 bytes (4 words) for the four register parameters...
                 # This 32 bytes is the shadow space."
                 # "Any parameters beyond the first four must be stored on the stack above the shadow space."
                 
                 extra_args = len(arg_temps) - 4
                 stack_args_size = extra_args * 8
                 # Align stack if needed? 
                 # RSP must be 16-aligned before call.
                 # 'sub rsp, 32' -> aligned?
                 # If total allocation (32 + extra) is not 16-aligned, we have an issue.
                 # But 'sub rsp, 32' is standard.
                 # If we need more space, we should subtract MORE.
                 total_stack_needed = 32 + stack_args_size
                 # Align total needed to 16 bytes
                 if total_stack_needed % 16 != 0:
                     total_stack_needed += 8 # Add padding
                 
                 # Adjust existing 'sub rsp, 32' to 'sub rsp, total'
                 self.output.pop() # Remove 'sub rsp, 32'
                 self.output.append(f"    sub rsp, {total_stack_needed}")
                 
                 # Now place stack args
                 # Shadow space is bottom 32 bytes (rsp to rsp+32).
                 # Arg 4 goes to rsp + 32.
                 for i in range(4, len(arg_temps)):
                     temp_offset, arg_type = arg_temps[i]
                     dest_offset = 32 + (i - 4) * 8
                     
                     # Move from temp (rbp-offset) to stack (rsp+dest)
                     # Can't mem-to-mem. Use RAX/XMM0.
                     
                     is_float = arg_type in ['float32', 'float64']
                     if is_float:
                         if arg_type == 'float32':
                             self.output.append(f"    movss xmm0, [rbp - {temp_offset}]")
                             self.output.append(f"    movss [rsp + {dest_offset}], xmm0")
                         else:
                             self.output.append(f"    movsd xmm0, [rbp - {temp_offset}]")
                             self.output.append(f"    movsd [rsp + {dest_offset}], xmm0")
                     else:
                         self.output.append(f"    mov rax, [rbp - {temp_offset}]")
                         self.output.append(f"    mov [rsp + {dest_offset}], rax")

            # Populate registers
            for i, (temp_offset, arg_type) in enumerate(arg_temps):
                if i >= 4: break
                
                is_float = arg_type in ['float32', 'float64']
                
                if is_float:
                    if arg_type == 'float32':
                         self.output.append(f"    movss xmm0, [rbp - {temp_offset}]")
                         self.output.append(f"    movss {float_arg_regs[i]}, xmm0")
                    else:
                         self.output.append(f"    movsd xmm0, [rbp - {temp_offset}]")
                         self.output.append(f"    movsd {float_arg_regs[i]}, xmm0")
                    
                    # Mirror to GPR for varargs
                    self.output.append(f"    movq {arg_regs[i]}, xmm0")
                else:
                    self.output.append(f"    mov rax, [rbp - {temp_offset}]")
                    self.output.append(f"    mov {arg_regs[i]}, rax")


        # Pass variadic info if needed for internal functions
        # Handle dotted function names (e.g., std.print -> print)
        check_func_name = expr.name
        if '.' in check_func_name:
            check_func_name = check_func_name.split('.', 1)[1]
        
        if check_func_name in self.internal_functions and check_func_name in self.variadic_info:
            vname, vstart = self.variadic_info[check_func_name]
            varargs_types = actual_arg_types[vstart:]
            count = len(varargs_types)
            
            if count > 0:
                array_label = self.new_label("type_array")
                self.output.append(f"    lea r10, [rip + {array_label}]")
                self.pending_type_arrays[array_label] = varargs_types
            else:
                self.output.append("    xor r10, r10")
            
            # Set count - use R11 on Linux to avoid conflict with AL
            # On Windows, use RAX as before
            if self.is_linux:
                self.output.append(f"    mov r11, {count}")
            else:
                self.output.append(f"    mov rax, {count}")
        
        # Set AL for Linux varargs (number of vector registers used)
        if self.is_linux:
            self.output.append(f"    mov al, {saved_float_arg_idx}")

        # Handle dotted function names (e.g., math.add -> add)
        func_name = expr.name
        if '.' in func_name:
            func_name = func_name.split('.', 1)[1]
        
        self.output.append(f"    call {func_name}")
        
        # Cleanup stack
        if self.is_windows:
            # Restore stack pointer
            # We calculated total_stack_needed above if > 4 args
            if len(arg_temps) > 4:
                 extra = len(arg_temps) - 4
                 sz = 32 + extra * 8
                 if sz % 16 != 0: sz += 8
                 self.output.append(f"    add rsp, {sz}")
            else:
                 self.output.append("    add rsp, 32")
                 
        # Restore RSP from RBX
        self.output.append("    mov rsp, rbx")

        # Reclaim temp locals
        self.next_local_offset = original_offset

    # =========================================================================
    # NEW: Optimized Code Generation Methods
    # =========================================================================
    
    def _emit_optimized_int_load(self, val: int):
        """
        Emit optimized integer load based on value.
        Uses smallest encoding possible.
        """
        if val == 0:
            # xor is smaller and faster than mov for zero
            self.output.append("    xor eax, eax")
        elif val == 1:
            # inc is smaller for getting 1
            self.output.append("    xor eax, eax")
            self.output.append("    inc eax")
        elif val == -1:
            # dec from 0 gives -1
            self.output.append("    xor eax, eax")
            self.output.append("    dec eax")
        elif 0 < val <= 0x7FFFFFFF:
            # 32-bit mov zero-extends to 64-bit (smaller encoding)
            self.output.append(f"    mov eax, {val}")
        elif -0x80000000 <= val < 0:
            # Negative 32-bit values
            self.output.append(f"    mov eax, {val}")
        else:
            # Full 64-bit immediate
            self.output.append(f"    mov rax, {val}")
    
    def _emit_optimized_add(self, val: int):
        """
        Emit optimized addition based on value.
        """
        if val == 0:
            pass  # No-op
        elif val == 1:
            self.output.append("    inc rax")
        elif val == -1:
            self.output.append("    dec rax")
        elif val > 0 and val <= 0x7FFFFFFF:
            self.output.append(f"    add rax, {val}")
        elif val < 0 and val >= -0x80000000:
            # Use sub with positive value
            self.output.append(f"    sub rax, {-val}")
        else:
            # Need to use register
            self.output.append(f"    mov rcx, {val}")
            self.output.append("    add rax, rcx")
    
    def _emit_optimized_mul(self, val: int):
        """
        Emit optimized multiplication using shifts and adds.
        """
        if val == 0:
            self.output.append("    xor eax, eax")
        elif val == 1:
            pass  # No-op
        elif val == -1:
            self.output.append("    neg rax")
        elif val > 0 and (val & (val - 1)) == 0:
            # Power of 2: use shift
            shift = val.bit_length() - 1
            self.output.append(f"    shl rax, {shift}")
        elif val == 3:
            # x * 3 = (x << 1) + x
            self.output.append("    mov rcx, rax")
            self.output.append("    shl rax, 1")
            self.output.append("    add rax, rcx")
        elif val == 5:
            # x * 5 = (x << 2) + x
            self.output.append("    mov rcx, rax")
            self.output.append("    shl rax, 2")
            self.output.append("    add rax, rcx")
        elif val == 6:
            # x * 6 = (x << 2) + (x << 1)
            self.output.append("    mov rcx, rax")
            self.output.append("    shl rcx, 1")
            self.output.append("    shl rax, 2")
            self.output.append("    add rax, rcx")
        elif val == 7:
            # x * 7 = (x << 3) - x
            self.output.append("    mov rcx, rax")
            self.output.append("    shl rax, 3")
            self.output.append("    sub rax, rcx")
        elif val == 9:
            # x * 9 = (x << 3) + x
            self.output.append("    mov rcx, rax")
            self.output.append("    shl rax, 3")
            self.output.append("    add rax, rcx")
        elif val == 10:
            # x * 10 = (x << 3) + (x << 1)
            self.output.append("    mov rcx, rax")
            self.output.append("    shl rcx, 1")
            self.output.append("    shl rax, 3")
            self.output.append("    add rax, rcx")
        else:
            # General case
            self.output.append(f"    imul rax, {val}")
    
    def _emit_optimized_div(self, val: int, is_signed: bool = True):
        """
        Emit optimized division using shifts for powers of 2.
        """
        if val == 0:
            raise RuntimeError("Division by zero")
        elif val == 1:
            pass  # No-op
        elif val == -1:
            self.output.append("    neg rax")
        elif val > 0 and (val & (val - 1)) == 0:
            # Power of 2: use shift
            shift = val.bit_length() - 1
            if is_signed:
                # For signed division, we need to handle sign
                self.output.append("    mov rcx, rax")
                self.output.append("    sar rcx, 63")  # Get sign bits
                self.output.append(f"    shr rcx, {64 - shift}")  # Adjust for rounding
                self.output.append("    add rax, rcx")  # Round toward zero
                self.output.append(f"    sar rax, {shift}")
            else:
                self.output.append(f"    shr rax, {shift}")
        else:
            # General case: use idiv
            self.output.append("    cqo")
            self.output.append(f"    mov rcx, {val}")
            self.output.append("    idiv rcx")
    
    def _emit_optimized_mod(self, val: int):
        """
        Emit optimized modulo using bitwise AND for powers of 2.
        """
        if val == 0:
            raise RuntimeError("Modulo by zero")
        elif val == 1:
            self.output.append("    xor eax, eax")  # x % 1 = 0
        elif val > 0 and (val & (val - 1)) == 0:
            # Power of 2: use bitwise AND
            self.output.append(f"    and rax, {val - 1}")
        else:
            # General case: use idiv, result in rdx
            self.output.append("    cqo")
            self.output.append(f"    mov rcx, {val}")
            self.output.append("    idiv rcx")
            self.output.append("    mov rax, rdx")
    
    def _emit_branch_hint(self, likely: bool):
        """
        Emit branch prediction hint for conditional jumps.
        Uses Intel's branch hint prefixes.
        """
        if self.optimization_level >= 3:
            # Note: Modern CPUs mostly ignore these, but they can help in some cases
            if likely:
                self.output.append("    # likely taken")
            else:
                self.output.append("    # unlikely taken")
    
    def _emit_optimized_comparison(self, left_type: str, op: str):
        """
        Emit optimized comparison sequence.
        """
        # Use test for comparison with zero (smaller encoding)
        if op == '==' or op == '!=':
            # test rax, rax is smaller than cmp rax, 0
            self.output.append("    test rax, rax")
        else:
            self.output.append("    cmp rax, 0")
    
    def _should_use_lea(self, base_offset: int, index_scale: int = 0) -> bool:
        """
        Determine if LEA would be more efficient than multiple adds.
        """
        # LEA can compute base + index*scale + disp in one instruction
        # More efficient when we have both base and index
        return self.optimization_level >= 2 and (index_scale > 0 or base_offset != 0)
    
    def _emit_efficient_address(self, base_reg: str, offset: int, index_reg: str = None, scale: int = 1):
        """
        Emit the most efficient address calculation.
        Uses LEA when beneficial.
        """
        if index_reg:
            # Complex addressing: use LEA
            if offset == 0:
                self.output.append(f"    lea rax, [{base_reg} + {index_reg}*{scale}]")
            else:
                self.output.append(f"    lea rax, [{base_reg} + {index_reg}*{scale} + {offset}]")
        elif offset != 0:
            if self.optimization_level >= 2 and abs(offset) <= 0x7FFFFFFF:
                # Use LEA for addition (doesn't affect flags)
                self.output.append(f"    lea rax, [{base_reg} + {offset}]")
            else:
                self.output.append(f"    mov rax, {base_reg}")
                if offset > 0:
                    self.output.append(f"    add rax, {offset}")
                else:
                    self.output.append(f"    sub rax, {-offset}")
        else:
            self.output.append(f"    mov rax, {base_reg}")
