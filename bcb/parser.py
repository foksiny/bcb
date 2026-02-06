import os
from .lexer import tokenize, TokenType

class ASTNode:
    def __init__(self, line=0, column=0):
        self.line = line
        self.column = column

class Program(ASTNode):
    def __init__(self, outtype, data_block, declarations, line=0, column=0):
        super().__init__(line, column)
        self.outtype = outtype
        self.data_block = data_block
        self.declarations = declarations

class StructDef(ASTNode):
    def __init__(self, name, fields, line=0, column=0):
        super().__init__(line, column)
        self.name = name
        self.fields = fields

class EnumDef(ASTNode):
    def __init__(self, name, values, line=0, column=0):
        super().__init__(line, column)
        self.name = name
        self.values = values

class DataBlock(ASTNode):
    def __init__(self, entries, structs=None, enums=None, line=0, column=0):
        super().__init__(line, column)
        self.entries = entries
        self.structs = structs or []
        self.enums = enums or []

class FunctionDecl(ASTNode):
    def __init__(self, name, params, return_type, line=0, column=0):
        super().__init__(line, column)
        self.name = name
        self.params = params
        self.return_type = return_type

class FunctionDef(ASTNode):
    def __init__(self, name, params, return_type, body, is_exported, line=0, column=0):
        super().__init__(line, column)
        self.name = name
        self.params = params
        self.return_type = return_type
        self.body = body
        self.is_exported = is_exported

class GlobalVarDecl(ASTNode):
    def __init__(self, type_name, name, expr, is_array=False, array_size=None, line=0, column=0):
        super().__init__(line, column)
        self.type_name = type_name
        self.name = name
        self.expr = expr
        self.is_array = is_array
        self.array_size = array_size

class CallExpr(ASTNode):
    def __init__(self, name, args, line=0, column=0):
        super().__init__(line, column)
        self.name = name
        self.args = args

class ReturnStmt(ASTNode):
    def __init__(self, return_type, expr, line=0, column=0):
        super().__init__(line, column)
        self.return_type = return_type
        self.expr = expr

class VarDeclStmt(ASTNode):
    def __init__(self, type_name, name, expr, is_array=False, array_size=None, line=0, column=0):
        super().__init__(line, column)
        self.type_name = type_name
        self.name = name
        self.expr = expr
        self.is_array = is_array
        self.array_size = array_size

class BinaryExpr(ASTNode):
    def __init__(self, left, op, right, line=0, column=0):
        super().__init__(line, column)
        self.left = left
        self.op = op
        self.right = right

class UnaryExpr(ASTNode):
    def __init__(self, op, expr, line=0, column=0):
        super().__init__(line, column)
        self.op = op
        self.expr = expr

class LiteralExpr(ASTNode):
    def __init__(self, value, line=0, column=0):
        super().__init__(line, column)
        self.value = value

class VarRefExpr(ASTNode):
    def __init__(self, name, line=0, column=0):
        super().__init__(line, column)
        self.name = name

class TypeCastExpr(ASTNode):
    def __init__(self, target_type, expr, line=0, column=0):
        super().__init__(line, column)
        self.target_type = target_type
        self.expr = expr

class StructLiteralExpr(ASTNode):
    def __init__(self, struct_type, field_values, line=0, column=0):
        super().__init__(line, column)
        self.struct_type = struct_type
        self.field_values = field_values

class FieldAccessExpr(ASTNode):
    def __init__(self, obj, field_name, line=0, column=0):
        super().__init__(line, column)
        self.obj = obj
        self.field_name = field_name

class EnumValueExpr(ASTNode):
    def __init__(self, enum_name, value_name, line=0, column=0):
        super().__init__(line, column)
        self.enum_name = enum_name
        self.value_name = value_name

class IfStmt(ASTNode):
    def __init__(self, conditions_and_bodies, line=0, column=0):
        super().__init__(line, column)
        self.conditions_and_bodies = conditions_and_bodies

class WhileStmt(ASTNode):
    def __init__(self, condition, body, line=0, column=0):
        super().__init__(line, column)
        self.condition = condition
        self.body = body

class VarAssignStmt(ASTNode):
    def __init__(self, type_name, name, expr, line=0, column=0):
        super().__init__(line, column)
        self.type_name = type_name
        self.name = name
        self.expr = expr

class FieldAssignStmt(ASTNode):
    def __init__(self, type_name, var_name, field_name, expr, line=0, column=0):
        super().__init__(line, column)
        self.type_name = type_name
        self.var_name = var_name
        self.field_name = field_name
        self.expr = expr

class ArrayAccessExpr(ASTNode):
    def __init__(self, arr, index, line=0, column=0):
        super().__init__(line, column)
        self.arr = arr
        self.index = index

class ArrayLiteralExpr(ASTNode):
    def __init__(self, values, line=0, column=0):
        super().__init__(line, column)
        self.values = values

class ArrayAssignStmt(ASTNode):
    def __init__(self, type_name, arr_name, index, expr, line=0, column=0):
        super().__init__(line, column)
        self.type_name = type_name
        self.arr_name = arr_name
        self.index = index
        self.expr = expr

class LengthExpr(ASTNode):
    def __init__(self, expr, line=0, column=0):
        super().__init__(line, column)
        self.expr = expr

class LabelDef(ASTNode):
    def __init__(self, name, line=0, column=0):
        super().__init__(line, column)
        self.name = name

class JmpStmt(ASTNode):
    def __init__(self, target, line=0, column=0):
        super().__init__(line, column)
        self.target = target

class IfnStmt(ASTNode):
    def __init__(self, condition, target, line=0, column=0):
        super().__init__(line, column)
        self.condition = condition
        self.target = target

class CmpTStmt(ASTNode):
    def __init__(self, condition, target, line=0, column=0):
        super().__init__(line, column)
        self.condition = condition
        self.target = target

class PushStmt(ASTNode):
    def __init__(self, type_name, expr, line=0, column=0):
        super().__init__(line, column)
        self.type_name = type_name
        self.expr = expr

class PopStmt(ASTNode):
    def __init__(self, type_name, var_name, line=0, column=0):
        super().__init__(line, column)
        self.type_name = type_name
        self.var_name = var_name

class NoValueExpr(ASTNode):
    def __init__(self, line=0, column=0):
        super().__init__(line, column)

class Parser:
    def __init__(self, tokens, base_dir=".", imported_files=None):
        self.tokens = tokens
        self.pos = 0
        self.enum_names = set()  # Track known enum names
        self.struct_names = set() # Track known struct names
        self.base_dir = base_dir
        self.imported_files = imported_files if imported_files is not None else set()

    def peek(self, offset=0):
        if self.pos + offset >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[self.pos + offset]

    def consume(self, expected_type=None, expected_value=None):
        token = self.peek()
        if expected_type and token.type != expected_type:
            raise RuntimeError(f"Expected {expected_type}, got {token.type} at line {token.line}")
        if expected_value and token.value != expected_value:
            raise RuntimeError(f"Expected {repr(expected_value)}, got {repr(token.value)} at line {token.line}")
        self.pos += 1
        return token

    def parse(self):
        outtype = None
        data_block = None
        declarations = []

        while self.peek().type != TokenType.EOF:
            token = self.peek()
            if token.type == TokenType.SYMBOL and token.value == '<':
                self.consume() # <
                if self.peek().value == 'outtype':
                    self.consume() # outtype
                    self.consume(TokenType.SYMBOL, ':')
                    outtype = self.consume(TokenType.IDENTIFIER).value
                    self.consume(TokenType.SYMBOL, '>')
                else:
                    # handle other tags if they exist
                    pass
            elif token.type == TokenType.KEYWORD:
                if token.value == 'data':
                    data_block = self.parse_data_block()
                elif token.value == 'define':
                    declarations.append(self.parse_function_decl())
                elif token.value == 'export':
                    declarations.append(self.parse_function_def(True))
                elif token.value == 'pub':
                    declarations.append(self.parse_global_var())
                elif token.value == 'import':
                    self.consume() # import
                    import_path = self.consume(TokenType.STRING).value
                    self.consume(TokenType.SYMBOL, ';')
                    
                    full_path = os.path.abspath(os.path.join(self.base_dir, import_path))
                    if full_path not in self.imported_files:
                        self.imported_files.add(full_path)
                        if os.path.exists(full_path):
                            with open(full_path, 'r') as f:
                                import_code = f.read()
                            import_tokens = tokenize(import_code)
                            import_parser = Parser(import_tokens, os.path.dirname(full_path), self.imported_files)
                            import_program = import_parser.parse()
                            
                            # Merge declarations
                            declarations.extend(import_program.declarations)
                            
                            # Merge data block
                            if import_program.data_block:
                                if not data_block:
                                    data_block = DataBlock([], [], [])
                                data_block.entries.extend(import_program.data_block.entries)
                                data_block.structs.extend(import_program.data_block.structs)
                                data_block.enums.extend(import_program.data_block.enums)
                                # Also update enum_names from the imported parser
                                self.enum_names.update(import_parser.enum_names)
                            
                            # Merge outtype if not already set
                            if not outtype:
                                outtype = import_program.outtype
                        else:
                            raise RuntimeError(f"Imported file not found: {full_path}")
                else:
                    raise RuntimeError(f"Unexpected keyword {token.value}")
            elif token.type == TokenType.IDENTIFIER:
                # Potential function definition without export
                if self.peek(1).type == TokenType.SYMBOL and self.peek(1).value == '(':
                    declarations.append(self.parse_function_def(False))
                else:
                    self.consume()
            else:
                self.consume() # Skip unknown tokens for now

        return Program(outtype, data_block, declarations, 1, 1)

    def parse_data_block(self):
        start_token = self.consume(TokenType.KEYWORD, 'data')
        self.consume(TokenType.SYMBOL, '{')
        entries = []
        structs = []
        enums = []
        while self.peek().type != TokenType.SYMBOL or self.peek().value != '}':
            if self.peek().type == TokenType.KEYWORD and self.peek().value == 'struct':
                s_def = self.parse_struct_def()
                structs.append(s_def)
                self.struct_names.add(s_def.name)
            elif self.peek().type == TokenType.KEYWORD and self.peek().value == 'enum':
                enums.append(self.parse_enum_def())
            else:
                type_name = self.consume(TokenType.KEYWORD).value
                name = self.consume(TokenType.IDENTIFIER).value
                self.consume(TokenType.SYMBOL, ':')
                value = self.consume().value  # Could be string or number
                entries.append((type_name, name, value))
        self.consume(TokenType.SYMBOL, '}')
        return DataBlock(entries, structs, enums, start_token.line, start_token.column)

    def parse_struct_def(self):
        start_token = self.consume(TokenType.KEYWORD, 'struct')
        name = self.consume(TokenType.IDENTIFIER).value
        self.consume(TokenType.SYMBOL, '{')
        fields = []
        while self.peek().type != TokenType.SYMBOL or self.peek().value != '}':
            field_type_token = self.consume()
            if field_type_token.type not in [TokenType.KEYWORD, TokenType.IDENTIFIER]:
                 raise RuntimeError(f"Expected field type, got {field_type_token.type} at line {field_type_token.line}")
            field_type = field_type_token.value
            
            while self.peek().type == TokenType.SYMBOL and self.peek().value == '*':
                self.consume()
                field_type += '*'
            
            # Array field? (e.g. int32 arr[10])
            # The current struct def implementation expects "type name;"
            # We need to peek later if brackets exist after name.
            field_name = self.consume(TokenType.IDENTIFIER).value
            
            if self.peek().type == TokenType.SYMBOL and self.peek().value == '[':
                # Array field
                self.consume() # [
                size = self.consume(TokenType.NUMBER).value
                self.consume(TokenType.SYMBOL, ']') # ]
                field_type += f"[{size}]" # Encode size in type string for now? Or handle differently?
                # CodeGen needs to know size. 'int32[10]' as type string.
            
            self.consume(TokenType.SYMBOL, ';')
            fields.append((field_type, field_name))
        self.consume(TokenType.SYMBOL, '}')
        return StructDef(name, fields, start_token.line, start_token.column)

    def parse_enum_def(self):
        start_token = self.consume(TokenType.KEYWORD, 'enum')
        name = self.consume(TokenType.IDENTIFIER).value
        self.enum_names.add(name)  # Register enum name
        self.consume(TokenType.SYMBOL, '{')
        values = []
        while self.peek().type != TokenType.SYMBOL or self.peek().value != '}':
            value_name = self.consume(TokenType.IDENTIFIER).value
            values.append(value_name)
            # Optional comma
            if self.peek().type == TokenType.SYMBOL and self.peek().value == ',':
                self.consume()
        self.consume(TokenType.SYMBOL, '}')
        return EnumDef(name, values, start_token.line, start_token.column)

    def parse_global_var(self):
        start_token = self.consume(TokenType.KEYWORD, 'pub')
        type_token = self.peek()
        if type_token.type not in [TokenType.KEYWORD, TokenType.IDENTIFIER]:
            raise RuntimeError(f"Expected type name, got {type_token.type} at line {type_token.line}")
        type_name = self.consume().value
        while self.peek().type == TokenType.SYMBOL and self.peek().value == '*':
            self.consume()
            type_name += '*'
        name = self.consume(TokenType.IDENTIFIER).value
        
        is_array = False
        array_size = None
        if self.peek().type == TokenType.SYMBOL and self.peek().value == '[':
            self.consume() # [
            size_token = self.consume(TokenType.NUMBER)
            array_size = int(size_token.value)
            self.consume(TokenType.SYMBOL, ']')
            is_array = True
        
        self.consume(TokenType.SYMBOL, '=')
        expr = self.parse_expression()
        self.consume(TokenType.SYMBOL, ';')
        return GlobalVarDecl(type_name, name, expr, is_array, array_size, start_token.line, start_token.column)

    def parse_function_decl(self):
        start_token = self.consume(TokenType.KEYWORD, 'define')
        name = self.consume(TokenType.IDENTIFIER).value
        self.consume(TokenType.SYMBOL, '(')
        params = self.parse_params()
        self.consume(TokenType.SYMBOL, ')')
        self.consume(TokenType.SYMBOL, '-')
        self.consume(TokenType.SYMBOL, '>')
        # Return type (allow identifiers for structs/enums and pointer suffixes)
        type_token = self.peek()
        if type_token.type not in [TokenType.KEYWORD, TokenType.IDENTIFIER]:
            raise RuntimeError(f"Expected return type, got {type_token.type} at line {type_token.line}")
        return_type = self.consume().value
        while self.peek().type == TokenType.SYMBOL and self.peek().value == '*':
            self.consume()
            return_type += '*'
        while self.peek().type == TokenType.SYMBOL and self.peek().value == '*':
            self.consume()
            return_type += '*'
        self.consume(TokenType.SYMBOL, ';')
        return FunctionDecl(name, params, return_type, start_token.line, start_token.column)

    def parse_function_def(self, is_exported):
        if is_exported:
            start_token = self.consume(TokenType.KEYWORD, 'export')
        else:
            start_token = self.peek() # Identifier or token before it
        name = self.consume(TokenType.IDENTIFIER).value
        self.consume(TokenType.SYMBOL, '(')
        params = self.parse_params()
        self.consume(TokenType.SYMBOL, ')')
        self.consume(TokenType.SYMBOL, '-')
        self.consume(TokenType.SYMBOL, '>')
        # Return type (allow identifiers and pointer suffixes)
        type_token = self.peek()
        if type_token.type not in [TokenType.KEYWORD, TokenType.IDENTIFIER]:
            raise RuntimeError(f"Expected return type, got {type_token.type} at line {type_token.line}")
        return_type = self.consume().value
        while self.peek().type == TokenType.SYMBOL and self.peek().value == '*':
            self.consume()
            return_type += '*'
        self.consume(TokenType.SYMBOL, '{')
        body = []
        while self.peek().type != TokenType.SYMBOL or self.peek().value != '}':
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
        self.consume(TokenType.SYMBOL, '}')
        return FunctionDef(name, params, return_type, body, is_exported, start_token.line, start_token.column)

    def parse_params(self):
        params = []
        if self.peek().type == TokenType.KEYWORD and self.peek().value == 'void':
            self.consume()
            return params
        
        while self.peek().type != TokenType.SYMBOL or self.peek().value != ')':
            name = self.consume(TokenType.IDENTIFIER).value
            self.consume(TokenType.SYMBOL, ':')
            # Handle ...args
            if self.peek().type == TokenType.SYMBOL and self.peek().value == '.':
                self.consume() # .
                self.consume() # .
                self.consume() # .
                if self.peek().type == TokenType.IDENTIFIER:
                    var_name = self.consume().value
                    type_name = f"...{var_name}"
                else:
                    type_name = "..."
            else:
                # Base type (keyword or identifier, e.g., int32, MyStruct)
                type_token = self.peek()
                if type_token.type not in [TokenType.KEYWORD, TokenType.IDENTIFIER]:
                    raise RuntimeError(f"Expected type name, got {type_token.type} at line {type_token.line}")
                type_name = self.consume().value
                # Optional pointer suffix(es) or array brackets, e.g., int32*, int32[]
                while True:
                    if self.peek().type == TokenType.SYMBOL and self.peek().value == '*':
                        self.consume()
                        type_name += '*'
                    elif self.peek().type == TokenType.SYMBOL and self.peek().value == '[':
                        self.consume() # [
                        self.consume(TokenType.SYMBOL, ']') # ]
                        type_name += '[]'
                    else:
                        break
            params.append((name, type_name))
            if self.peek().type == TokenType.SYMBOL and self.peek().value == ',':
                self.consume()
        return params

    def parse_statement(self):
        token = self.peek()
        if token.type == TokenType.KEYWORD:
            if token.value == 'call':
                return self.parse_call_stmt()
            elif token.value == 'return':
                self.consume()
                ret_type = self.consume(TokenType.KEYWORD).value
                if ret_type == 'void':
                    res_token = self.consume(TokenType.SYMBOL, ';')
                    return ReturnStmt(ret_type, None, res_token.line, res_token.column)
                expr = self.parse_expression()
                self.consume(TokenType.SYMBOL, ';')
                # Use 'return' token loc? Yes, that was consumed earlier.
                # Accessing token from start of statement logic
                return ReturnStmt(ret_type, expr, token.line, token.column)
            elif token.value == '$if':
                return self.parse_if_stmt()
            elif token.value == '$while':
                return self.parse_while_stmt()
            elif token.value == 'md':
                self.consume()  # md
                # Allow KEYWORD or IDENTIFIER for type_name (e.g., md int32 x = ... or md Color c = ...)
                type_token = self.peek()
                if type_token.type not in [TokenType.KEYWORD, TokenType.IDENTIFIER]:
                    raise RuntimeError(f"Expected type name, got {type_token.type} at line {type_token.line}")
                type_name = self.consume().value
                # Optional pointer suffixes for md (e.g., md int32* ptr = 2;)
                while self.peek().type == TokenType.SYMBOL and self.peek().value == '*':
                    self.consume()
                    type_name += '*'
                name = self.consume(TokenType.IDENTIFIER).value
                
                # Check if it's a field assignment (var.field)
                if self.peek().type == TokenType.SYMBOL and self.peek().value == '.':
                    self.consume()  # .
                    field_name = self.consume(TokenType.IDENTIFIER).value
                    self.consume(TokenType.SYMBOL, '=')
                    expr = self.parse_expression()
                    self.consume(TokenType.SYMBOL, ';')
                    return FieldAssignStmt(type_name, name, field_name, expr, token.line, token.column)
                elif self.peek().type == TokenType.SYMBOL and self.peek().value == '[':
                    self.consume() # [
                    index = self.parse_expression()
                    self.consume(TokenType.SYMBOL, ']') # ]
                    self.consume(TokenType.SYMBOL, '=')
                    expr = self.parse_expression()
                    self.consume(TokenType.SYMBOL, ';')
                    return ArrayAssignStmt(type_name, name, index, expr, token.line, token.column)
                else:
                    self.consume(TokenType.SYMBOL, '=')
                    expr = self.parse_expression()
                    self.consume(TokenType.SYMBOL, ';')
                    return VarAssignStmt(type_name, name, expr, token.line, token.column)
            elif token.value == 'jmp':
                self.consume()
                target = self.consume(TokenType.LABEL).value
                self.consume(TokenType.SYMBOL, ';')
                return JmpStmt(target, token.line, token.column)
            elif token.value == 'ifn':
                self.consume()
                cond = self.consume(TokenType.IDENTIFIER).value
                self.consume(TokenType.SYMBOL, ',')
                target = self.consume(TokenType.LABEL).value
                self.consume(TokenType.SYMBOL, ';')
                return IfnStmt(cond, target, token.line, token.column)
            elif token.value == 'cmp_t':
                self.consume()
                cond = self.consume(TokenType.IDENTIFIER).value
                self.consume(TokenType.SYMBOL, ',')
                target = self.consume(TokenType.LABEL).value
                self.consume(TokenType.SYMBOL, ';')
                return CmpTStmt(cond, target, token.line, token.column)
            elif token.value == 'push':
                self.consume()
                # Parse type
                type_token = self.peek()
                if type_token.type not in [TokenType.KEYWORD, TokenType.IDENTIFIER]:
                     raise RuntimeError(f"Expected type name, got {type_token.type} at line {type_token.line}")
                type_name = self.consume().value
                while self.peek().type == TokenType.SYMBOL and self.peek().value == '*':
                    self.consume()
                    type_name += '*'
                
                expr = self.parse_expression()
                self.consume(TokenType.SYMBOL, ';')
                return PushStmt(type_name, expr, token.line, token.column)
            elif token.value == 'pop':
                self.consume()
                # Parse type
                type_token = self.peek()
                if type_token.type not in [TokenType.KEYWORD, TokenType.IDENTIFIER]:
                     raise RuntimeError(f"Expected type name, got {type_token.type} at line {type_token.line}")
                type_name = self.consume().value
                while self.peek().type == TokenType.SYMBOL and self.peek().value == '*':
                    self.consume()
                    type_name += '*'
                
                var_name = self.consume(TokenType.IDENTIFIER).value
                self.consume(TokenType.SYMBOL, ';')
                return PopStmt(type_name, var_name, token.line, token.column)
            elif token.value in ['int8', 'int16', 'int32', 'int64', 'float32', 'float64', 'string', 'char']:
                # Variable declaration, including pointer types (e.g., int32* ptr)
                type_name = self.consume().value
                while self.peek().type == TokenType.SYMBOL and self.peek().value == '*':
                    self.consume()
                    type_name += '*'
                name = self.consume(TokenType.IDENTIFIER).value
                
                is_array = False
                array_size = None
                if self.peek().type == TokenType.SYMBOL and self.peek().value == '[':
                    self.consume() # [
                    size_token = self.consume(TokenType.NUMBER)
                    array_size = int(size_token.value)
                    self.consume(TokenType.SYMBOL, ']')
                    is_array = True
                
                self.consume(TokenType.SYMBOL, '=')
                expr = self.parse_expression()
                self.consume(TokenType.SYMBOL, ';')
                return VarDeclStmt(type_name, name, expr, is_array, array_size, token.line, token.column)
        elif token.type == TokenType.LABEL:
            label_name = self.consume().value
            self.consume(TokenType.SYMBOL, ':')
            return LabelDef(label_name, token.line, token.column)
        elif token.type == TokenType.IDENTIFIER:
            # Could be struct variable declaration: StructName varName = { ... };
            struct_type = self.consume(TokenType.IDENTIFIER).value
            var_name = self.consume(TokenType.IDENTIFIER).value
            
            is_array = False
            array_size = None
            if self.peek().type == TokenType.SYMBOL and self.peek().value == '[':
                self.consume() # [
                size_token = self.consume(TokenType.NUMBER)
                array_size = int(size_token.value)
                self.consume(TokenType.SYMBOL, ']')
                is_array = True

            self.consume(TokenType.SYMBOL, '=')
            expr = self.parse_expression()
            self.consume(TokenType.SYMBOL, ';')
            return VarDeclStmt(struct_type, var_name, expr, is_array, array_size, token.line, token.column)
        raise RuntimeError(f"Unknown statement {token}")

    def parse_expression(self, min_prec=1):
        token = self.peek()

        # Handle Casts and Unary Operators (Prefix)
        if token.type == TokenType.SYMBOL and token.value == '-' and self.peek(1).type == TokenType.NUMBER:
             self.consume() # -
             num_token = self.consume()
             return LiteralExpr(-num_token.value, token.line, token.column)
             
        # Support both primitive keywords and custom struct/enum types as casts
        is_type = False
        if token.type == TokenType.KEYWORD and token.value in ['int8', 'int16', 'int32', 'int64', 'float32', 'float64', 'string', 'char', 'void']:
             is_type = True
        elif token.type == TokenType.IDENTIFIER and (token.value in self.struct_names or token.value in self.enum_names):
             # Only treat as type cast if NOT followed by '.' (Enum.Value access)
             if self.peek(1).type != TokenType.SYMBOL or self.peek(1).value != '.':
                 is_type = True
        
        if is_type:
            type_name = self.consume().value
            # Handle pointers and array brackets (e.g. int32* x, int32[] x)
            while True:
                if self.peek().type == TokenType.SYMBOL and self.peek().value == '*':
                    self.consume()
                    type_name += '*'
                elif self.peek().type == TokenType.SYMBOL and self.peek().value == '[':
                    self.consume() # [
                    self.consume(TokenType.SYMBOL, ']') # ]
                    type_name += '[]'
                else:
                    break

            if self.peek().type == TokenType.SYMBOL and self.peek().value == '(':
                # type(expr)
                self.consume(TokenType.SYMBOL, '(')
                expr = self.parse_expression(1)
                self.consume(TokenType.SYMBOL, ')')
                lhs = TypeCastExpr(type_name, expr, token.line, token.column)
            else:
                # type expr
                rhs = self.parse_expression(11) # High precedence
                lhs = TypeCastExpr(type_name, rhs, token.line, token.column)
            return self.parse_op_continuation(lhs, min_prec)
            
        elif token.type == TokenType.SYMBOL and token.value in ['~', '&', '*']:
            op = self.consume().value
            rhs = self.parse_expression(11) # High precedence
            lhs = UnaryExpr(op, rhs, token.line, token.column)
            return self.parse_op_continuation(lhs, min_prec)
            
        lhs = self.parse_primary()
        return self.parse_op_continuation(lhs, min_prec)

    def parse_op_continuation(self, lhs, min_prec):
        PRECEDENCE = {
            '||': 1,
            '&&': 2,
            '|': 3,
            '^': 4,
            '&': 5,
            '==': 6, '!=': 6,
            '<': 7, '>': 7, '<=': 7, '>=': 7,
            '<<': 8, '>>': 8,
            '+': 9, '-': 9,
            '*': 10, '/': 10, '%': 10
        }
        
        while True:
            token = self.peek()
            if token.type != TokenType.SYMBOL or token.value not in PRECEDENCE:
                break
            
            op = token.value
            prec = PRECEDENCE[op]
            
            if prec < min_prec:
                break
                
            self.consume()
            rhs = self.parse_expression(prec + 1)
            # BinaryExpr should take location of the operator? Or LHS? usually op.
            lhs = BinaryExpr(lhs, op, rhs, token.line, token.column)
            
        return lhs

    def parse_if_stmt(self):
        conditions_and_bodies = []
        
        # Parse $if
        start_token = self.consume(TokenType.KEYWORD, '$if')
        condition = self.parse_expression()
        body = []
        while self.peek().type != TokenType.KEYWORD or self.peek().value not in ['$elseif', '$else', '$endif']:
            body.append(self.parse_statement())
        conditions_and_bodies.append((condition, body))
        
        # Parse $elseif
        while self.peek().type == TokenType.KEYWORD and self.peek().value == '$elseif':
            self.consume()
            condition = self.parse_expression()
            body = []
            while self.peek().type != TokenType.KEYWORD or self.peek().value not in ['$elseif', '$else', '$endif']:
                body.append(self.parse_statement())
            conditions_and_bodies.append((condition, body))
            
        # Parse $else
        if self.peek().type == TokenType.KEYWORD and self.peek().value == '$else':
            self.consume()
            body = []
            while self.peek().type != TokenType.KEYWORD or self.peek().value != '$endif':
                body.append(self.parse_statement())
            conditions_and_bodies.append((None, body))
            
        self.consume(TokenType.KEYWORD, '$endif')
        return IfStmt(conditions_and_bodies, start_token.line, start_token.column)

    def parse_while_stmt(self):
        start_token = self.consume(TokenType.KEYWORD, '$while')
        condition = self.parse_expression()
        body = []
        while self.peek().type != TokenType.KEYWORD or self.peek().value != '$endwhile':
            body.append(self.parse_statement())
        self.consume(TokenType.KEYWORD, '$endwhile')
        return WhileStmt(condition, body, start_token.line, start_token.column)

    def parse_primary(self):
        token = self.peek()
        if token.type == TokenType.NUMBER:
            return LiteralExpr(self.consume().value, token.line, token.column)
        elif token.type == TokenType.STRING:
            return LiteralExpr(self.consume().value, token.line, token.column)
        elif token.type == TokenType.CHAR:
            return LiteralExpr(self.consume().value, token.line, token.column)
        elif token.type == TokenType.IDENTIFIER and token.value == 'length':
            self.consume() # length
            self.consume(TokenType.SYMBOL, '(')
            expr = self.parse_expression()
            self.consume(TokenType.SYMBOL, ')')
            return LengthExpr(expr, token.line, token.column)
        elif token.type == TokenType.KEYWORD and token.value == 'call':
            self.consume()  # call
            name = self.consume(TokenType.IDENTIFIER).value
            self.consume(TokenType.SYMBOL, '(')
            args = []
            while self.peek().type != TokenType.SYMBOL or self.peek().value != ')':
                # Argument type (base keyword or identifier; pointer/deref handled in expression)
                type_token = self.peek()
                if type_token.type not in [TokenType.KEYWORD, TokenType.IDENTIFIER]:
                    raise RuntimeError(f"Expected argument type, got {type_token.type} at line {type_token.line}")
                arg_type = self.consume().value
                while self.peek().type == TokenType.SYMBOL and self.peek().value == '*':
                    self.consume()
                    arg_type += '*'
                arg_expr = self.parse_expression()
                args.append((arg_type, arg_expr))
                if self.peek().type == TokenType.SYMBOL and self.peek().value == ',':
                    self.consume()
            self.consume(TokenType.SYMBOL, ')')
            return CallExpr(name, args, token.line, token.column)
        elif token.type == TokenType.KEYWORD and token.value == 'no_value':
            self.consume()
            return NoValueExpr(token.line, token.column)
        elif token.type == TokenType.SYMBOL and token.value == '(':
            self.consume()  # (
            expr = self.parse_expression()
            self.consume(TokenType.SYMBOL, ')')
            return expr
        elif token.type == TokenType.SYMBOL and token.value == '{':
            # Check if it looks like a struct literal (name :) or array literal
            is_struct = False
            if self.peek(1).type == TokenType.IDENTIFIER and self.peek(2).type == TokenType.SYMBOL and self.peek(2).value == ':':
                 is_struct = True
            
            if is_struct:
                return self.parse_struct_literal()
            else:
                return self.parse_array_literal()
        elif token.type == TokenType.IDENTIFIER:
            name = self.consume().value
            
            if name in self.enum_names and self.peek().type == TokenType.SYMBOL and self.peek().value == '.':
                self.consume() # .
                val_name = self.consume(TokenType.IDENTIFIER).value
                return EnumValueExpr(name, val_name, token.line, token.column)

            expr = VarRefExpr(name, token.line, token.column)
            
            while True:
                if self.peek().type == TokenType.SYMBOL and self.peek().value == '.':
                    # Enum checking inside loop? Enum is only topmost?
                    # Original code checked enum_names on the first identifier.
                    # We can keep that check if it's the *first* iteration and purely identifier.
                    # But if we are here, we already consumed the first identifier.
                    # So we need to handle Enum special case BEFORE creating VarRefExpr or wrap it.
                    # But wait, original code did:
                    # if name in self.enum_names: ... return EnumValueExpr
                    # element access and enum access usually don't mix (Enum.VAL.field is unlikely in this lang?)
                    
                    dot_token = self.consume()
                    field_name = self.consume(TokenType.IDENTIFIER).value
                    expr = FieldAccessExpr(expr, field_name, dot_token.line, dot_token.column)
                elif self.peek().type == TokenType.SYMBOL and self.peek().value == '[':
                    bracket_token = self.consume() # [
                    if self.peek().type == TokenType.SYMBOL and self.peek().value == ']':
                         self.consume() # ]
                         expr = ArrayAccessExpr(expr, None, bracket_token.line, bracket_token.column)
                    else:
                         index = self.parse_expression()
                         self.consume(TokenType.SYMBOL, ']')
                         expr = ArrayAccessExpr(expr, index, bracket_token.line, bracket_token.column)
                else:
                    break
            
            # Post-check: If it was ONLY an identifier and it is an Enum name?
            # But the loop might have consumed dots.
            # If it's an Enum, it should be EnumName.Value.
            # If we parsed `EnumName` as VarRefExpr, then saw `.Value` and made FieldAccessExpr...
            # We might need to correct it or handle it at the start.
            
            return expr
        raise RuntimeError(f"Unexpected token in expression: {token}")

    def parse_struct_literal(self):
        start_token = self.consume(TokenType.SYMBOL, '{')
        field_values = []
        while self.peek().type != TokenType.SYMBOL or self.peek().value != '}':
            field_name = self.consume(TokenType.IDENTIFIER).value
            self.consume(TokenType.SYMBOL, ':')
            field_type = self.consume().value
            while True:
                if self.peek().type == TokenType.SYMBOL and self.peek().value == '*':
                    self.consume()
                    field_type += '*'
                elif self.peek().type == TokenType.SYMBOL and self.peek().value == '[':
                    self.consume() # [
                    self.consume(TokenType.SYMBOL, ']') # ]
                    field_type += '[]'
                else:
                    break
            field_expr = self.parse_expression()
            field_values.append((field_name, field_type, field_expr))
            if self.peek().type == TokenType.SYMBOL and self.peek().value == ',':
                self.consume()
        self.consume(TokenType.SYMBOL, '}')
        return StructLiteralExpr(None, field_values, start_token.line, start_token.column)  # struct_type será inferido do contexto

    def parse_array_literal(self):
        start_token = self.consume(TokenType.SYMBOL, '{')
        values = []
        while self.peek().type != TokenType.SYMBOL or self.peek().value != '}':
            values.append(self.parse_expression())
            if self.peek().type == TokenType.SYMBOL and self.peek().value == ',':
                self.consume()
        self.consume(TokenType.SYMBOL, '}')
        return ArrayLiteralExpr(values, start_token.line, start_token.column)

    def parse_call_stmt(self):
        # We already consumed 'call' in some cases, but here we expect it as a statement
        token = self.consume(TokenType.KEYWORD, 'call')
        name = self.consume(TokenType.IDENTIFIER).value
        self.consume(TokenType.SYMBOL, '(')
        args = []
        while self.peek().type != TokenType.SYMBOL or self.peek().value != ')':
            # Argument type (base keyword or identifier; pointer/deref handled in expression)
            type_token = self.peek()
            if type_token.type not in [TokenType.KEYWORD, TokenType.IDENTIFIER]:
                raise RuntimeError(f"Expected argument type, got {type_token.type} at line {type_token.line}")
            arg_type = self.consume().value
            while self.peek().type == TokenType.SYMBOL and self.peek().value == '*':
                self.consume()
                arg_type += '*'
            if self.peek().type == TokenType.SYMBOL and self.peek().value == '[':
                 self.consume()
                 self.consume(TokenType.SYMBOL, ']')
                 arg_type += '[]'
            arg_expr = self.parse_expression()
            
            # Auto-correct type if expression is array access (arr[]) and type is base (int32)
            if isinstance(arg_expr, ArrayAccessExpr) and arg_expr.index is None:
                 if not arg_type.endswith('[]') and not arg_type.endswith('*'):
                      arg_type += '[]'
            
            args.append((arg_type, arg_expr))
            if self.peek().type == TokenType.SYMBOL and self.peek().value == ',':
                self.consume()
        self.consume(TokenType.SYMBOL, ')')
        self.consume(TokenType.SYMBOL, ';')
        return CallExpr(name, args, token.line, token.column)
