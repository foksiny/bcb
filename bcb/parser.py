import os
from .lexer import tokenize, TokenType

class ASTNode:
    def __init__(self, line=0, column=0, source_file=None):
        self.line = line
        self.column = column
        self.source_file = source_file  # Track which file this node originated from

class Attribute(ASTNode):
    def __init__(self, name, args, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.name = name  # Attribute name (e.g., "NoWarning", "SonOf")
        self.args = args  # List of arguments (strings, identifiers, numbers)

class Program(ASTNode):
    def __init__(self, outtype, data_block, declarations, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.outtype = outtype
        self.data_block = data_block
        self.declarations = declarations

class StructDef(ASTNode):
    def __init__(self, name, fields, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.name = name
        self.fields = fields

class EnumDef(ASTNode):
    def __init__(self, name, values, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.name = name
        self.values = values

class DataBlock(ASTNode):
    def __init__(self, entries, structs=None, enums=None, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.entries = entries
        self.structs = structs or []
        self.enums = enums or []

class FunctionDecl(ASTNode):
    def __init__(self, name, params, return_type, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.name = name
        self.params = params
        self.return_type = return_type

class FunctionDef(ASTNode):
    def __init__(self, name, params, return_type, body, is_exported, attributes=None, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.name = name
        self.params = params
        self.return_type = return_type
        self.body = body
        self.is_exported = is_exported
        self.attributes = attributes or []  # List of Attribute nodes

class GlobalVarDecl(ASTNode):
    def __init__(self, type_name, name, expr, is_array=False, array_size=None, attributes=None, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.type_name = type_name
        self.name = name
        self.expr = expr
        self.is_array = is_array
        self.array_size = array_size
        self.attributes = attributes or []  # List of Attribute nodes

class AsmImport(ASTNode):
    """Represents an imported assembly file."""
    def __init__(self, path, asm_code, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.path = path  # Original import path
        self.asm_code = asm_code  # Raw assembly code from the file

class CallExpr(ASTNode):
    def __init__(self, name, args, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.name = name
        self.args = args

class ReturnStmt(ASTNode):
    def __init__(self, return_type, expr, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.return_type = return_type
        self.expr = expr

class VarDeclStmt(ASTNode):
    def __init__(self, type_name, name, expr, is_array=False, array_size=None, attributes=None, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.type_name = type_name
        self.name = name
        self.expr = expr
        self.is_array = is_array
        self.array_size = array_size
        self.attributes = attributes or []  # List of Attribute nodes

class BinaryExpr(ASTNode):
    def __init__(self, left, op, right, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.left = left
        self.op = op
        self.right = right

class UnaryExpr(ASTNode):
    def __init__(self, op, expr, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.op = op
        self.expr = expr

class LiteralExpr(ASTNode):
    def __init__(self, value, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.value = value

class VarRefExpr(ASTNode):
    def __init__(self, name, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.name = name

class TypeCastExpr(ASTNode):
    def __init__(self, target_type, expr, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.target_type = target_type
        self.expr = expr

class StructLiteralExpr(ASTNode):
    def __init__(self, struct_type, field_values, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.struct_type = struct_type
        self.field_values = field_values

class FieldAccessExpr(ASTNode):
    def __init__(self, obj, field_name, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.obj = obj
        self.field_name = field_name

class EnumValueExpr(ASTNode):
    def __init__(self, enum_name, value_name, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.enum_name = enum_name
        self.value_name = value_name

class IfStmt(ASTNode):
    def __init__(self, conditions_and_bodies, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.conditions_and_bodies = conditions_and_bodies

class WhileStmt(ASTNode):
    def __init__(self, condition, body, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.condition = condition
        self.body = body

class VarAssignStmt(ASTNode):
    def __init__(self, type_name, name, expr, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.type_name = type_name
        self.name = name
        self.expr = expr

class FieldAssignStmt(ASTNode):
    def __init__(self, type_name, var_name, field_name, expr, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.type_name = type_name
        self.var_name = var_name
        self.field_name = field_name
        self.expr = expr

class ArrayAccessExpr(ASTNode):
    def __init__(self, arr, index, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.arr = arr
        self.index = index

class ArrayLiteralExpr(ASTNode):
    def __init__(self, values, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.values = values

class ArrayAssignStmt(ASTNode):
    def __init__(self, type_name, arr_name, index, expr, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.type_name = type_name
        self.arr_name = arr_name
        self.index = index
        self.expr = expr

class LengthExpr(ASTNode):
    def __init__(self, expr, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.expr = expr

class GetTypeExpr(ASTNode):
    def __init__(self, expr, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.expr = expr

class ArgsAccessExpr(ASTNode):
    def __init__(self, name, index, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.name = name
        self.index = index

class LabelDef(ASTNode):
    def __init__(self, name, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.name = name

class JmpStmt(ASTNode):
    def __init__(self, target, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.target = target

class IfnStmt(ASTNode):
    def __init__(self, condition, target, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.condition = condition
        self.target = target

class CmpTStmt(ASTNode):
    def __init__(self, condition, target, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.condition = condition
        self.target = target

class PushStmt(ASTNode):
    def __init__(self, type_name, expr, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.type_name = type_name
        self.expr = expr

class PopStmt(ASTNode):
    def __init__(self, type_name, var_name, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.type_name = type_name
        self.var_name = var_name

class SwapStmt(ASTNode):
    def __init__(self, type_name, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.type_name = type_name

class DupStmt(ASTNode):
    def __init__(self, type_name, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.type_name = type_name

class NoValueExpr(ASTNode):
    def __init__(self, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)

class AddIndexStmt(ASTNode):
    """Add an index to a dynamic array (add_i array_name;)"""
    def __init__(self, arr_name, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.arr_name = arr_name

class RemoveIndexStmt(ASTNode):
    """Remove an index from a dynamic array (rem_i array_name;)"""
    def __init__(self, arr_name, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.arr_name = arr_name

class HereExpr(ASTNode):
    """Built-in macro that returns the current source location as a string (file:line:column)"""
    def __init__(self, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)

class Attribute(ASTNode):
    """Represents an attribute like #NoWarning("unused function") or #SonOf(math)"""
    def __init__(self, name, args, line=0, column=0, source_file=None):
        super().__init__(line, column, source_file)
        self.name = name  # e.g., "NoWarning", "SonOf"
        self.args = args  # list of (arg_type, arg_value) tuples

class Parser:
    def __init__(self, tokens, base_dir=".", imported_files=None, source_file=None):
        self.tokens = tokens
        self.pos = 0
        self.enum_names = set()  # Track known enum names
        self.struct_names = set() # Track known struct names
        self.macros = {} # Track known macros
        self.base_dir = base_dir
        self.imported_files = imported_files if imported_files is not None else set()
        self.source_file = source_file  # Track the source file being parsed

    def parse_type(self):
        # Base type (keyword, identifier, etc.)
        type_token = self.peek()
        if type_token.type not in [TokenType.KEYWORD, TokenType.IDENTIFIER]:
            raise RuntimeError(f"Expected type name, got {type_token.type} at line {type_token.line}")
        
        type_name = self.consume().value
        
        # Optional pointer suffix(es) or array brackets
        while True:
            if self.peek().type == TokenType.SYMBOL and self.peek().value == '*':
                self.consume()
                type_name += '*'
            elif self.peek().type == TokenType.SYMBOL and self.peek().value == '[':
                self.consume() # [
                if self.peek().type == TokenType.SYMBOL and self.peek().value == ']':
                    self.consume() # ]
                    type_name += '[]'
                else:
                    # Fixed size e.g. int32[10] or int32[arraysize]
                    # Accept either a number or an identifier (variable name) for array size
                    if self.peek().type == TokenType.NUMBER:
                        size_token = self.consume(TokenType.NUMBER)
                        type_name += f"[{size_token.value}]"
                    elif self.peek().type == TokenType.IDENTIFIER:
                        # Variable name as array size
                        size_token = self.consume(TokenType.IDENTIFIER)
                        type_name += f"[{size_token.value}]"
                    else:
                        raise RuntimeError(f"Expected number or identifier for array size at line {self.peek().line}")
                    self.consume(TokenType.SYMBOL, ']')
            else:
                break
        return type_name

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

    def parse_attributes(self):
        """Parse attributes like #NoWarning("unused function") or #SonOf(math)
        
        Multiple attributes can be specified using :: as separator:
        #NoWarning("unused function")::#SonOf(myfunc)
        """
        attributes = []
        while self.peek().type == TokenType.SYMBOL and self.peek().value == '#':
            start_token = self.consume()  # #
            attr_name = self.consume(TokenType.IDENTIFIER).value  # Attribute name
            args = []
            
            # Check for arguments in parentheses
            if self.peek().type == TokenType.SYMBOL and self.peek().value == '(':
                self.consume()  # (
                while not (self.peek().type == TokenType.SYMBOL and self.peek().value == ')'):
                    arg_token = self.peek()
                    if arg_token.type == TokenType.STRING:
                        args.append(('string', self.consume().value))
                    elif arg_token.type == TokenType.NUMBER:
                        args.append(('number', self.consume().value))
                    elif arg_token.type == TokenType.IDENTIFIER:
                        args.append(('identifier', self.consume().value))
                    else:
                        raise RuntimeError(f"Unexpected attribute argument type {arg_token.type} at line {arg_token.line}")
                    
                    # Optional comma
                    if self.peek().type == TokenType.SYMBOL and self.peek().value == ',':
                        self.consume()
                self.consume(TokenType.SYMBOL, ')')  # )
            
            attributes.append(Attribute(attr_name, args, start_token.line, start_token.column, self.source_file))
            
            # Check for :: separator to continue parsing more attributes
            if self.peek().type == TokenType.SYMBOL and self.peek().value == '::':
                self.consume()  # ::
                # Continue to next attribute (must be followed by #)
                if not (self.peek().type == TokenType.SYMBOL and self.peek().value == '#'):
                    raise RuntimeError(f"Expected attribute after '::' at line {self.peek().line}")
        return attributes

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
            elif token.type == TokenType.SYMBOL and token.value == '#':
                # Parse attributes and attach to the following declaration
                attrs = self.parse_attributes()
                # Peek at next token to determine what follows
                next_token = self.peek()
                if next_token.type == TokenType.KEYWORD:
                    if next_token.value == 'export':
                        declarations.append(self.parse_function_def(True, attrs))
                    elif next_token.value == 'pub':
                        declarations.append(self.parse_global_var(attrs))
                    else:
                        raise RuntimeError(f"Unexpected keyword {next_token.value} after attribute at line {next_token.line}")
                elif next_token.type == TokenType.IDENTIFIER:
                    # Function definition without export
                    if self.peek(1).type == TokenType.SYMBOL and self.peek(1).value == '(':
                        declarations.append(self.parse_function_def(False, attrs))
                    else:
                        raise RuntimeError(f"Expected function definition after attribute at line {next_token.line}")
                else:
                    raise RuntimeError(f"Expected declaration after attribute at line {token.line}")
            elif token.type == TokenType.KEYWORD:
                if token.value == 'data':
                    data_block = self.parse_data_block()
                elif token.value == 'macro':
                    self.parse_macro_def()
                elif token.value == 'define':
                    declarations.append(self.parse_function_decl())
                elif token.value == 'export':
                    declarations.append(self.parse_function_def(True))
                elif token.value == 'pub':
                    declarations.append(self.parse_global_var())
                elif token.value == 'import':
                    self.consume() # import
                    import_path = self.consume(TokenType.STRING).value
                    
                    # Check for 'asmf' marker for assembly file imports
                    is_asm_import = False
                    if self.peek().type == TokenType.IDENTIFIER and self.peek().value == 'asmf':
                        self.consume()  # asmf
                        is_asm_import = True
                    
                    self.consume(TokenType.SYMBOL, ';')
                    
                    full_path = os.path.abspath(os.path.join(self.base_dir, import_path))
                    if full_path not in self.imported_files:
                        self.imported_files.add(full_path)
                        if os.path.exists(full_path):
                            with open(full_path, 'r') as f:
                                import_code = f.read()
                            
                            # Handle assembly file imports
                            if is_asm_import or import_path.endswith('.s'):
                                asm_import = AsmImport(import_path, import_code, token.line, token.column, self.source_file)
                                declarations.append(asm_import)
                            else:
                                # Regular BCB file import
                                import_tokens = tokenize(import_code)
                                import_parser = Parser(import_tokens, os.path.dirname(full_path), self.imported_files, full_path)
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
                                    self.macros.update(import_parser.macros)
                                
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

        return Program(outtype, data_block, declarations, 1, 1, self.source_file)

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
        return DataBlock(entries, structs, enums, start_token.line, start_token.column, self.source_file)

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
        return StructDef(name, fields, start_token.line, start_token.column, self.source_file)

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
        return EnumDef(name, values, start_token.line, start_token.column, self.source_file)

    def parse_global_var(self, attributes=None):
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
            # Accept either a number or an identifier (variable name) for array size
            if self.peek().type == TokenType.NUMBER:
                size_token = self.consume(TokenType.NUMBER)
                array_size = int(size_token.value)
            elif self.peek().type == TokenType.IDENTIFIER:
                # Variable name as array size
                size_token = self.consume(TokenType.IDENTIFIER)
                array_size = size_token.value  # Store as string (variable name)
            else:
                raise RuntimeError(f"Expected number or identifier for array size at line {self.peek().line}")
            self.consume(TokenType.SYMBOL, ']')
            is_array = True
        
        self.consume(TokenType.SYMBOL, '=')
        expr = self.parse_expression()
        self.consume(TokenType.SYMBOL, ';')
        return GlobalVarDecl(type_name, name, expr, is_array, array_size, attributes, start_token.line, start_token.column, self.source_file)

    def parse_function_decl(self):
        start_token = self.consume(TokenType.KEYWORD, 'define')
        name = self.consume(TokenType.IDENTIFIER).value
        self.consume(TokenType.SYMBOL, '(')
        params = self.parse_params()
        self.consume(TokenType.SYMBOL, ')')
        self.consume(TokenType.SYMBOL, '-')
        self.consume(TokenType.SYMBOL, '>')
        
        return_type = self.parse_type()
        self.consume(TokenType.SYMBOL, ';')
        return FunctionDecl(name, params, return_type, start_token.line, start_token.column, self.source_file)

    def parse_function_def(self, is_exported, attributes=None):
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
        
        return_type = self.parse_type()
        self.consume(TokenType.SYMBOL, '{')
        body = []
        while self.peek().type != TokenType.SYMBOL or self.peek().value != '}':
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
        self.consume(TokenType.SYMBOL, '}')
        return FunctionDef(name, params, return_type, body, is_exported, attributes, start_token.line, start_token.column, self.source_file)

    def parse_macro_def(self):
        self.consume(TokenType.KEYWORD, 'macro')
        name = self.consume(TokenType.IDENTIFIER).value
        self.consume(TokenType.SYMBOL, '(')
        
        # Parse params (name: type, ...) but we only care about names for substitution
        param_names = []
        while self.peek().type != TokenType.SYMBOL or self.peek().value != ')':
            p_name = self.consume(TokenType.IDENTIFIER).value
            self.consume(TokenType.SYMBOL, ':')
            # Skip type
            type_token = self.peek()
            if type_token.type not in [TokenType.KEYWORD, TokenType.IDENTIFIER]:
                 raise RuntimeError(f"Expected type name, got {type_token.type} at line {type_token.line}")
            self.consume() # base type
            while True:
                if self.peek().type == TokenType.SYMBOL and self.peek().value == '*':
                    self.consume()
                elif self.peek().type == TokenType.SYMBOL and self.peek().value == '[':
                    self.consume()
                    self.consume(TokenType.SYMBOL, ']')
                # Handle ...args style varargs for macros? Example uses ...args in define but specific known params in macro
                # Given 'macro add()', empty params. 'macro print(msg: string)', typed params. 
                # Assuming standard type parsing logic applies but result is discard.
                else:
                    break
            
            param_names.append(p_name)
            
            if self.peek().type == TokenType.SYMBOL and self.peek().value == ',':
                self.consume()
        self.consume(TokenType.SYMBOL, ')')
        
        self.consume(TokenType.SYMBOL, '{')
        
        # Capture body tokens until matching }
        body_tokens = []
        brace_count = 1
        while brace_count > 0:
            token = self.consume()
            if token.type == TokenType.SYMBOL:
                if token.value == '{':
                    brace_count += 1
                elif token.value == '}':
                    brace_count -= 1
            
            if brace_count > 0:
                body_tokens.append(token)
        
        # Store macro
        self.macros[name] = (param_names, body_tokens)

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
                # Base type
                type_name = self.parse_type()
            params.append((name, type_name))
            if self.peek().type == TokenType.SYMBOL and self.peek().value == ',':
                self.consume()
        return params

    def parse_statement(self):
        token = self.peek()
        
        # Allow empty statement
        if token.type == TokenType.SYMBOL and token.value == ';':
             self.consume()
             return None

        # Handle attributes at statement level
        if token.type == TokenType.SYMBOL and token.value == '#':
            attrs = self.parse_attributes()
            # Attributes at statement level should be followed by a variable declaration
            next_token = self.peek()
            if next_token.type == TokenType.KEYWORD and next_token.value in ['int8', 'int16', 'int32', 'int64', 'float32', 'float64', 'string', 'char']:
                # Variable declaration with attributes
                type_name = self.parse_type()
                name = self.consume(TokenType.IDENTIFIER).value
                
                is_array = False
                array_size = None
                if self.peek().type == TokenType.SYMBOL and self.peek().value == '[':
                    self.consume() # [
                    # Accept either a number or an identifier (variable name) for array size
                    if self.peek().type == TokenType.NUMBER:
                        size_token = self.consume(TokenType.NUMBER)
                        array_size = int(size_token.value)
                    elif self.peek().type == TokenType.IDENTIFIER:
                        # Variable name as array size
                        size_token = self.consume(TokenType.IDENTIFIER)
                        array_size = size_token.value  # Store as string (variable name)
                    else:
                        raise RuntimeError(f"Expected number or identifier for array size at line {self.peek().line}")
                    self.consume(TokenType.SYMBOL, ']')
                    is_array = True
                
                if '[' in type_name and not type_name.endswith('[]'):
                    open_b = type_name.rfind('[')
                    size_str = type_name[open_b+1:-1]
                    array_size = int(size_str)
                    type_name = type_name[:open_b]
                    is_array = True
                
                self.consume(TokenType.SYMBOL, '=')
                expr = self.parse_expression()
                self.consume(TokenType.SYMBOL, ';')
                return VarDeclStmt(type_name, name, expr, is_array, array_size, attrs, token.line, token.column, self.source_file)
            elif next_token.type == TokenType.IDENTIFIER:
                # Struct variable declaration with attributes
                struct_type = self.consume(TokenType.IDENTIFIER).value
                var_name = self.consume(TokenType.IDENTIFIER).value
                
                is_array = False
                array_size = None
                if self.peek().type == TokenType.SYMBOL and self.peek().value == '[':
                    self.consume() # [
                    # Accept either a number or an identifier (variable name) for array size
                    if self.peek().type == TokenType.NUMBER:
                        size_token = self.consume(TokenType.NUMBER)
                        array_size = int(size_token.value)
                    elif self.peek().type == TokenType.IDENTIFIER:
                        # Variable name as array size
                        size_token = self.consume(TokenType.IDENTIFIER)
                        array_size = size_token.value  # Store as string (variable name)
                    else:
                        raise RuntimeError(f"Expected number or identifier for array size at line {self.peek().line}")
                    self.consume(TokenType.SYMBOL, ']')
                    is_array = True

                self.consume(TokenType.SYMBOL, '=')
                expr = self.parse_expression()
                self.consume(TokenType.SYMBOL, ';')
                return VarDeclStmt(struct_type, var_name, expr, is_array, array_size, attrs, token.line, token.column, self.source_file)
            else:
                raise RuntimeError(f"Expected variable declaration after attribute at line {next_token.line}")

        if token.type == TokenType.IDENTIFIER and token.value in self.macros:
             self.expand_macro_invocation()
             
             # Check if we reached end of block or control structure after expansion
             next_tok = self.peek()
             if next_tok.type == TokenType.SYMBOL and next_tok.value == '}':
                 return None
             if next_tok.type == TokenType.KEYWORD and next_tok.value in ['$elseif', '$else', '$endif', '$endwhile']:
                 return None
                 
             return self.parse_statement()

        if token.type == TokenType.KEYWORD:
            if token.value == 'call':
                return self.parse_call_stmt()
            elif token.value == 'return':
                self.consume()
                # Check for void return
                if self.peek().type == TokenType.KEYWORD and self.peek().value == 'void':
                    self.consume()
                    self.consume(TokenType.SYMBOL, ';')
                    return ReturnStmt('void', None, token.line, token.column, self.source_file)
                
                ret_type = self.parse_type()
                expr = self.parse_expression()
                self.consume(TokenType.SYMBOL, ';')
                # Use 'return' token loc? Yes, that was consumed earlier.
                # Accessing token from start of statement logic
                return ReturnStmt(ret_type, expr, token.line, token.column, self.source_file)
            elif token.value == '$if':
                return self.parse_if_stmt()
            elif token.value == '$while':
                return self.parse_while_stmt()
            elif token.value == 'md':
                self.consume()  # md
                type_name = self.parse_type()
                name = self.consume(TokenType.IDENTIFIER).value
                
                # Check if it's a dotted name (could be field assignment or SonOf variable)
                if self.peek().type == TokenType.SYMBOL and self.peek().value == '.':
                    self.consume()  # .
                    second_part = self.consume(TokenType.IDENTIFIER).value
                    
                    # Check if there's another dot (math.x.field would be field access on SonOf var)
                    # or if this is a field assignment on a local struct
                    # For now, we'll treat it as a dotted variable name (SonOf)
                    # Store the full dotted name
                    dotted_name = f"{name}.{second_part}"
                    
                    # Check if it's array access on the dotted name
                    if self.peek().type == TokenType.SYMBOL and self.peek().value == '[':
                        self.consume() # [
                        index = self.parse_expression()
                        self.consume(TokenType.SYMBOL, ']') # ]
                        self.consume(TokenType.SYMBOL, '=')
                        expr = self.parse_expression()
                        self.consume(TokenType.SYMBOL, ';')
                        return ArrayAssignStmt(type_name, dotted_name, index, expr, token.line, token.column, self.source_file)
                    
                    self.consume(TokenType.SYMBOL, '=')
                    expr = self.parse_expression()
                    self.consume(TokenType.SYMBOL, ';')
                    return VarAssignStmt(type_name, dotted_name, expr, token.line, token.column, self.source_file)
                elif self.peek().type == TokenType.SYMBOL and self.peek().value == '[':
                    self.consume() # [
                    index = self.parse_expression()
                    self.consume(TokenType.SYMBOL, ']') # ]
                    self.consume(TokenType.SYMBOL, '=')
                    expr = self.parse_expression()
                    self.consume(TokenType.SYMBOL, ';')
                    return ArrayAssignStmt(type_name, name, index, expr, token.line, token.column, self.source_file)
                else:
                    self.consume(TokenType.SYMBOL, '=')
                    expr = self.parse_expression()
                    self.consume(TokenType.SYMBOL, ';')
                    return VarAssignStmt(type_name, name, expr, token.line, token.column, self.source_file)
            elif token.value == 'jmp':
                self.consume()
                target = self.consume(TokenType.LABEL).value
                self.consume(TokenType.SYMBOL, ';')
                return JmpStmt(target, token.line, token.column, self.source_file)
            elif token.value == 'ifn':
                self.consume()
                cond = self.consume(TokenType.IDENTIFIER).value
                self.consume(TokenType.SYMBOL, ',')
                target = self.consume(TokenType.LABEL).value
                self.consume(TokenType.SYMBOL, ';')
                return IfnStmt(cond, target, token.line, token.column, self.source_file)
            elif token.value == 'cmp_t':
                self.consume()
                cond = self.consume(TokenType.IDENTIFIER).value
                self.consume(TokenType.SYMBOL, ',')
                target = self.consume(TokenType.LABEL).value
                self.consume(TokenType.SYMBOL, ';')
                return CmpTStmt(cond, target, token.line, token.column, self.source_file)
            elif token.value == 'push':
                self.consume()
                type_name = self.parse_type()
                expr = self.parse_expression()
                self.consume(TokenType.SYMBOL, ';')
                return PushStmt(type_name, expr, token.line, token.column, self.source_file)
            elif token.value == 'pop':
                self.consume()
                type_name = self.parse_type()
                var_name = self.consume(TokenType.IDENTIFIER).value
                self.consume(TokenType.SYMBOL, ';')
                return PopStmt(type_name, var_name, token.line, token.column, self.source_file)
            elif token.value == 'swap':
                self.consume()
                type_name = self.parse_type()
                self.consume(TokenType.SYMBOL, ';')
                return SwapStmt(type_name, token.line, token.column, self.source_file)
            elif token.value == 'dup':
                self.consume()
                type_name = self.parse_type()
                self.consume(TokenType.SYMBOL, ';')
                return DupStmt(type_name, token.line, token.column, self.source_file)
            elif token.value == 'add_i':
                self.consume()
                arr_name = self.consume(TokenType.IDENTIFIER).value
                self.consume(TokenType.SYMBOL, ';')
                return AddIndexStmt(arr_name, token.line, token.column, self.source_file)
            elif token.value == 'rem_i':
                self.consume()
                arr_name = self.consume(TokenType.IDENTIFIER).value
                self.consume(TokenType.SYMBOL, ';')
                return RemoveIndexStmt(arr_name, token.line, token.column, self.source_file)
            elif token.value in ['int8', 'int16', 'int32', 'int64', 'float32', 'float64', 'string', 'char']:
                # Variable declaration
                type_name = self.parse_type()
                name = self.consume(TokenType.IDENTIFIER).value
                
                is_array = False
                array_size = None
                # Check for fixed-size array in declaration? (e.g. int32 x[10])
                # Note: parse_type might have already consumed [10] if it was formatted like that.
                # However, the current BCB syntax seems to separate type and array bracket sometimes.
                # Let's support both.
                if self.peek().type == TokenType.SYMBOL and self.peek().value == '[':
                    # If it's something like 'int32 x[10]', parse_type got 'int32'.
                    self.consume() # [
                    # Accept either a number or an identifier (variable name) for array size
                    if self.peek().type == TokenType.NUMBER:
                        size_token = self.consume(TokenType.NUMBER)
                        array_size = int(size_token.value)
                    elif self.peek().type == TokenType.IDENTIFIER:
                        # Variable name as array size
                        size_token = self.consume(TokenType.IDENTIFIER)
                        array_size = size_token.value  # Store as string (variable name)
                    else:
                        raise RuntimeError(f"Expected number or identifier for array size at line {self.peek().line}")
                    self.consume(TokenType.SYMBOL, ']')
                    is_array = True
                
                # If parse_type got 'int32[10]', then is_array is already somewhat encoded.
                # CodeGen expects is_array and array_size for stack allocation.
                if '[' in type_name and not type_name.endswith('[]'):
                    # Extract size from type_name
                    open_b = type_name.rfind('[')
                    size_str = type_name[open_b+1:-1]
                    array_size = int(size_str)
                    type_name = type_name[:open_b]
                    is_array = True
                
                self.consume(TokenType.SYMBOL, '=')
                expr = self.parse_expression()
                self.consume(TokenType.SYMBOL, ';')
                return VarDeclStmt(type_name, name, expr, is_array, array_size, None, token.line, token.column, self.source_file)
        elif token.type == TokenType.LABEL:
            label_name = self.consume().value
            self.consume(TokenType.SYMBOL, ':')
            return LabelDef(label_name, token.line, token.column, self.source_file)
        elif token.type == TokenType.IDENTIFIER:
            # Could be struct variable declaration: StructName varName = { ... };
            struct_type = self.consume(TokenType.IDENTIFIER).value
            var_name = self.consume(TokenType.IDENTIFIER).value
            
            is_array = False
            array_size = None
            if self.peek().type == TokenType.SYMBOL and self.peek().value == '[':
                self.consume() # [
                # Accept either a number or an identifier (variable name) for array size
                if self.peek().type == TokenType.NUMBER:
                    size_token = self.consume(TokenType.NUMBER)
                    array_size = int(size_token.value)
                elif self.peek().type == TokenType.IDENTIFIER:
                    # Variable name as array size
                    size_token = self.consume(TokenType.IDENTIFIER)
                    array_size = size_token.value  # Store as string (variable name)
                else:
                    raise RuntimeError(f"Expected number or identifier for array size at line {self.peek().line}")
                self.consume(TokenType.SYMBOL, ']')
                is_array = True

            self.consume(TokenType.SYMBOL, '=')
            expr = self.parse_expression()
            self.consume(TokenType.SYMBOL, ';')
            return VarDeclStmt(struct_type, var_name, expr, is_array, array_size, None, token.line, token.column, self.source_file)
        raise RuntimeError(f"Unknown statement {token}")

    def parse_expression(self, min_prec=1):
        token = self.peek()

        # Handle Casts and Unary Operators (Prefix)
        if token.type == TokenType.SYMBOL and token.value == '-' and self.peek(1).type == TokenType.NUMBER:
             self.consume() # -
             num_token = self.consume()
             return LiteralExpr(-num_token.value, token.line, token.column, self.source_file)
             
        # Support both primitive keywords and custom struct/enum types as casts
        is_type = False
        if token.type == TokenType.KEYWORD and token.value in ['int8', 'int16', 'int32', 'int64', 'float32', 'float64', 'string', 'char', 'void']:
             is_type = True
        elif token.type == TokenType.IDENTIFIER and (token.value in self.struct_names or token.value in self.enum_names):
             # Only treat as type cast if NOT followed by '.' (Enum.Value access)
             if self.peek(1).type != TokenType.SYMBOL or self.peek(1).value != '.':
                 is_type = True
        
        if is_type:
            type_name = self.parse_type()

            if self.peek().type == TokenType.SYMBOL and self.peek().value == '(':
                # type(expr)
                self.consume(TokenType.SYMBOL, '(')
                expr = self.parse_expression(1)
                self.consume(TokenType.SYMBOL, ')')
                lhs = TypeCastExpr(type_name, expr, token.line, token.column, self.source_file)
            else:
                # type expr
                rhs = self.parse_expression(11) # High precedence
                lhs = TypeCastExpr(type_name, rhs, token.line, token.column, self.source_file)
            return self.parse_op_continuation(lhs, min_prec)
            
        elif token.type == TokenType.SYMBOL and token.value in ['~', '&', '*']:
            op = self.consume().value
            rhs = self.parse_expression(11) # High precedence
            lhs = UnaryExpr(op, rhs, token.line, token.column, self.source_file)
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
            lhs = BinaryExpr(lhs, op, rhs, token.line, token.column, self.source_file)
            
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
        return IfStmt(conditions_and_bodies, start_token.line, start_token.column, self.source_file)

    def parse_while_stmt(self):
        start_token = self.consume(TokenType.KEYWORD, '$while')
        condition = self.parse_expression()
        body = []
        while self.peek().type != TokenType.KEYWORD or self.peek().value != '$endwhile':
            body.append(self.parse_statement())
        self.consume(TokenType.KEYWORD, '$endwhile')
        return WhileStmt(condition, body, start_token.line, start_token.column, self.source_file)

    def parse_primary(self):
        token = self.peek()
        if token.type == TokenType.IDENTIFIER and token.value in self.macros:
             self.expand_macro_invocation()
             # We need to re-parse from the start of the injected tokens.
             # parse_expression calls parse_primary.
             # If we call parse_expression() here, it parses the *full* expression starting with the macro expansion.
             # This is correct.
             return self.parse_expression() # Recurse

        if token.type == TokenType.NUMBER:
            return LiteralExpr(self.consume().value, token.line, token.column, self.source_file)
        elif token.type == TokenType.STRING:
            return LiteralExpr(self.consume().value, token.line, token.column, self.source_file)
        elif token.type == TokenType.CHAR:
            return LiteralExpr(self.consume().value, token.line, token.column, self.source_file)
        elif token.type == TokenType.IDENTIFIER and token.value == 'length':
            self.consume() # length
            self.consume(TokenType.SYMBOL, '(')
            expr = self.parse_expression()
            self.consume(TokenType.SYMBOL, ')')
            return LengthExpr(expr, token.line, token.column, self.source_file)
        elif token.type == TokenType.IDENTIFIER and token.value == 'gettype':
            self.consume() # gettype
            self.consume(TokenType.SYMBOL, '(')
            expr = self.parse_expression()
            self.consume(TokenType.SYMBOL, ')')
            return GetTypeExpr(expr, token.line, token.column, self.source_file)
        elif token.type == TokenType.IDENTIFIER and token.value == 'here':
            # Built-in macro that returns current source location
            start_token = self.consume() # here
            self.consume(TokenType.SYMBOL, '(')
            self.consume(TokenType.SYMBOL, ')')
            return HereExpr(start_token.line, start_token.column, self.source_file)
        elif token.type == TokenType.KEYWORD and token.value == 'call':
            self.consume()  # call
            name = self.consume(TokenType.IDENTIFIER).value
            
            # Check for dotted name (e.g., math.add for SonOf attribute)
            while self.peek().type == TokenType.SYMBOL and self.peek().value == '.':
                self.consume()  # consume '.'
                next_part = self.consume(TokenType.IDENTIFIER).value
                name = f"{name}.{next_part}"
            
            self.consume(TokenType.SYMBOL, '(')
            args = []
            while self.peek().type != TokenType.SYMBOL or self.peek().value != ')':
                arg_type = self.parse_type()
                arg_expr = self.parse_expression()
                args.append((arg_type, arg_expr))
                if self.peek().type == TokenType.SYMBOL and self.peek().value == ',':
                    self.consume()
            self.consume(TokenType.SYMBOL, ')')
            return CallExpr(name, args, token.line, token.column, self.source_file)
        elif token.type == TokenType.KEYWORD and token.value == 'no_value':
            self.consume()
            return NoValueExpr(token.line, token.column, self.source_file)
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
                return EnumValueExpr(name, val_name, token.line, token.column, self.source_file)

            expr = VarRefExpr(name, token.line, token.column, self.source_file)
            
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
                    expr = FieldAccessExpr(expr, field_name, dot_token.line, dot_token.column, self.source_file)
                elif self.peek().type == TokenType.SYMBOL and self.peek().value == '[':
                    bracket_token = self.consume() # [
                    if self.peek().type == TokenType.SYMBOL and self.peek().value == ']':
                         self.consume() # ]
                         expr = ArrayAccessExpr(expr, None, bracket_token.line, bracket_token.column, self.source_file)
                    else:
                         index = self.parse_expression()
                         self.consume(TokenType.SYMBOL, ']')
                         expr = ArrayAccessExpr(expr, index, bracket_token.line, bracket_token.column, self.source_file)
                elif self.peek().type == TokenType.SYMBOL and self.peek().value == '(':
                    # Args access: myargs(y)
                    paren_token = self.consume() # (
                    index = self.parse_expression()
                    self.consume(TokenType.SYMBOL, ')')
                    # expr should be a VarRefExpr for the variadic param name
                    if not isinstance(expr, VarRefExpr):
                         # It might be a regular call, but in BCB calls use 'call' keyword.
                         # This syntax is specifically for variadic arg access.
                         pass
                    expr = ArgsAccessExpr(expr.name if isinstance(expr, VarRefExpr) else expr, index, paren_token.line, paren_token.column, self.source_file)
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
        return StructLiteralExpr(None, field_values, start_token.line, start_token.column, self.source_file)  # struct_type será inferido do contexto

    def parse_array_literal(self):
        start_token = self.consume(TokenType.SYMBOL, '{')
        values = []
        while self.peek().type != TokenType.SYMBOL or self.peek().value != '}':
            values.append(self.parse_expression())
            if self.peek().type == TokenType.SYMBOL and self.peek().value == ',':
                self.consume()
        self.consume(TokenType.SYMBOL, '}')
        return ArrayLiteralExpr(values, start_token.line, start_token.column, self.source_file)

    def parse_call_stmt(self):
        # We already consumed 'call' in some cases, but here we expect it as a statement
        token = self.consume(TokenType.KEYWORD, 'call')
        name = self.consume(TokenType.IDENTIFIER).value
        
        # Check for dotted name (e.g., math.add for SonOf attribute)
        while self.peek().type == TokenType.SYMBOL and self.peek().value == '.':
            self.consume()  # consume '.'
            next_part = self.consume(TokenType.IDENTIFIER).value
            name = f"{name}.{next_part}"
        
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
        return CallExpr(name, args, token.line, token.column, self.source_file)

    def expand_macro_invocation(self):
        name = self.consume(TokenType.IDENTIFIER).value
        
        if name not in self.macros:
            # Should not happen if called correctly
             raise RuntimeError(f"Unknown macro {name}")

        param_names, body_tokens = self.macros[name]
        
        self.consume(TokenType.SYMBOL, '(')
        args_tokens = []
        
        # If not immediately closing parenthesis, parse args
        if self.peek().type != TokenType.SYMBOL or self.peek().value != ')':
            while True:
                # Capture arg tokens
                current_arg = []
                paren_count = 0
                brace_count = 0
                bracket_count = 0
                
                while True:
                    t = self.peek()
                    if t.type == TokenType.SYMBOL:
                        if t.value == '(': paren_count += 1
                        elif t.value == ')':
                             if paren_count == 0 and brace_count == 0 and bracket_count == 0: break
                             paren_count -= 1
                        elif t.value == '[': bracket_count += 1
                        elif t.value == ']': bracket_count -= 1
                        elif t.value == '{': brace_count += 1
                        elif t.value == '}': brace_count -= 1
                        elif t.value == ',':
                             if paren_count == 0 and brace_count == 0 and bracket_count == 0: break
                        
                    current_arg.append(self.consume())
                
                args_tokens.append(current_arg)
                
                if self.peek().type == TokenType.SYMBOL and self.peek().value == ',':
                    self.consume()
                else:
                    break
        
        self.consume(TokenType.SYMBOL, ')')
        
        # Check arg count
        if len(args_tokens) != len(param_names):
             raise RuntimeError(f"Macro {name} expects {len(param_names)} args, got {len(args_tokens)} at line {self.peek().line}")
             
        # Expand
        mapping = dict(zip(param_names, args_tokens))
        expanded = []
        for t in body_tokens:
            if t.type == TokenType.IDENTIFIER and t.value in mapping:
                expanded.extend(mapping[t.value])
            else:
                # We reuse the token. Note: Line numbers will point to macro definition
                expanded.append(t)
        
        # Inject at pos
        self.tokens[self.pos:self.pos] = expanded
