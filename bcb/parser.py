from .lexer import TokenType

class ASTNode:
    pass

class Program(ASTNode):
    def __init__(self, outtype, data_block, declarations):
        self.outtype = outtype
        self.data_block = data_block
        self.declarations = declarations

class StructDef(ASTNode):
    def __init__(self, name, fields):
        self.name = name
        self.fields = fields  # list of (type, name)

class EnumDef(ASTNode):
    def __init__(self, name, values):
        self.name = name
        self.values = values  # list of value names

class DataBlock(ASTNode):
    def __init__(self, entries, structs=None, enums=None):
        self.entries = entries  # list of (type, name, value)
        self.structs = structs or []  # list of StructDef
        self.enums = enums or []  # list of EnumDef

class FunctionDecl(ASTNode):
    def __init__(self, name, params, return_type):
        self.name = name
        self.params = params
        self.return_type = return_type

class FunctionDef(ASTNode):
    def __init__(self, name, params, return_type, body, is_exported):
        self.name = name
        self.params = params
        self.return_type = return_type
        self.body = body
        self.is_exported = is_exported

class CallExpr(ASTNode):
    def __init__(self, name, args):
        self.name = name
        self.args = args # list of (type, expr)

class ReturnStmt(ASTNode):
    def __init__(self, return_type, expr):
        self.return_type = return_type
        self.expr = expr

class VarDeclStmt(ASTNode):
    def __init__(self, type_name, name, expr):
        self.type_name = type_name
        self.name = name
        self.expr = expr

class BinaryExpr(ASTNode):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

class UnaryExpr(ASTNode):
    def __init__(self, op, expr):
        self.op = op
        self.expr = expr

class LiteralExpr(ASTNode):
    def __init__(self, value):
        self.value = value

class VarRefExpr(ASTNode):
    def __init__(self, name):
        self.name = name

class TypeCastExpr(ASTNode):
    def __init__(self, target_type, expr):
        self.target_type = target_type
        self.expr = expr

class StructLiteralExpr(ASTNode):
    def __init__(self, struct_type, field_values):
        self.struct_type = struct_type
        self.field_values = field_values  # list of (field_name, type, expr)

class FieldAccessExpr(ASTNode):
    def __init__(self, obj, field_name):
        self.obj = obj  # VarRefExpr or another FieldAccessExpr
        self.field_name = field_name

class EnumValueExpr(ASTNode):
    def __init__(self, enum_name, value_name):
        self.enum_name = enum_name
        self.value_name = value_name

class IfStmt(ASTNode):
    def __init__(self, conditions_and_bodies):
        # List of (condition, body) pairs. 
        # The last one might have condition=None for 'else'.
        self.conditions_and_bodies = conditions_and_bodies

class WhileStmt(ASTNode):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

class VarAssignStmt(ASTNode):
    def __init__(self, type_name, name, expr):
        self.type_name = type_name
        self.name = name
        self.expr = expr

class FieldAssignStmt(ASTNode):
    def __init__(self, type_name, var_name, field_name, expr):
        self.type_name = type_name
        self.var_name = var_name
        self.field_name = field_name
        self.expr = expr

class LabelDef(ASTNode):
    def __init__(self, name):
        self.name = name

class JmpStmt(ASTNode):
    def __init__(self, target):
        self.target = target

class IfnStmt(ASTNode):
    def __init__(self, condition, target):
        self.condition = condition
        self.target = target

class CmpTStmt(ASTNode):
    def __init__(self, condition, target):
        self.condition = condition
        self.target = target

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.enum_names = set()  # Track known enum names

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

        return Program(outtype, data_block, declarations)

    def parse_data_block(self):
        self.consume(TokenType.KEYWORD, 'data')
        self.consume(TokenType.SYMBOL, '{')
        entries = []
        structs = []
        enums = []
        while self.peek().type != TokenType.SYMBOL or self.peek().value != '}':
            if self.peek().type == TokenType.KEYWORD and self.peek().value == 'struct':
                structs.append(self.parse_struct_def())
            elif self.peek().type == TokenType.KEYWORD and self.peek().value == 'enum':
                enums.append(self.parse_enum_def())
            else:
                type_name = self.consume(TokenType.KEYWORD).value
                name = self.consume(TokenType.IDENTIFIER).value
                self.consume(TokenType.SYMBOL, ':')
                value = self.consume().value  # Could be string or number
                entries.append((type_name, name, value))
        self.consume(TokenType.SYMBOL, '}')
        return DataBlock(entries, structs, enums)

    def parse_struct_def(self):
        self.consume(TokenType.KEYWORD, 'struct')
        name = self.consume(TokenType.IDENTIFIER).value
        self.consume(TokenType.SYMBOL, '{')
        fields = []
        while self.peek().type != TokenType.SYMBOL or self.peek().value != '}':
            field_type = self.consume(TokenType.KEYWORD).value
            field_name = self.consume(TokenType.IDENTIFIER).value
            self.consume(TokenType.SYMBOL, ';')
            fields.append((field_type, field_name))
        self.consume(TokenType.SYMBOL, '}')
        return StructDef(name, fields)

    def parse_enum_def(self):
        self.consume(TokenType.KEYWORD, 'enum')
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
        return EnumDef(name, values)

    def parse_function_decl(self):
        self.consume(TokenType.KEYWORD, 'define')
        name = self.consume(TokenType.IDENTIFIER).value
        self.consume(TokenType.SYMBOL, '(')
        params = self.parse_params()
        self.consume(TokenType.SYMBOL, ')')
        self.consume(TokenType.SYMBOL, '-')
        self.consume(TokenType.SYMBOL, '>')
        return_type = self.consume(TokenType.KEYWORD).value
        self.consume(TokenType.SYMBOL, ';')
        return FunctionDecl(name, params, return_type)

    def parse_function_def(self, is_exported):
        if is_exported:
            self.consume(TokenType.KEYWORD, 'export')
        name = self.consume(TokenType.IDENTIFIER).value
        self.consume(TokenType.SYMBOL, '(')
        params = self.parse_params()
        self.consume(TokenType.SYMBOL, ')')
        self.consume(TokenType.SYMBOL, '-')
        self.consume(TokenType.SYMBOL, '>')
        return_type = self.consume(TokenType.KEYWORD).value
        self.consume(TokenType.SYMBOL, '{')
        body = []
        while self.peek().type != TokenType.SYMBOL or self.peek().value != '}':
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
        self.consume(TokenType.SYMBOL, '}')
        return FunctionDef(name, params, return_type, body, is_exported)

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
                type_name = self.consume(TokenType.KEYWORD).value
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
                    self.consume(TokenType.SYMBOL, ';')
                    return ReturnStmt(ret_type, None)
                expr = self.parse_expression()
                self.consume(TokenType.SYMBOL, ';')
                return ReturnStmt(ret_type, expr)
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
                name = self.consume(TokenType.IDENTIFIER).value
                
                # Check if it's a field assignment (var.field)
                if self.peek().type == TokenType.SYMBOL and self.peek().value == '.':
                    self.consume()  # .
                    field_name = self.consume(TokenType.IDENTIFIER).value
                    self.consume(TokenType.SYMBOL, '=')
                    expr = self.parse_expression()
                    self.consume(TokenType.SYMBOL, ';')
                    return FieldAssignStmt(type_name, name, field_name, expr)
                else:
                    self.consume(TokenType.SYMBOL, '=')
                    expr = self.parse_expression()
                    self.consume(TokenType.SYMBOL, ';')
                    return VarAssignStmt(type_name, name, expr)
            elif token.value == 'jmp':
                self.consume()
                target = self.consume(TokenType.LABEL).value
                self.consume(TokenType.SYMBOL, ';')
                return JmpStmt(target)
            elif token.value == 'ifn':
                self.consume()
                cond = self.consume(TokenType.IDENTIFIER).value
                self.consume(TokenType.SYMBOL, ',')
                target = self.consume(TokenType.LABEL).value
                self.consume(TokenType.SYMBOL, ';')
                return IfnStmt(cond, target)
            elif token.value == 'cmp_t':
                self.consume()
                cond = self.consume(TokenType.IDENTIFIER).value
                self.consume(TokenType.SYMBOL, ',')
                target = self.consume(TokenType.LABEL).value
                self.consume(TokenType.SYMBOL, ';')
                return CmpTStmt(cond, target)
            elif token.value in ['int32', 'int64', 'float32', 'float64', 'string']:
                type_name = self.consume().value
                name = self.consume(TokenType.IDENTIFIER).value
                self.consume(TokenType.SYMBOL, '=')
                expr = self.parse_expression()
                self.consume(TokenType.SYMBOL, ';')
                return VarDeclStmt(type_name, name, expr)
        elif token.type == TokenType.LABEL:
            label_name = self.consume().value
            self.consume(TokenType.SYMBOL, ':')
            return LabelDef(label_name)
        elif token.type == TokenType.IDENTIFIER:
            # Could be struct variable declaration: StructName varName = { ... };
            struct_type = self.consume(TokenType.IDENTIFIER).value
            var_name = self.consume(TokenType.IDENTIFIER).value
            self.consume(TokenType.SYMBOL, '=')
            expr = self.parse_expression()
            self.consume(TokenType.SYMBOL, ';')
            return VarDeclStmt(struct_type, var_name, expr)
        raise RuntimeError(f"Unknown statement {token}")

    def parse_expression(self, min_prec=1):
        token = self.peek()

        # Handle Casts and Unary Operators (Prefix)
        if token.type == TokenType.KEYWORD and token.value in ['int32', 'int64', 'float32', 'float64', 'string']:
            type_name = self.consume().value
            if self.peek().type == TokenType.SYMBOL and self.peek().value == '(':
                # int32(expr)
                self.consume(TokenType.SYMBOL, '(')
                expr = self.parse_expression(1)
                self.consume(TokenType.SYMBOL, ')')
                lhs = TypeCastExpr(type_name, expr)
            else:
                # int32 expr
                rhs = self.parse_expression(11) # High precedence
                lhs = TypeCastExpr(type_name, rhs)
            return self.parse_op_continuation(lhs, min_prec)
            
        elif token.type == TokenType.SYMBOL and token.value == '~':
            op = self.consume().value
            rhs = self.parse_expression(11) # High precedence
            lhs = UnaryExpr(op, rhs)
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
            lhs = BinaryExpr(lhs, op, rhs)
            
        return lhs

    def parse_if_stmt(self):
        conditions_and_bodies = []
        
        # Parse $if
        self.consume(TokenType.KEYWORD, '$if')
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
        return IfStmt(conditions_and_bodies)

    def parse_while_stmt(self):
        self.consume(TokenType.KEYWORD, '$while')
        condition = self.parse_expression()
        body = []
        while self.peek().type != TokenType.KEYWORD or self.peek().value != '$endwhile':
            body.append(self.parse_statement())
        self.consume(TokenType.KEYWORD, '$endwhile')
        return WhileStmt(condition, body)

    def parse_primary(self):
        token = self.peek()
        if token.type == TokenType.NUMBER:
            return LiteralExpr(self.consume().value)
        elif token.type == TokenType.STRING:
            return LiteralExpr(self.consume().value)
        elif token.type == TokenType.KEYWORD and token.value == 'call':
            self.consume()  # call
            name = self.consume(TokenType.IDENTIFIER).value
            self.consume(TokenType.SYMBOL, '(')
            args = []
            while self.peek().type != TokenType.SYMBOL or self.peek().value != ')':
                arg_type = self.consume(TokenType.KEYWORD).value
                arg_expr = self.parse_expression()
                args.append((arg_type, arg_expr))
                if self.peek().type == TokenType.SYMBOL and self.peek().value == ',':
                    self.consume()
            self.consume(TokenType.SYMBOL, ')')
            return CallExpr(name, args)
        elif token.type == TokenType.SYMBOL and token.value == '(':
            self.consume()  # (
            expr = self.parse_expression()
            self.consume(TokenType.SYMBOL, ')')
            return expr
        elif token.type == TokenType.SYMBOL and token.value == '{':
            # Struct literal: { field1: type value, field2: type value }
            return self.parse_struct_literal()
        elif token.type == TokenType.IDENTIFIER:
            name = self.consume().value
            # Check for dot access (e.g., p.x or Color.RED)
            if self.peek().type == TokenType.SYMBOL and self.peek().value == '.':
                # Check if it's an enum value (EnumName.VALUE)
                if name in self.enum_names:
                    self.consume()  # .
                    value_name = self.consume(TokenType.IDENTIFIER).value
                    return EnumValueExpr(name, value_name)
                else:
                    # It's a struct field access
                    expr = VarRefExpr(name)
                    while self.peek().type == TokenType.SYMBOL and self.peek().value == '.':
                        self.consume()  # .
                        field_name = self.consume(TokenType.IDENTIFIER).value
                        expr = FieldAccessExpr(expr, field_name)
                    return expr
            return VarRefExpr(name)
        raise RuntimeError(f"Unexpected token in expression: {token}")

    def parse_struct_literal(self):
        self.consume(TokenType.SYMBOL, '{')
        field_values = []
        while self.peek().type != TokenType.SYMBOL or self.peek().value != '}':
            field_name = self.consume(TokenType.IDENTIFIER).value
            self.consume(TokenType.SYMBOL, ':')
            field_type = self.consume(TokenType.KEYWORD).value
            field_expr = self.parse_expression()
            field_values.append((field_name, field_type, field_expr))
            if self.peek().type == TokenType.SYMBOL and self.peek().value == ',':
                self.consume()
        self.consume(TokenType.SYMBOL, '}')
        return StructLiteralExpr(None, field_values)  # struct_type será inferido do contexto

    def parse_call_stmt(self):
        # We already consumed 'call' in some cases, but here we expect it as a statement
        token = self.consume(TokenType.KEYWORD, 'call')
        name = self.consume(TokenType.IDENTIFIER).value
        self.consume(TokenType.SYMBOL, '(')
        args = []
        while self.peek().type != TokenType.SYMBOL or self.peek().value != ')':
            arg_type = self.consume(TokenType.KEYWORD).value
            arg_expr = self.parse_expression()
            args.append((arg_type, arg_expr))
            if self.peek().type == TokenType.SYMBOL and self.peek().value == ',':
                self.consume()
        self.consume(TokenType.SYMBOL, ')')
        self.consume(TokenType.SYMBOL, ';')
        return CallExpr(name, args)
