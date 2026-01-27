import re

class TokenType:
    IDENTIFIER = 'IDENTIFIER'
    KEYWORD = 'KEYWORD'
    STRING = 'STRING'
    NUMBER = 'NUMBER'
    SYMBOL = 'SYMBOL'
    LABEL = 'LABEL'
    EOF = 'EOF'

KEYWORDS = {
    'outtype', 'data', 'string', 'define', 'export', 'void', 'int32', 'int64', 'float32', 'float64', 'call', 'return',
    '$if', '$elseif', '$else', '$endif',
    '$while', '$endwhile', 'md',
    'jmp', 'ifn', 'cmp_t',
    'struct', 'enum'
}

TOKEN_SPEC = [
    ('STRING', r'"[^"]*"'),
    ('LABEL', r'@[a-zA-Z_][a-zA-Z0-9_]*'),
    ('COMMENT', r'//.*'),
    ('NUMBER', r'\d+\.\d+|\d+'),
    ('IDENTIFIER', r'[$a-zA-Z_][$a-zA-Z0-9_]*'),
    ('SYMBOL', r'<<|>>|==|!=|<=|>=|&&|\|\||[:;(){}\[\]\.,\-\>\<=\+\*\/\&\|\^\~\%]'),
    ('NEWLINE', r'\n'),
    ('SKIP', r'[ \t]+'),
    ('MISMATCH', r'.'),
]

class Token:
    def __init__(self, type, value, line, column):
        self.type = type
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f"Token({self.type}, {repr(self.value)}, {self.line}, {self.column})"

def tokenize(code):
    tokens = []
    line_num = 1
    line_start = 0
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in TOKEN_SPEC)
    for mo in re.finditer(tok_regex, code):
        kind = mo.lastgroup
        value = mo.group()
        column = mo.start() - line_start
        if kind == 'STRING':
            tokens.append(Token(TokenType.STRING, value[1:-1], line_num, column))
        elif kind == 'NUMBER':
            val = value
            if '.' in val:
                tokens.append(Token(TokenType.NUMBER, float(val), line_num, column))
            else:
                tokens.append(Token(TokenType.NUMBER, int(val), line_num, column))
        elif kind == 'IDENTIFIER':
            if value in KEYWORDS:
                tokens.append(Token(TokenType.KEYWORD, value, line_num, column))
            else:
                tokens.append(Token(TokenType.IDENTIFIER, value, line_num, column))
        elif kind == 'SYMBOL':
            tokens.append(Token(TokenType.SYMBOL, value, line_num, column))
        elif kind == 'LABEL':
            tokens.append(Token(TokenType.LABEL, value, line_num, column))
        elif kind == 'COMMENT':
            pass
        elif kind == 'NEWLINE':
            line_start = mo.end()
            line_num += 1
        elif kind == 'SKIP':
            pass
        elif kind == 'MISMATCH':
            raise RuntimeError(f'{value!r} unexpected on line {line_num}')
    tokens.append(Token(TokenType.EOF, None, line_num, 0))
    return tokens
