import sys
from bcb.lexer import tokenize
from bcb.parser import Parser
from bcb.codegen import CodeGen

def main():
    if len(sys.argv) < 2:
        print("Usage: bcb <input_file> [-o <output_file>]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = "output.s"
    if "-o" in sys.argv:
        idx = sys.argv.index("-o")
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]

    with open(input_file, 'r') as f:
        code = f.read()

    try:
        tokens = tokenize(code)
        parser = Parser(tokens)
        ast = parser.parse()
        codegen = CodeGen(ast)
        asm = codegen.generate()

        with open(output_file, 'w') as f:
            f.write(asm)
        
        print(f"Compiled {input_file} to {output_file}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
