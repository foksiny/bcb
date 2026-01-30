import sys
import os
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
        abs_input = os.path.abspath(input_file)
        tokens = tokenize(code)
        parser = Parser(tokens, os.path.dirname(abs_input), {abs_input})
        ast = parser.parse()
        
        # Diagnostics and Analysis
        from bcb.errors import ErrorManager
        from bcb.analyzer import SemanticAnalyzer
        
        error_manager = ErrorManager(code, input_file)
        analyzer = SemanticAnalyzer(ast, error_manager)
        analyzer.analyze()
        
        if error_manager.diagnostics:
             error_manager.print_diagnostics()
             
        if error_manager.has_error:
             print("Compilation failed due to errors.")
             sys.exit(1)

        codegen = CodeGen(ast)
        asm = codegen.generate()

        with open(output_file, 'w') as f:
            f.write(asm)
        
        print(f"Compiled {input_file} to {output_file}")

    except Exception as e:
        # Fallback for unhandled exceptions (like in lexer/parser runtime errors if not caught)
        print(f"Internal Compiler Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
