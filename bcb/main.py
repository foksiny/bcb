import sys
import os
import time
from bcb.lexer import tokenize
from bcb.parser import Parser
from bcb.codegen import CodeGen

VERSION = "1.0.6"
VERSION_STRING = f"BCB Compiler {VERSION} Final Release"

HELP_TEXT = f"""
{VERSION_STRING}
High-Performance Native Code Compiler

USAGE:
    bcb <input_file> [OPTIONS]

OPTIONS:
    -o <file>       Output assembly file (default: output.s)
    -O0             No optimization
    -O1             Basic optimizations (constant folding)
    -O2             Standard optimizations (default)
    -O3             Aggressive optimizations (maximum performance)
    --analyze, -a   Only run semantic analysis (check for errors/warnings)
    --stats         Show optimization statistics
    --help, -h      Show this help message
    --version, -v   Show version information

EXAMPLES:
    bcb main.bcb                    Compile with default settings
    bcb main.bcb -o main.s          Specify output file
    bcb main.bcb -O3 --stats        Maximum optimization with stats
    bcb main.bcb -O0                No optimization (fastest compile)
    bcb main.bcb --analyze          Only check for errors and warnings

OPTIMIZATION LEVELS:
    O0  No optimization - fastest compilation, largest/slowest output
    O1  Basic - constant folding, algebraic simplifications, XOR zero optimization
    O2  Standard - dead code elimination, strength reduction, value numbering,
                   branch prediction, conditional moves, tail merging (default)
    O3  Aggressive - function inlining, LICM, peephole optimization,
                     partial redundancy elimination, global value numbering,
                     instruction scheduling, macro fusion, short jump optimization

For more information, visit: https://github.com/foksiny/bcb
"""

def print_version():
    print(VERSION_STRING)
    print(f"Target: x86-64 (Windows/Linux)")
    print(f"Python: {sys.version.split()[0]}")

def print_help():
    print(HELP_TEXT)

def main():
    # Handle --help and --version early
    if len(sys.argv) < 2 or "--help" in sys.argv or "-h" in sys.argv:
        print_help()
        sys.exit(0 if "--help" in sys.argv or "-h" in sys.argv else 1)
    
    if "--version" in sys.argv or "-v" in sys.argv:
        print_version()
        sys.exit(0)

    input_file = sys.argv[1]
    
    # Check if input file looks like an option
    if input_file.startswith("-"):
        print(f"Error: Expected input file, got '{input_file}'")
        print("Use 'bcb --help' for usage information.")
        sys.exit(1)
    
    output_file = "output.s"
    optimization_level = 2  # Default to O2
    show_stats = "--stats" in sys.argv
    analyze_only = "--analyze" in sys.argv or "-a" in sys.argv
    
    if "-o" in sys.argv:
        idx = sys.argv.index("-o")
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]
    
    # Parse optimization level
    for arg in sys.argv:
        if arg.startswith("-O") and len(arg) == 3 and arg[2].isdigit():
            optimization_level = int(arg[2])
            optimization_level = min(3, max(0, optimization_level))

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    with open(input_file, 'r') as f:
        code = f.read()

    try:
        start_time = time.perf_counter()
        
        abs_input = os.path.abspath(input_file)
        tokens = tokenize(code)
        parser = Parser(tokens, os.path.dirname(abs_input), {abs_input})
        ast = parser.parse()
        
        # Diagnostics and Analysis
        from bcb.errors import ErrorManager
        from bcb.analyzer import SemanticAnalyzer
        from bcb.optimizer import ASTOptimizer, optimize_assembly
        
        error_manager = ErrorManager(code, input_file)
        
        # Register imported files for error display
        for imported_path in parser.imported_files:
            if imported_path != abs_input and os.path.exists(imported_path):
                with open(imported_path, 'r') as f:
                    error_manager.add_source_file(imported_path, f.read())
        
        analyzer = SemanticAnalyzer(ast, error_manager)
        analyzer.analyze()
        
        if error_manager.diagnostics:
             error_manager.print_diagnostics()
              
        if error_manager.has_error:
             print("Compilation failed due to errors.")
             sys.exit(1)
        
        # If --analyze mode, stop here
        if analyze_only:
            total_time = time.perf_counter() - start_time
            if not error_manager.diagnostics:
                print(f"Analysis complete: No issues found in {input_file}")
            print(f"Analysis time: {total_time*1000:.2f}ms")
            sys.exit(0)

        # Optimization Phase
        opt_start = time.perf_counter()
        
        if optimization_level > 0:
            optimizer = ASTOptimizer(ast, optimization_level)
            ast = optimizer.optimize()
            
            if show_stats:
                stats = optimizer.get_stats()
                print(stats)
        
        # Code Generation Phase
        codegen = CodeGen(ast, optimization_level)
        asm = codegen.generate()
        
        # Assembly-level optimizations (peephole)
        if optimization_level >= 1:
            asm = optimize_assembly(asm, optimization_level)
        
        opt_time = time.perf_counter() - opt_start
        total_time = time.perf_counter() - start_time

        with open(output_file, 'w') as f:
            f.write(asm)
        
        opt_name = ["none", "basic", "standard", "aggressive"][optimization_level]
        print(f"Compiled {input_file} -> {output_file} (O{optimization_level}: {opt_name})")
        
        if show_stats:
            print(f"Optimization time: {opt_time*1000:.2f}ms")
            print(f"Total compile time: {total_time*1000:.2f}ms")

    except Exception as e:
        # Fallback for unhandled exceptions
        print(f"Internal Compiler Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
