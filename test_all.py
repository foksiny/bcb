import os
import subprocess
import glob
import sys
import time

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def run_command(cmd, shell=False):
    """Runs a command and returns (stdout, stderr, returncode)"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=shell, timeout=10)
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Timeout expired", 1
    except Exception as e:
        return "", str(e), 1

def parse_expected_output(bcb_file):
    """Parses expected output from comments in the .bcb file.
    Format: // EXPECT: Some Output\nSecond Line
    """
    expected = []
    with open(bcb_file, 'r') as f:
        for line in f:
            if "// EXPECT:" in line:
                # Replace literal \n with actual newline character
                part = line.split("// EXPECT:")[1].strip()
                expected.append(part.replace("\\n", "\n"))
    return "\n".join(expected) if expected else None

def test_file(bcb_path, base_dir=""):
    display_name = os.path.relpath(bcb_path, base_dir) if base_dir else os.path.basename(bcb_path)
    filename = os.path.basename(bcb_path)
    base_name = os.path.splitext(filename)[0]
    asm_path = f"{base_name}_temp.s"
    exe_path = f"{base_name}_temp.exe"
    
    print(f"Testing {display_name:30} ", end="", flush=True)
    
    start_time = time.time()
    
    # 1. Compile to Assembly
    # We use -m bcb.main to run the package logic
    stdout, stderr, code = run_command([sys.executable, "-m", "bcb.main", bcb_path, "-o", asm_path])
    if code != 0:
        print(f"[{RED}FAILED{RESET}] (Compilation)")
        print(f"  Error: {stderr or stdout}")
        return False
    
    # 2. Assemble and Link with GCC
    stdout, stderr, code = run_command(["gcc", asm_path, "-o", exe_path])
    if code != 0:
        print(f"[{RED}FAILED{RESET}] (Linking)")
        print(f"  Error: {stderr or stdout}")
        if os.path.exists(asm_path): os.remove(asm_path)
        return False
    
    # 3. Execute
    # On Windows, we just run the exe. subprocess.run handles the path.
    exe_abs = os.path.abspath(exe_path)
    stdout, stderr, code = run_command([exe_abs])
    duration = time.time() - start_time
    
    # 4. Cleanup
    if os.path.exists(asm_path): os.remove(asm_path)
    if os.path.exists(exe_path): os.remove(exe_path)
    
    if code != 0:
        print(f"[{RED}FAILED{RESET}] (Execution)")
        print(f"  Exit code: {code}")
        print(f"  Stderr: {stderr}")
        return False
    
    # 5. Check Output (Optional: if EXPECT comments exist)
    expected = parse_expected_output(bcb_path)
    if expected is not None:
        if expected.strip() == stdout.strip():
            print(f"[{GREEN}PASSED{RESET}] ({duration:.2f}s)")
            return True
        else:
            print(f"[{RED}FAILED{RESET}] (Output Mismatch)")
            print(f"  Expected: {repr(expected)}")
            print(f"  Got:      {repr(stdout)}")
            return False
    else:
        print(f"[{GREEN}OK{RESET}] ({duration:.2f}s)")
        return True

def main():
    # Ensure bcb is in path if not installed
    sys.path.append(os.getcwd())

    # Select example directory based on host OS
    if sys.platform.startswith("win"):
        examples_dir = "ex_windows"
    elif sys.platform.startswith("linux"):
        examples_dir = "ex_linux"
    else:
        # Fallback to original examples directory for other platforms
        examples_dir = "examples"

    if not os.path.isdir(examples_dir):
        print(f"No example directory found for this OS: {examples_dir}")
        return

    test_files = glob.glob(os.path.join(examples_dir, "*.bcb"))

    # Check for subdirectories with main.bcb
    for item in os.listdir(examples_dir):
        item_path = os.path.join(examples_dir, item)
        if os.path.isdir(item_path):
            main_bcb = os.path.join(item_path, "main.bcb")
            if os.path.exists(main_bcb):
                test_files.append(main_bcb)
    
    if not test_files:
        print(f"No .bcb files found in {examples_dir}")
        return

    print("=" * 50)
    print(f"BCB Compiler Functional Tests")
    print("=" * 50)
    
    passed = 0
    total = len(test_files)
    
    for bcb_file in test_files:
        if test_file(bcb_file, examples_dir):
            passed += 1
            
    print("=" * 50)
    result_color = GREEN if passed == total else RED
    print(f"Summary: {result_color}{passed}/{total} passed{RESET}")
    print("=" * 50)

if __name__ == "__main__":
    main()
