# 🛠️ BCB: Basic Compiler Backend

**BCB** (Basic Compiler Backend) is a minimalist, industrial-strength compiler designed to bridge the gap between human-readable high-level logic and native machine-level assembly. Targeting **Windows x64**, BCB generates optimized Intel-syntax assembly (`.s`) that maps efficiently to modern processor instructions.

BCB is not just a language; it is a backend focused on simplicity, predictability, and performance.

---

## 🌟 Key Features

- **Intel x64 Architecture**: Native generation of Windows x64 assembly, following the standard Microsoft x64 Calling Convention.
- **Hybrid Syntax**: Enjoy C-style high-level constructs (`$if`, `$while`, `struct`) while retaining the ability to use low-level control flow (`jmp`, `label`, `cmp_t`).
- **Rich Type System**: Native support for:
  - `int32` / `int64` (Signed integers)
  - `float32` / `float64` (IEEE 754 floating point)
  - `string` (Null-terminated C-strings)
- **Seamless C Interoperability**: Call any external C function (like `printf`, `malloc`, or `ShellExecute`) with a single `define` statement.
- **Explicit Mutation**: A unique `md` (modify) keyword system that makes state changes obvious and prevents accidental reassignments.
- **Zero Overhead**: No garbage collection, no heavy runtime. Just code and lightning-fast execution.

---

## � Quick Start

### Installation
BCB is built with Python 3.6+. Install it in editable mode:
```bash
git clone https://github.com/your-repo/bcb.git
cd bcb
pip install -e .
```

### Your First Program (`hello.bcb`)
```bcb
<outtype: win64>

data {
    string msg : "Hello from the Basic Compiler Backend!\n"
}

define printf(msg: string, all: ...args) -> int32;

export main(void) -> int32 {
    call printf(string msg);
    return int32 0;
}
```

### Compile & Run
```bash
# 1. Compile to Assembly
bcb hello.bcb -o hello.s

# 2. Assemble and Link with GCC
gcc hello.s -o hello.exe

# 3. Execute
./hello.exe
```

---

## 📖 Comprehensive Language Reference

### 1. Program Structure
A BCB file is divided into four primary sections:
1. **Target Header**: `<outtype: win64>` - Must be the first line.
2. **Data Block**: `data { ... }` - Global constants, string literals, structs, and enums.
3. **External Definitions**: `define [name]([params]) -> [type];` - Link to external functions.
4. **Function Implementations**: `export` or local function blocks.

### 2. The Type System
| Type | Bits | Description |
| :--- | :--- | :--- |
| `int32` | 32 | Standard signed integer |
| `int64` | 64 | Long signed integer |
| `float32` | 32 | Single-precision float |
| `float64` | 64 | Double-precision float |
| `string` | 64 | Pointer to a null-terminated string |

#### Type Casting
Cast values explicitly using the syntax `TypeName(expression)`:
```bcb
int32 i = 10;
float64 d = float64(i); // Converts int to double
```

### 3. Variables and the `md` Keyword
In BCB, variable declaration and modification are separate operations:
- **Declare**: `Type Name = Value;` (Always creates a new local)
- **Modify**: `md Type Name = Value;` (Updates existing variable)

```bcb
int32 x = 5;
md int32 x = x * 2; // Required for updates
```

### 4. Data Structures

#### Structs
Define memory layouts in the `data` block:
```bcb
data {
    struct Player {
        int32 id;
        float32 health;
    }
}

export main(void) -> int32 {
    Player p = { id: int32 1, health: float32 100.0 };
    md float32 p.health = p.health - float32(10.5);
}
```

#### Enums
Constants represented as integers starting from 0:
```bcb
data {
    enum State { IDLE, RUNNING, DEAD }
}
// Usage: State.IDLE resolve to 0
```

### 5. Control Flow

#### High-Level Constructs (Recommended)
Prefix with `$` to distinguish from labels:
```bcb
$if a > b
    // logic
$elseif a == b
    // logic
$else
    // logic
$endif
```

#### Low-Level Labels (Advanced)
Directly use assembly-style jumps for maximum control:
```bcb
    int32 check = x < 0;
    ifn check, @negative; // If not true, jump
    call printf(string pos_msg);
    jmp @end;
@negative:
    call printf(string neg_msg);
@end:
```

### 6. External Functions and Variadics
BCB supports **Variadic Functions** (like `printf`). Use the `...args` syntax:
```bcb
define printf(fmt: string, all: ...args) -> int32;

// Call with any number of arguments
call printf(string format, int32 val, float64 score);
```

---

## 🏗️ Compiler Internals

BCB is designed as a **Single-Pass Backend**. It performs:
1. **Tokenization**: Pattern matching using regex.
2. **Recursive Descent Parsing**: Building a simplified AST (Abstract Syntax Tree).
3. **Code Generation**: Translating AST nodes directly into x64 instructions.

### Calling Convention (Win64 ABI)
- **Registers**: First 4 arguments in `RCX`, `RDX`, `R8`, `R9`.
- **Floating Point**: First 4 float arguments in `XMM0` through `XMM3`.
- **Shadow Space**: Automatically allocates 32 bytes of "Shadow Space" on the stack before calls, as required by the Windows ABI.

---

## 🧪 Testing and Quality Assurance
Every feature of the compiler is validated using the `test_all.py` suite. The suite performs:
- **E2E Compilation Check**: Verification that `.bcb` compiles to `.s`.
- **Linking Validation**: Verification that GCC can link the output.
- **Execution Output Mapping**: Compares program `stdout` against `// EXPECT:` comments in the source.

```bash
python test_all.py
```

---

## 📁 Project Structure
- `bcb/`: Core compiler source code.
  - `lexer.py`: Lexical analysis.
  - `parser.py`: AST generation.
  - `codegen.py`: Native x64 assembly generator.
- `examples/`: Comprehensive library of language features.
- `test_all.py`: Functional test suite.

---

## 📜 License
BCB (Basic Compiler Backend) is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

**Crafted with precision by Antigravity.**
