# 🛠️ BCB: Basic Compiler Backend
**"The Definitive Guide to High-Performance Low-Level Programming"**

**BCB** (Basic Compiler Backend) is a minimalist yet powerful compiler designed to bridge the gap between human-readable high-level logic and native **Windows x64 assembly** (`.s`). BCB is't just a language; it's a backend system that gives you total control over the machine while providing the comfort of modern syntax.

---

## 📖 Table of Contents
1. [Core Philosophy](#core-philosophy)
2. [Project Anatomy](#project-anatomy)
3. [The Type System & Variables](#the-type-system--variables)
4. [The Mutation System (md)](#the-mutation-system-md)
5. [Data & Memory Structures](#data--memory-structures)
6. [Structured Control Flow ($)](#structured-control-flow-)
7. [Assembly-Level Control Flow (@)](#assembly-level-control-flow-)
8. [Advanced Math & Logic](#advanced-math--logic)
9. [C Interopterability & Custom Functions](#c-interopterability--custom-functions)
10. [Compiler Internals & Win64 ABI](#compiler-internals--win64-abi)

---

## 1. Core Philosophy
BCB is built on the principle of **Deterministic Code Generation**. 
- **No Garbage Collection**: Memory is managed on the stack or via manual C calls.
- **Explicit over Implicit**: State changes are marked with `md`.
- **Transparency**: You can always read the generated `.s` file and see a 1:1 mapping to your BCB code.

---

## 2. Project Anatomy
Every BCB program follows a strict, predictable layout. Understanding this layout is key to mastering the language.

```bcb
<outtype: win64> // HEADER: Mandatory. Tells the backend to target Windows x64.

data {           // STATIC SECTION: Where all constants and structures live.
    struct Point { int32 x; int32 y; }
    string welcome : "System Initialized...\n"
}

// DEFINITIONS: Prototypes for external (C/DLL) functions.
define printf(fmt: string, all: ...args) -> int32;

// FUNCTIONS: Your logic starts here.
export main(void) -> int32 {
    call printf(string welcome);
    return int32 0;
}
```

---

## 3. The Type System & Variables
BCB is statically typed. Every variable occupies a specific amount of space on the stack.

| Type | Bit Width | Assembly Reg | Usage |
| :--- | :--- | :--- | :--- |
| `int32` | 32-bit | EAX/R-Dword | Standard numbers, loop counters. |
| `int64` | 64-bit | RAX/R-Qword | Large numbers, pointers, memory addresses. |
| `float32`| 32-bit | XMM0-15 | Single-precision decimal math. |
| `float64`| 64-bit | XMM0-15 | High-precision decimal math. |
| `string` | 64-bit | RIP-Relative | Pointers to constant text in the `.rdata` section. |

### Declaration Basics
```bcb
int32 age = 25;
float64 price = 99.99;
string label = "Inventory Item";
```

---

## 4. The Mutation System (md)
The most unique feature of BCB is the `md` (Modify) statement. In traditional languages, `x = 10` can be both a declaration and an assignment. In BCB, they are strictly separated.

- **Initialization**: `int32 x = 10;` (Creates space on the stack)
- **Modification**: `md int32 x = 20;` (Reaches into established stack space and overwrites)

**Why `md`?**
1. **Safety**: You can never accidentally overwrite a variable you didn't mean to.
2. **Speed**: The compiler knows exactly where the variable is (offset from RBP) and performs a single `mov` instruction.
3. **Clarity**: When reading code, `md` acts as a red flag for state change.

```bcb
int32 i = 0;
md int32 i = i + 1; // Correct
i = 5;              // ERROR: Redeclaration of 'i' in the same scope.
```

---

## 5. Data & Memory Structures

### Structs: Custom Layouts
Structs are blueprints for memory. They are declared in the `data` block.
```bcb
data {
    struct Color {
        int32 r;
        int32 g;
        int32 b;
    }
}
```
**Initialization & Usage:**
```bcb
export main(void) -> int32 {
    Color red = { r: int32 255, g: int32 0, b: int32 0 };
    
    // Modification requires the 'md' keyword and the full path
    md int32 red.g = int32 50; 
    
    return int32 red.r;
}
```

### Enums: Named Integers
Enums are simple integer mappings (0, 1, 2...).
```bcb
data {
    enum Status { IDLE, BUSY, DONE }
}
// Status.IDLE is 0, Status.BUSY is 1, etc.
```

---

## 6. Structured Control Flow ($)
BCB uses high-level control flow structures prefixed with `$`. These are luxurious abstractions over raw assembly jumps.

### Conditionals ($if)
Unlike C, BCB often requires you to be explicit about types inside conditions to ensure correct comparison instructions (`cmp` vs `ucomiss`).
```bcb
$if int32 age >= int32 18
    call drink_beer();
$elseif int32 age > int32 16
    call drive_car();
$else
    call go_home();
$endif
```

### Loops ($while)
```bcb
int32 i = 0;
$while int32 i < int32 10
    call printf(string fmt, int32 i);
    md int32 i = i + 1;
$endwhile
```

---

## 7. Assembly-Level Control Flow (@)
For those who want to "hand-roll" their logic, BCB supports raw labels and jumps. This is what the `$` constructs compile into!

| Instruction | Meaning |
| :--- | :--- |
| `@label:` | Defines a target location in the code. |
| `jmp @label;` | Unconditional jump to a label. |
| `cmp_t var, @label;` | Jump to label if `var` is **True** (non-zero). |
| `ifn var, @label;` | Jump to label if `var` is **False** (zero). |

**Example of a manual loop:**
```bcb
    int32 i = 0;
@loop_start:
    int32 check = i < 10;
    ifn check, @loop_end; // Exit if i is not < 10
    
    call printf(string msg, int32 i);
    md int32 i = i + 1;
    jmp @loop_start;
@loop_end:
```

---

## 8. Advanced Math & Logic
BCB supports all standard arithmetic and bitwise operations.

### Operators
- **Math**: `+`, `-`, `*`, `/`, `%`
- **Bitwise**: `&` (AND), `|` (OR), `^` (XOR), `~` (NOT), `<<` (Left Shift), `>>` (Right Shift)
- **Logic**: `&&`, `||`, `==`, `!=`, `<`, `>`, `<=`, `>=`

### The "Math Default" Trap
**Crucial Lesson**: BCB defaults to `int32` math. If you are working with floats, you **MUST** cast the expression to ensure the compiler uses the Floating Point Unit (FPU/XMM).
```bcb
float32 a = 1.5;
float32 b = 2.5;

// WRONG: This will perform int math and lose precision!
md float32 a = a + b; 

// CORRECT: Forces XMM additions (addss/addsd)
md float32 a = float32(a + b); 
```

---

## 9. C Interopterability & Custom Functions
BCB is a "good neighbor" to C. You can link with any C library (like the Windows CRT).

### Defining External Functions
```bcb
define malloc(size: int64) -> int64;
define free(ptr: int64) -> void;
define getchar(void) -> int32;
```

### Creating Your Own Functions
Functions can be `export` (visible to the linker/externals) or local (internal to the file).
```bcb
// A custom function for calculating squares
square(n: int32) -> int32 {
    return int32 n * n;
}

export main(void) -> int32 {
    int32 result = call square(int32 5);
    return int32 result;
}
```

---

## 10. Compiler Internals & Win64 ABI
To write truly advanced BCB, you must understand how it talks to the CPU.

### The Calling Convention (Microsoft x64)
BCB handles the complexity of the Windows x64 ABI for you:
1. **Regs 1-4**: The first 4 arguments are automatically placed in `RCX`, `RDX`, `R8`, and `R9`.
2. **Floats**: Floating point arguments go into `XMM0-XMM3`.
3. **Shadow Space**: Windows requires 32 bytes of "scratch space" on the stack before any function call. BCB automatically subtracts `rsp, 32` before a call and cleans it up after.
4. **Variadics**: For functions like `printf(fmt, ...)`, BCB is smart enough to copy float values from `XMM` registers into the General Purpose Registers (GPRs) because variadic C functions expect them there.

### Stack Frame Layout
When a function starts, BCB:
1. Saves the old Base Pointer (`push rbp`).
2. Sets up a new Base Pointer (`mov rbp, rsp`).
3. Allocates space for locals (`sub rsp, 64`).
4. Every variable you declare is assigned a negative offset from `RBP` (e.g., `[rbp - 8]`).

---

## 🔨 Building and Testing
```bash
# Install the bcb command
pip install -e .

# Run the full test suite
python test_all.py

# Manual Compilation
bcb examples/math.bcb -o math.s
gcc math.s -o math.exe
./math.exe
```

---

**Master the machine. Control the stack. Build with BCB.**

*Crafted by Antigravity.*
