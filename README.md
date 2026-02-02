# 🛠️ BCB: Basic Compiler Backend
**"The Definitive Guide to High-Performance Low-Level Programming"**

**BCB** (Basic Compiler Backend) is a minimalist yet powerful compiler designed to bridge the gap between human-readable high-level logic and native **Windows x64 assembly/Linux x86_64 assembly** (`.s`). BCB is't just a language; it's a backend system that gives you total control over the machine while providing the comfort of modern syntax.

---

## 📖 Table of Contents
1. [Core Philosophy](#core-philosophy)
2. [Project Anatomy](#project-anatomy)
3. [The Type System & Variables](#the-type-system--variables)
4. [The Semantic Analyzer & Diagnostics](#the-semantic-analyzer--diagnostics)
5. [The Mutation System (md)](#the-mutation-system-md)
6. [Data & Memory Structures (Structs, Enums, Arrays)](#data--memory-structures)
7. [Structured Control Flow ($)](#structured-control-flow-)
8. [Assembly-Level Control Flow (@)](#assembly-level-control-flow-)
9. [Advanced Math & Logic](#advanced-math--logic)
10. [C Interopterability & Custom Functions](#c-interopterability--custom-functions)
11. [Pointer Dereferencing Syntax](#pointer-dereferencing-syntax)
12. [Compiler Internals & Win64 ABI](#compiler-internals--win64-abi)
13. [Building and Testing](#building-and-testing)

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
<outtype: win64> // HEADER: Mandatory. Tells the backend to target Windows x64. (change to linux64 for linux target)

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

// Delayed Initialization using 'no_value' (initializes to zero)
int32 future_val = no_value;
```

### Pointers & Addressing
BCB supports **typed pointers** to stack variables, with explicit address-of and dereference operations.

- **Pointer declaration**
  ```bcb
  int32  x   = 1;
  int32* px  = &x;   // px points to x
  ```

- **Address-of (`&`)**
  - `&var` evaluates to the address of a local variable.
  - Result type is `<base_type>*` (e.g., `int32*`).

- **Dereference (`*`)**
  - `*ptr` loads the value from the address contained in `ptr`.
  - Example:
    ```bcb
    int32  x   = 1;
    int32* px  = &x;
    int32  y   = *px;   // y = 1
    ```

- **Passing pointers to functions**
  ```bcb
  // Pointer parameter (like int32* in C)
  change(ptr: int32*) -> void {
      // Overwrite the pointee: *ptr = 2;
      md int32* ptr = 2;
      return void;
  }

  export main(void) -> int32 {
      int32 a   = 1;
      int32* p  = &a;

      // Pass the pointer itself to the function (still uses *p)
      call change(int32 *p);

      // a was modified through the pointer
      call printf(string "%d\n", int32 a);  // prints 2
      return int32 0;
  }
  ```

- **Pointers in calls vs. values**
  - `int32* p` in a **declaration** or parameter type means "pointer to int32".
  - In a **call**, the type before the expression describes the expected base type:
    - `call change(int32 *p);` → passes the pointer value `p` into `ptr: int32*`.
    - `call printf(string fmt, int32* p);` → passes the **value pointed to by** `p` (because of the `*` in the expression).

BCB treats pointers as 64‑bit integers at the ABI level (Windows x64), so they are passed and returned in the standard integer registers (`RCX`, `RDX`, `R8`, `R9`, `RAX`) just like `int64`.

---

## 4. The Semantic Analyzer & Diagnostics
BCB now includes a powerful **Semantic Analyzer** that runs before compilation. It checks your code for logical errors, type mismatches, and potential runtime hazards.

### Analyzer Capabilities
1.  **Type Safety**: Ensures you don't accidentally assign a `float32` to an `int64` without casting.
2.  **Scope Validation**: Checks that variables and functions are declared before use.
3.  **Strict Declarations**: External functions (like `printf`) **must** be explicitly defined using `define` or imported. Implicit usage is no longer allowed.
4.  **Pointer Safety**: Validates pointer levels (e.g., passing `int32` where `int32*` is expected produces an error).

### Diagnostic Levels
The compiler reports feedback in four levels, using colored output for readability:

| Level | Color | Meaning |
| :--- | :--- | :--- |
| **ERROR** | 🔴 Red | A hard failure. Compilation stops (e.g., Syntax error, Type mismatch). |
| **WARNING** | 🟡 Yellow | The code will compile, but it's suspicious (e.g., Implicit conversion that might be unintended). |
| **PRE** | 🟠 Orange | **Possible Runtime Error**. See detail below. |
| **TIP** | 🔵 Blue | A helpful suggestion to fix an error (e.g., "Did you forget to define 'printf'?"). |

### What is a PRE (Possible Runtime Error)?
A **PRE** is a special warning category unique to BCB. It detects code that is syntactically valid but statistically likely to cause bugs or crashes at runtime.

**Common PREs:**
- **Implicit Truncation**: Converting a large integer (`int64`) to a smaller one (`int32`) without an explicit cast.
  - *Example*: `int32 x = my_int64_var;` -> **PRE**: "Implicit conversion from 'int64' to 'int32' may truncate value."
- **Supression**: The analyzer is smart! If you assign a safe literal like `int32 x = 1;` (where `1` fits in 32 bits), the PRE is automatically suppressed.

---

## 5. The Mutation System (md)
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
int32 i = 5;              // ERROR: 'i' is already declared in this scope.
```

---

## 6. Data & Memory Structures

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

### Arrays & Lists: Contiguous Memory
BCB supports fixed-size stack-allocated arrays. Arrays in BCB are "smart" and carry their length in a hidden 8-byte header before the data.

**Declaration & Initialization:**
```bcb
// Declare an array of 5 integers
int32 numbers[5] = {10, 20, 30, 40, 50};

// Partial initialization (remaining elements zeroed)
int32 zeroes[10] = {0}; 
```

**Access & Modification:**
```bcb
// Accessing an element
int32 first = numbers[0];

// Modifying an element (requires 'md' and index)
md int32 numbers[1] = 99;
```

**Array Utilities:**
- **`length(arr)`**: Returns the number of elements in the array (read from the hidden header).
- **Passing to functions**: Use `type name[]` in the parameter list. Inside the function, the array acts as a pointer to the first element.
  ```bcb
  print_ints(arr: int32[]) -> void {
      int32 len = length(arr);
      // ... loop through arr[i] ...
  }

  // Call using [] syntax to pass the base address
  call print_ints(int32 arr[]);
  ```

**Advanced: Arrays of Structs**
BCB allows you to nest structs within arrays for complex data handling:
```bcb
MyStruct list[2] = { {x: 1, y: 2}, {x: 3, y: 4} };
int32 val = list[0].x; // Accessing field of array element
```

---

## 7. Structured Control Flow ($)
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

## 8. Assembly-Level Control Flow (@)
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

### Manual Stack Operations
You can manually push and pop values to/from the stack. Be careful to balance your stack!

```bcb
push int64 100;    // Pushes 100 onto the stack
push int32 42;     // Pushes 42 (padded to 64-bit on stack)

pop int32 val1;    // Pops into variable val1
pop int64 val2;    // Pops into variable val2
```

---

## 9. Advanced Math & Logic
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

## 10. C Interopterability & Custom Functions
BCB is a "good neighbor" to C. You can link with any C library (like the Windows CRT).

### Defining External Functions
**Strict Rule**: You must explicitly define any external function you use. Check your standard library (or `msvcrt` on Windows) for signatures.
```bcb
define printf(fmt: string, all: ...args) -> int32;
define malloc(size: int64) -> int64;
define free(ptr: int64) -> void;
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

## 11. Pointer Dereferencing Syntax
When passing pointers to functions (like `printf`), explicit syntax controls whether you pass the **address** or the **value**.

- **Passing the Address (Pointer)**:
  ```bcb
  int32* ptr = ...;
  call printf(msg, int32* ptr); // Passes the address 0x1234ABC...
  ```

- **Passing the Value (Dereference)**:
  Use parentheses around the dereference expression `(*ptr)` to extract the value.
  ```bcb
  int32* ptr = ...;
  call printf(msg, int32 (*ptr)); // Dereferences and passes the value (e.g., 42)
  ```

---

## 12. Compiler Internals & Win64 ABI
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

## 13. Building and Testing
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
