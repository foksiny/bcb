# 🛠️ BCB: Basic Compiler Backend
**Version 1.0.6**

**"The Definitive Guide to High-Performance Low-Level Programming"**

**BCB** (Basic Compiler Backend) is a minimalist yet powerful compiler designed to bridge the gap between human-readable high-level logic and native **Windows x64 assembly/Linux x86_64 assembly** (`.s`). BCB is't just a language; it's a backend system that gives you total control over the machine while providing the comfort of modern syntax.

---

## 📖 Table of Contents
1. [Core Philosophy](#core-philosophy)
2. [Quick Start](#quick-start)
3. [Command Line Options](#command-line-options)
4. [Optimization Levels](#optimization-levels)
5. [Project Anatomy](#project-anatomy)
6. [The Type System & Variables](#the-type-system--variables)
    - [Declaration Basics](#declaration-basics)
    - [Public Variables (Global Scope)](#public-variables-global-scope)
    - [Pointers & Addressing](#pointers--addressing)
7. [The Semantic Analyzer & Diagnostics](#the-semantic-analyzer--diagnostics)
8. [The Mutation System (md)](#the-mutation-system-md)
9. [Data & Memory Structures (Structs, Enums, Arrays)](#data--memory-structures)
10. [Structured Control Flow ($)](#structured-control-flow-)
11. [Assembly-Level Control Flow (@)](#assembly-level-control-flow-)
12. [Advanced Math & Logic](#advanced-math--logic)
13. [C Interopterability & Custom Functions](#c-interopterability--custom-functions)
14. [Runtime Type Information & Variadic Arguments](#runtime-type-information--variadic-arguments)
    - [The gettype() Function](#the-gettype-function)
    - [Variadic Functions with ...args](#variadic-functions-with-args)
    - [myargs.amount - Argument Count](#myargsamount---argument-count)
    - [myargs(index) - Accessing Arguments by Index](#myargsindex---accessing-arguments-by-index)
15. [Pointer Dereferencing Syntax](#pointer-dereferencing-syntax)
16. [Compiler Internals & Win64 ABI](#compiler-internals--win64-abi)
17. [Macros (Metaprogramming)](#macros-metaprogramming)
18. [Building and Testing](#building-and-testing)

---

## 1. Core Philosophy
BCB is built on the principle of **Deterministic Code Generation**. 
- **No Garbage Collection**: Memory is managed on the stack or via manual C calls.
- **Explicit over Implicit**: State changes are marked with `md`.
- **Transparency**: You can always read the generated `.s` file and see a 1:1 mapping to your BCB code.
- **High Performance**: Advanced optimizations generate code that rivals C compilers.

---

## 2. Quick Start
```bash
# Install BCB
pip install -e .

# Check version
bcb --version

# Get help
bcb --help

# Compile a program
bcb hello.bcb -o hello.s
gcc hello.s -o hello.exe
./hello.exe
```

---

## 3. Command Line Options
BCB provides a rich set of command-line options for controlling compilation:

```
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
    --help, -h      Show help message
    --version, -v   Show version information
```

### Examples
```bash
# Compile with default settings (O2)
bcb main.bcb

# Specify output file
bcb main.bcb -o main.s

# Maximum optimization with statistics
bcb main.bcb -O3 --stats

# No optimization (fastest compile time)
bcb main.bcb -O0

# Only check for errors and warnings (no compilation)
bcb main.bcb --analyze
```

---

## 4. Optimization Levels
BCB features a **powerful multi-pass optimizer** that can generate code rivaling or exceeding C compiler performance.

| Level | Name | Description |
|:------|:-----|:------------|
| `-O0` | None | No optimization. Fastest compilation, largest output. |
| `-O1` | Basic | Constant folding, algebraic simplifications. |
| `-O2` | Standard | Dead code elimination, strength reduction. **(Default)** |
| `-O3` | Aggressive | Function inlining, LICM, peephole optimization. |

### Optimization Techniques

#### High-Level (AST) Optimizations
| Technique | Description | Example |
|:----------|:------------|:--------|
| **Constant Folding** | Evaluates expressions at compile time | `3 + 5` → `8` |
| **Strength Reduction** | Replaces expensive ops with cheaper ones | `x * 8` → `x << 3` |
| **Multiply Decomposition** | Breaks down multiplication into shifts | `x * 3` → `(x << 1) + x` |
| **Dead Code Elimination** | Removes unreachable code after `return` | |
| **Copy Propagation** | Propagates constant values through variables | |
| **Function Inlining** | Inlines simple functions (O3 only) | |
| **LICM** | Moves loop-invariant code outside loops | |
| **Algebraic Simplification** | Simplifies expressions | `x - x` → `0` |

#### Low-Level (Assembly) Optimizations
| Technique | Description | Benefit |
|:----------|:------------|:--------|
| **XOR Zeroing** | `xor eax, eax` instead of `mov rax, 0` | Smaller, faster |
| **32-bit MOV** | Uses `mov eax, val` for small values | Zero-extends automatically |
| **LEA Arithmetic** | `lea rax, [rax + rbx]` for addition | Doesn't affect flags |
| **Peephole Optimization** | Removes redundant push/pop sequences | |

### Using `--stats`
The `--stats` flag shows detailed optimization statistics:
```bash
$ bcb example.bcb -O3 --stats

Optimization Statistics:
  Constants folded:      12
  Dead code eliminated:  3
  CSE applied:           0
  Loops unrolled:        0
  Functions inlined:     2
  Strength reductions:   5
  Copy propagations:     4
  Dead stores eliminated:0
  LICM applied:          1
  Peephole optimizations:8

Optimization time: 2.34ms
Total compile time: 15.67ms
```

---

## 5. Project Anatomy
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
| `int8` | 8-bit | AL/R-Byte | Standard numbers, loop counters. |
| `int16` | 16-bit | AX/R-Word | Standard numbers, loop counters. |
| `int32` | 32-bit | EAX/R-Dword | Standard numbers, loop counters. |
| `int64` | 64-bit | RAX/R-Qword | Large numbers, pointers, memory addresses. |
| `float32`| 32-bit | XMM0-15 | Single-precision decimal math. |
| `float64`| 64-bit | XMM0-15 | High-precision decimal math. |
| `string` | 64-bit | RIP-Relative | Pointers to constant text in the `.rdata` section. |
| `char` | 8-bit | AL/R-Byte | Single character. |

### Declaration Basics
```bcb
int32 age = 25;
float64 price = 99.99;
string label = "Inventory Item";

// Delayed Initialization using 'no_value' (initializes to zero)
int32 future_val = no_value;
```

### Public Variables (Global Scope)
BCB supports variables declared at the top-level of the program using the `pub` keyword. These variables are stored in the `.data` section and remain persistent throughout the program's execution.

**Syntax:**
```bcb
pub <type> <name> = <initializer>;
```

**Key Features:**
- **Persistent State**: Unlike stack variables, public variables live in the global data segment.
- **Complex Initializers**: Support for scalars, strings, arrays, enums, and structs.
- **Mutation**: Use `md` to modify global state from any function.
- **Shadowing**: Local variables can have the same name as global variables; the local version takes precedence within its scope.

**Example:**
```bcb
data {
    struct Config { int32 port; int32 debug; }
}

pub int32 max_retries = 3;
pub Config cfg = { port: int32 8080, debug: int32 1 };

export main(void) -> int32 {
    // Reading global
    int32 current = max_retries;
    
    // Writing global
    md int32 max_retries = 5;
    md int32 cfg.port = 443;
    
    return int32 0;
}
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

### Attributes System
BCB supports **attributes** that can be applied to functions, variables, and statements to modify compiler behavior. Attributes are specified using the `#` prefix.

#### Available Attributes

| Attribute | Description | Example |
| :--- | :--- | :--- |
| `#NoWarning("type")` | Suppresses specific warning types | `#NoWarning("unused variable")` |
| `#SonOf(parent)` | Creates hierarchical function/variable relationships | `#SonOf(math)` |

#### Multiple Attributes
You can apply multiple attributes to the same declaration using the `::` separator:

```bcb
// Apply multiple attributes to a function
#NoWarning("unused function")::#SonOf(math)
add(a: int32, b: int32) -> int32 {
    return int32 a + b;
}

// Apply multiple attributes to a global variable
#NoWarning("unused variable")::#SonOf(config)
pub int32 debug_level = 1;
```

The `::` separator allows you to chain attributes on a single line, making the code more compact and readable.

#### Suppressing Warnings
The `#NoWarning` attribute allows you to suppress specific warning types:

```bcb
// Suppress unused function warning
#NoWarning("unused function")
helper() -> int32 {
    return int32 42;
}

// Suppress unused variable warning
#NoWarning("unused variable")
int32 debug_flag = 1;

// Suppress unused parameter warning
process(data: int32, _unused: int32) -> void {
    // _unused parameter won't trigger warning
    return void;
}
```

**Supported Warning Types:**
- `"unused function"` - Suppresses warnings about functions that are never called
- `"unused variable"` - Suppresses warnings about variables that are never used
- `"unused parameter"` - Suppresses warnings about function parameters that are never used

**Alternative Suppression:**
Variables and parameters starting with underscore (`_`) are automatically excluded from unused warnings:
```bcb
process(data: int32, _scratch: int32) -> void {
    int32 _temp = 0;  // No warning
    return void;
}
```

#### SonOf Attribute (Hierarchical Organization)
The `#SonOf` attribute creates parent-child relationships between functions and variables, enabling namespace-like organization:

```bcb
// Parent function
math(void) -> void {
    call printf(string "Math module loaded\n");
    return void;
}

// Child functions (must be called as math.add, math.sub)
#SonOf(math)
add(a: int32, b: int32) -> int32 {
    return int32 a + b;
}

#SonOf(math)
sub(a: int32, b: int32) -> int32 {
    return int32 a - b;
}

// Child variable (accessed as math.x)
#SonOf(math)
pub int32 x = 10;

export main() -> int32 {
    // Call child functions with parent prefix
    int32 result = call math.add(int32 5, int32 3);
    
    // Access child variable
    md int32 math.x = 20;
    
    // Call parent function
    call math();
    
    return int32 0;
}
```

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

### Arrays inside Structs
Structs can contain fixed-size arrays as fields. These arrays are stored contiguously within the struct's memory layout.

**Declaration:**
```bcb
data {
    struct Player {
        int32 id;
        int32 scores[3]; // Fixed-size array inside struct
        char name[16];   // Fixed-size char array
    }
}
```

**Initialization & Access:**
```bcb
export main(void) -> int32 {
    // Initializing with an array literal
    Player p = { 
        id: int32 1, 
        scores: int32[] {10, 20, 30}, 
        name: char[] {'A', 'l', 'e', 'x'} 
    };
    
    // Accessing an array element within a struct
    int32 first_score = p.scores[0];
    
    // Modifying an element
    md int32 p.scores[1] = 50;
    
    // Using length() on struct array fields
    int32 num_scores = length(p.scores); // Returns 3
    
    return int32 0;
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

**Returning Lists from Functions:**
BCB supports functions that return arrays/lists of any type. When a function returns a list, it typically returns a pointer to the data start of a hidden-header-managed array.
```bcb
ret_list() -> int32[] {
    return int32[] { int32 1, int32 2, int32 3 };
}

export main() -> int32 {
    int32 my_list[3] = call ret_list();
    int32 val = my_list[1]; // 2
    return int32 0;
}
```
*Note: When assigning a returned list to a local array variable, the data is copied into the local variable's buffer.*

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

swap int64;        // Swaps the top two elements
dup int32;         // Duplicates the top element
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

## 11. Runtime Type Information & Variadic Arguments

BCB provides powerful runtime type introspection and variadic argument handling capabilities.

### The `gettype()` Function

`gettype()` is a built-in function that returns the type of a variable or expression as a string at runtime. This enables type-safe generic programming.

**Syntax:**
```bcb
string type_name = gettype(expression);
```

**Return Values:**
| Expression Type | Returns |
|:----------------|:--------|
| `int32` variable | `"int32"` |
| `int64` variable | `"int64"` |
| `string` variable | `"string"` |
| `float64` variable | `"float64"` |
| `float32` variable | `"float32"` |
| `int32[]` array | `"int32[]"` |
| `string[]` array | `"string[]"` |
| `float64[]` array | `"float64[]"` |
| `void` | `"void"` |

**Example:**
```bcb
export main() -> int32 {
    int32 a = 10;
    string type = gettype(a);       // type = "int32"
    
    float64 pi = 3.14159;
    string float_type = gettype(pi); // float_type = "float64"
    
    int32[] numbers = { 1, 2, 3 };
    string arr_type = gettype(numbers); // arr_type = "int32[]"
    
    return int32 0;
}
```

### Variadic Functions with `...args`

BCB supports variadic functions that can accept a variable number of arguments. Use `...args` as the parameter name to capture all additional arguments.

**Declaration:**
```bcb
print(myargs: ...args) -> void {
    // Function body can access all passed arguments
}
```

### `myargs.amount` - Argument Count

The `.amount` property returns the number of arguments passed to the variadic parameter.

**Syntax:**
```bcb
int32 count = myargs.amount;
```

**Example:**
```bcb
print_all(myargs: ...args) -> void {
    $if int32 myargs.amount == int32 0
        call printf(string "No arguments provided\n");
        return void;
    $endif
    
    call printf(string "Received %d arguments\n", int32 myargs.amount);
    return void;
}
```

### `myargs(index)` - Accessing Arguments by Index

Individual variadic arguments can be accessed using the `myargs(index)` syntax, where index is 0-based.

**Syntax:**
```bcb
// Access the first argument (index 0)
type myargs(0)

// Access the second argument (index 1)
type myargs(1)
```

**Important:** You must cast `myargs(index)` to the expected type when using it:
```bcb
int32 value = int32 myargs(0);      // Cast to int32
string text = string myargs(1);     // Cast to string
float64 num = float64 myargs(2);    // Cast to float64
```

### Complete Example: Type-Safe Generic Print Function

This example demonstrates combining `gettype()`, `myargs.amount`, and `myargs(index)` to create a generic print function:

```bcb
data {
    string fmt : "%s\n"
    string fmt_i : "%d\n"
    string fmt_f : "%f\n"
}

define printf(fmt: string, all: ...args) -> int32;
define strcmp(a: string, b: string) -> int32;

export main() -> int32 {
    call print(int32 10);
    call print(string "hello");
    call print(float64 10.5);
    call print(int32[] { 10, 20, 30 });
    call print();  // No arguments
    return int32 0;
}

print(myargs: ...args) -> void {
    // Handle no arguments case
    $if int32 myargs.amount == int32 0
        call printf(string fmt, string "No arguments");
        return void;
    $endif
    
    int32 i = 0;
    // Iterate through all arguments
    $while int32 i < int32 myargs.amount
        string type = gettype(myargs(i));
        
        $if int32 call strcmp(string type, string "int32") == int32 0
            call printf(string fmt_i, int32 myargs(i));
        $elseif int32 call strcmp(string type, string "string") == int32 0
            call printf(string fmt, string myargs(i));
        $elseif int32 call strcmp(string type, string "float64") == int32 0
            call printf(string fmt_f, float64 myargs(i));
        $elseif int32 call strcmp(string type, string "int32[]") == int32 0
            // Cast to array type for iteration
            int32[] arr = int32[](myargs(i));
            int32 j = 0;
            $while int32 j < int32 length(arr)
                call printf(string fmt_i, int32 arr[j]);
                md int32 j = j + 1;
            $endwhile
        $endif
        
        md int32 i = i + 1;
    $endwhile
    
    return void;
}
```

### Type Casting Variadic Arguments

When retrieving arguments from `myargs(index)`, cast to the appropriate type:

| Target Type | Cast Syntax |
|:------------|:------------|
| Scalar types | `int32 myargs(0)`, `float64 myargs(0)` |
| String | `string myargs(0)` |
| Array types | `int32[](myargs(0))`, `string[](myargs(0))` |

**Note:** Array types require parentheses around the cast: `type[](myargs(index))`.

---

## 15. Pointer Dereferencing Syntax
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

## 16. Compiler Internals & Win64 ABI
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

## 17. Macros (Metaprogramming)
Macros in BCB allow you to define reusable code snippets that are expanded at compile time (text substitution). This is powerful for reducing boilerplate.

**Definition:**
Use the `macro` keyword. Parameters are typed in the definition for clarity, but the types are currently soft-checked (duck typing) during expansion.
```bcb
macro print(msg: string) {
    call printf(string msg);
}

macro add(a: int32, b: int32) {
    a + b
}
```

**Usage:**
Macros can be used as statements or expressions.
```bcb
export main(void) -> int32 {
    print("Hello");  // Expands to: call printf(string "Hello");
    
    int32 x = 10;
    int32 y = 20;
    int32 res = add(x, y);  // Expands to: int32 res = x + y;
    
    return int32 0;
}
```

**Key Features:**
- **Text Substitution**: The body of the macro is injected directly into the code where it is called.
- **Statement & Expression Support**: Can be used as a standalone statement or part of an expression.
- **Parametrized**: Arguments passed to the macro replace the parameter names in the body.

---

## 18. Building and Testing
```bash
# Install the bcb command
pip install -e .

# Check installation
bcb --version
bcb --help

# Run the full test suite
python test_all.py

# Manual Compilation (with optimization)
bcb examples/math.bcb -o math.s -O3
gcc math.s -o math.exe
./math.exe

# View optimization stats
bcb examples/math.bcb -O3 --stats
```

---

**Master the machine. Control the stack. Build with BCB.**

*Crafted by Antigravity.*
