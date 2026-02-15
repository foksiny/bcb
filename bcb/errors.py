import sys

class DiagnosticLevel:
    ERROR = "ERROR"
    WARNING = "WARNING"
    TIP = "TIP"
    PRE = "POSSIBLE RUNTIME ERROR"

class Diagnostic:
    def __init__(self, level, message, line=0, column=0, hint=None, source_file=None):
        self.level = level
        self.message = message
        self.line = line
        self.column = column
        self.hint = hint
        self.source_file = source_file

class ErrorManager:
    def __init__(self, source_code="", filename="<unknown>"):
        self.diagnostics = []
        self.source_files = {}  # Map of filename -> source lines
        self.source_files[filename] = source_code.splitlines() if source_code else []
        self.filename = filename
        self.has_error = False
    
    def add_source_file(self, filename, source_code):
        """Add a source file for error display."""
        if filename not in self.source_files:
            self.source_files[filename] = source_code.splitlines() if source_code else []

    def add(self, level, message, line=0, column=0, hint=None, source_file=None):
        self.diagnostics.append(Diagnostic(level, message, line, column, hint, source_file))
        if level == DiagnosticLevel.ERROR:
            self.has_error = True

    def error(self, message, line=0, column=0, hint=None, source_file=None):
        self.add(DiagnosticLevel.ERROR, message, line, column, hint, source_file)

    def warning(self, message, line=0, column=0, hint=None, source_file=None):
        self.add(DiagnosticLevel.WARNING, message, line, column, hint, source_file)

    def tip(self, message, line=0, column=0, hint=None, source_file=None):
        self.add(DiagnosticLevel.TIP, message, line, column, hint, source_file)

    def pre(self, message, line=0, column=0, hint=None, source_file=None):
        self.add(DiagnosticLevel.PRE, message, line, column, hint, source_file)

    def print_diagnostics(self):
        # ANSI colors
        RESET = "\033[0m"
        RED = "\033[91m"     # Errors
        YELLOW = "\033[93m"  # Warnings
        BLUE = "\033[94m"    # Tips
        MAGENTA = "\033[95m" # PREs
        BOLD = "\033[1m"
        WHITE = "\033[97m"

        for diag in self.diagnostics:
            color = WHITE
            label = diag.level
            
            if diag.level == DiagnosticLevel.ERROR:
                color = RED
            elif diag.level == DiagnosticLevel.WARNING:
                color = YELLOW
            elif diag.level == DiagnosticLevel.TIP:
                color = BLUE
            elif diag.level == DiagnosticLevel.PRE:
                color = MAGENTA
            
            # Print location Header
            filename = diag.source_file if diag.source_file else self.filename
            print(f"{BOLD}{filename}:{diag.line}:{diag.column}: {color}{label}: {diag.message}{RESET}")
            
            # Print source snippet - look up the correct file's source
            source_lines = self.source_files.get(filename, [])
            if diag.line > 0 and diag.line <= len(source_lines):
                line_content = source_lines[diag.line - 1]
                print(f"  {line_content}")
                # Print caret
                caret_trace = " " * diag.column + "^"
                print(f"  {color}{caret_trace}{RESET}")
            
            if diag.hint:
                print(f"{BLUE}  Tip: {diag.hint}{RESET}")
            print()
