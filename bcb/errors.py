import sys

class DiagnosticLevel:
    ERROR = "ERROR"
    WARNING = "WARNING"
    TIP = "TIP"
    PRE = "POSSIBLE RUNTIME ERROR"

class Diagnostic:
    def __init__(self, level, message, line=0, column=0, hint=None):
        self.level = level
        self.message = message
        self.line = line
        self.column = column
        self.hint = hint

class ErrorManager:
    def __init__(self, source_code="", filename="<unknown>"):
        self.diagnostics = []
        self.source_code = source_code.splitlines()
        self.filename = filename
        self.has_error = False

    def add(self, level, message, line=0, column=0, hint=None):
        self.diagnostics.append(Diagnostic(level, message, line, column, hint))
        if level == DiagnosticLevel.ERROR:
            self.has_error = True

    def error(self, message, line=0, column=0, hint=None):
        self.add(DiagnosticLevel.ERROR, message, line, column, hint)

    def warning(self, message, line=0, column=0, hint=None):
        self.add(DiagnosticLevel.WARNING, message, line, column, hint)

    def tip(self, message, line=0, column=0, hint=None):
        self.add(DiagnosticLevel.TIP, message, line, column, hint)

    def pre(self, message, line=0, column=0, hint=None):
        self.add(DiagnosticLevel.PRE, message, line, column, hint)

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
            print(f"{BOLD}{self.filename}:{diag.line}:{diag.column}: {color}{label}: {diag.message}{RESET}")
            
            # Print source snippet
            if diag.line > 0 and diag.line <= len(self.source_code):
                line_content = self.source_code[diag.line - 1]
                print(f"  {line_content}")
                # Print caret
                caret_trace = " " * diag.column + "^"
                print(f"  {color}{caret_trace}{RESET}")
            
            if diag.hint:
                print(f"{BLUE}  Tip: {diag.hint}{RESET}")
            print()
