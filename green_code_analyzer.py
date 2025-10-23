import ast
import re

class GreenCodeAnalyzer:
    def __init__(self, filename):
        self.filename = filename
        self.total_lines = 0
        self.comment_lines = 0
        self.blank_lines = 0
        self.function_count = 0
        self.class_count = 0
        self.complexity = 0

    def analyze(self):
        with open(self.filename, 'r', encoding='utf-8') as f:
            code = f.readlines()

        self.total_lines = len(code)

        for line in code:
            stripped = line.strip()
            if not stripped:
                self.blank_lines += 1
            elif stripped.startswith("#"):
                self.comment_lines += 1

        # Use AST to analyze structure
        with open(self.filename, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self.function_count += 1
                # Rough estimate of complexity: count control keywords
                self.complexity += sum(
                    isinstance(n, (ast.If, ast.For, ast.While, ast.Try)) for n in ast.walk(node)
                )
            elif isinstance(node, ast.ClassDef):
                self.class_count += 1

    def calculate_score(self):
        """Simple scoring logic for demonstration."""
        score = 100
        if self.total_lines > 300:
            score -= 10
        if self.comment_lines / (self.total_lines or 1) < 0.1:
            score -= 20
        if self.complexity > 20:
            score -= 20
        return max(score, 0)

    def report(self):
        print(f"📄 File: {self.filename}")
        print("=" * 40)
        print(f"Total Lines:      {self.total_lines}")
        print(f"Comment Lines:    {self.comment_lines}")
        print(f"Blank Lines:      {self.blank_lines}")
        print(f"Functions:        {self.function_count}")
        print(f"Classes:          {self.class_count}")
        print(f"Complexity Score: {self.complexity}")
        print(f"🟢 Green Code Score: {self.calculate_score()} / 100")
        print("=" * 40)


# Example Usage
if __name__ == "__main__":
    filename = input("Enter Python file to analyze: ")
    analyzer = GreenCodeAnalyzer(filename)
    analyzer.analyze()
    analyzer.report()
