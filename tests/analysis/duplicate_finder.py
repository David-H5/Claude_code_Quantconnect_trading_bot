"""
Test Duplicate Finder

Utility to identify duplicate and redundant tests in the codebase.
Helps with test consolidation efforts.

UPGRADE-015: Test Analysis Tools
"""

import ast
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class TestInfo:
    """Information about a test function."""
    name: str
    file: Path
    line: int
    class_name: str | None
    docstring: str | None
    body_hash: str
    assertions: list[str] = field(default_factory=list)


@dataclass
class DuplicateGroup:
    """Group of potentially duplicate tests."""
    pattern: str
    tests: list[TestInfo]
    reason: str


class DuplicateFinder:
    """
    Find duplicate and redundant tests.

    Usage:
        finder = DuplicateFinder("tests/")
        duplicates = finder.find_all_duplicates()
        for group in duplicates:
            print(f"Pattern: {group.pattern}")
            for test in group.tests:
                print(f"  - {test.file}:{test.line} {test.name}")
    """

    # Common duplicate patterns
    DUPLICATE_PATTERNS = [
        r"test_to_dict",
        r"test_default_config",
        r"test_default_values",
        r"test_creation",
        r"test_create_\w+",
        r"test_factory",
        r"test_initialization",
        r"test_valid_input",
        r"test_invalid_input",
    ]

    def __init__(self, test_dir: str | Path = "tests"):
        self.test_dir = Path(test_dir)
        self.tests: list[TestInfo] = []

    def scan_tests(self) -> list[TestInfo]:
        """Scan all test files and extract test information."""
        self.tests = []

        for test_file in self._find_test_files():
            try:
                self.tests.extend(self._parse_test_file(test_file))
            except SyntaxError:
                continue

        return self.tests

    def _find_test_files(self) -> Iterator[Path]:
        """Find all test files."""
        for pattern in ["test_*.py", "*_test.py"]:
            yield from self.test_dir.rglob(pattern)

    def _parse_test_file(self, file_path: Path) -> list[TestInfo]:
        """Parse a test file and extract test functions."""
        tests = []
        content = file_path.read_text()
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Test class
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                        tests.append(self._extract_test_info(
                            item, file_path, class_name=node.name
                        ))
            elif isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                # Standalone test function
                tests.append(self._extract_test_info(node, file_path))

        return tests

    def _extract_test_info(
        self,
        node: ast.FunctionDef,
        file_path: Path,
        class_name: str | None = None,
    ) -> TestInfo:
        """Extract information from a test function AST node."""
        docstring = ast.get_docstring(node)
        body_hash = self._hash_body(node)
        assertions = self._extract_assertions(node)

        return TestInfo(
            name=node.name,
            file=file_path,
            line=node.lineno,
            class_name=class_name,
            docstring=docstring,
            body_hash=body_hash,
            assertions=assertions,
        )

    def _hash_body(self, node: ast.FunctionDef) -> str:
        """Create a hash of the function body for similarity detection."""
        # Simple hash based on structure
        body_str = ast.dump(node)
        return str(hash(body_str) % (10 ** 8))

    def _extract_assertions(self, node: ast.FunctionDef) -> list[str]:
        """Extract assertion statements from a test."""
        assertions = []
        for child in ast.walk(node):
            if isinstance(child, ast.Assert):
                assertions.append(ast.dump(child.test))
            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    if child.func.attr.startswith("assert"):
                        assertions.append(child.func.attr)
        return assertions

    def find_by_name_pattern(self, pattern: str) -> DuplicateGroup:
        """Find tests matching a name pattern."""
        regex = re.compile(pattern)
        matching = [t for t in self.tests if regex.search(t.name)]

        return DuplicateGroup(
            pattern=pattern,
            tests=matching,
            reason="Name pattern match",
        )

    def find_all_duplicates(self) -> list[DuplicateGroup]:
        """Find all potential duplicates."""
        if not self.tests:
            self.scan_tests()

        duplicates = []

        # Check each pattern
        for pattern in self.DUPLICATE_PATTERNS:
            group = self.find_by_name_pattern(pattern)
            if len(group.tests) > 1:
                duplicates.append(group)

        # Find tests with identical body hashes
        body_groups = defaultdict(list)
        for test in self.tests:
            body_groups[test.body_hash].append(test)

        for hash_val, tests in body_groups.items():
            if len(tests) > 1:
                duplicates.append(DuplicateGroup(
                    pattern=f"body_hash:{hash_val}",
                    tests=tests,
                    reason="Identical function body",
                ))

        return duplicates

    def find_assertion_free_tests(self) -> list[TestInfo]:
        """Find tests with no assertions."""
        if not self.tests:
            self.scan_tests()

        return [t for t in self.tests if not t.assertions]

    def generate_report(self) -> str:
        """Generate a duplicate analysis report."""
        if not self.tests:
            self.scan_tests()

        duplicates = self.find_all_duplicates()
        assertion_free = self.find_assertion_free_tests()

        report = []
        report.append("=" * 60)
        report.append("TEST DUPLICATE ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"\nTotal tests scanned: {len(self.tests)}")
        report.append(f"Duplicate groups found: {len(duplicates)}")
        report.append(f"Tests without assertions: {len(assertion_free)}")

        report.append("\n" + "-" * 60)
        report.append("DUPLICATE PATTERNS")
        report.append("-" * 60)

        for group in duplicates:
            if len(group.tests) > 3:  # Only show groups with 4+ duplicates
                report.append(f"\nPattern: {group.pattern}")
                report.append(f"Reason: {group.reason}")
                report.append(f"Count: {len(group.tests)}")
                for test in group.tests[:5]:
                    report.append(f"  - {test.file.name}:{test.line} {test.name}")
                if len(group.tests) > 5:
                    report.append(f"  ... and {len(group.tests) - 5} more")

        report.append("\n" + "-" * 60)
        report.append("CONSOLIDATION RECOMMENDATIONS")
        report.append("-" * 60)

        # Generate recommendations
        for group in duplicates:
            if "test_to_dict" in group.pattern:
                report.append(f"\n{group.pattern} ({len(group.tests)} tests):")
                report.append("  → Use assert_dataclass_to_dict() from conftest.py")
            elif "test_default" in group.pattern:
                report.append(f"\n{group.pattern} ({len(group.tests)} tests):")
                report.append("  → Use assert_config_defaults() from conftest.py")
            elif "test_create" in group.pattern or "test_factory" in group.pattern:
                report.append(f"\n{group.pattern} ({len(group.tests)} tests):")
                report.append("  → Use assert_factory_creates_valid() from conftest.py")

        return "\n".join(report)


def analyze_test_duplicates(test_dir: str = "tests") -> str:
    """Convenience function to analyze test duplicates."""
    finder = DuplicateFinder(test_dir)
    return finder.generate_report()


if __name__ == "__main__":
    print(analyze_test_duplicates())
