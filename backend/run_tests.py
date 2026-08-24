#!/usr/bin/env python3
"""
Test runner script to execute all unit tests for the backend.
Run with: python run_tests.py
"""
import subprocess
import sys
import os


def main():
    """Run all backend tests using pytest"""
    print("🧪 Running backend unit tests...\n")
    
    # Get project root (one level up from backend directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # Run pytest with coverage reporting (run backend's tests from project root)
    cmd = [
        sys.executable, "-m", "pytest",
        "backend/tests/",
        "-v",
        "--cov=backend/",
        "--cov-report=term-missing",
        "--cov-fail-under=45",
    ]
    
    try:
        completed = subprocess.run(cmd, check=False)
        if completed.returncode:
            print("\n❌ Some tests failed. Please check the output above.")
            return completed.returncode
        print("\n✅ All tests passed successfully!")
        return 0
    except OSError as e:
        print(f"\n❌ Could not start pytest: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
