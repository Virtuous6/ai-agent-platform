"""
Agents Test Suite

This directory contains all tests for the AI Agent Platform agents.

Test organization:
- Agent lifecycle and integration tests
- Individual agent component tests
- Platform integration tests
- Performance and cost optimization tests
"""

# Test discovery helper
def get_all_test_files():
    """Return list of all test files in this directory."""
    import os
    import glob
    
    test_dir = os.path.dirname(__file__)
    return glob.glob(os.path.join(test_dir, "test_*.py"))

# Common test utilities
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # Add agents parent to path 