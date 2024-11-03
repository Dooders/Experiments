import unittest
import sys
import logging
from pathlib import Path

def run_tests():
    """Run all tests and return True if all tests pass."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent / 'tests'
    suite = loader.discover(start_dir)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1) 