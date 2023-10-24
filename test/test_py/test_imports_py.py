# test_imports.py

import unittest

# Try importing the modules
class TestImports(unittest.TestCase):

    def test_geometry_utils_import(self):
        try:
            import avatar_behavior_cloning.utils.geometry_utils
        except ImportError:
            self.fail("Failed to import geometry_utils")

    def test_drake_utils_import(self):
        try:
            import avatar_behavior_cloning.utils.drake_utils 
        except ImportError:
            self.fail("Failed to import drake_utils")

unittest.main()
