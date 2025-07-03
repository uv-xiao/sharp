#!/usr/bin/env python3
"""Simple test to verify Sharp Python bindings work."""

import sys
import os

# Add Sharp to path
sys.path.insert(0, "/home/uvxiao/sharp/build/python_packages")

try:
    # Import Sharp
    import sharp
    print("✓ Successfully imported Sharp")
    
    # Check what's available
    print("\nSharp module contents:")
    print(dir(sharp))
    
    print("\nSharp._sharp contents:")
    print(dir(sharp._sharp))
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)