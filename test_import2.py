import sys
sys.path.insert(0, '/home/uvxiao/sharp/build/python_packages')

# Import pysharp's __init__ directly to see errors
import importlib.util
spec = importlib.util.spec_from_file_location("pysharp", "/home/uvxiao/sharp/build/python_packages/pysharp/pysharp/__init__.py")
pysharp = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(pysharp)
    print("PySharp loaded successfully")
except Exception as e:
    print("Error loading PySharp:", e)
    import traceback
    traceback.print_exc()