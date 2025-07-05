import sys
sys.path.insert(0, '/home/uvxiao/sharp/build/python_packages')

import sharp
print("Sharp imported successfully")
print("Sharp attributes:", [x for x in dir(sharp) if not x.startswith('_')])
print("\nChecking _sharp module:")
print("_sharp attributes:", [x for x in dir(sharp._sharp) if not x.startswith('_')])