import sys
sys.path.insert(0, '/home/uvxiao/sharp/build/python_packages')

try:
    import sharp
    print('Sharp imported successfully')
    print('Has register_sharp_dialects:', hasattr(sharp, 'register_sharp_dialects'))
except Exception as e:
    print('Failed to import sharp:', e)
    import traceback
    traceback.print_exc()

try:
    import pysharp
    print('\nPySharp imported successfully')
    print('Has ir:', hasattr(pysharp, 'ir'))
    print('Has DefaultContext:', hasattr(pysharp, 'DefaultContext'))
    print('Has sharp:', hasattr(pysharp, 'sharp'))
    
    # Check what actually got imported
    print('\nPySharp attributes:', [x for x in dir(pysharp) if not x.startswith('_')])
except Exception as e:
    print('Failed to import pysharp:', e)
    import traceback
    traceback.print_exc()