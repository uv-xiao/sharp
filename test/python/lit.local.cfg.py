# Configuration for Python tests
import os
import sys

# Add build directory to Python path
config.environment['PYTHONPATH'] = os.pathsep.join([
    os.path.join(config.sharp_obj_root, 'python_packages'),
    os.path.join(config.sharp_obj_root, 'python_packages', 'pysharp'),
    config.environment.get('PYTHONPATH', '')
])

# Python tests use the RUN: marker
config.suffixes.add('.py')

# Exclude lit configuration files from being treated as tests
config.excludes = ['lit.cfg.py', 'lit.local.cfg.py', 'lit.site.cfg.py']