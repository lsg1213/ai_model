from glob import glob
import os
modules = []
for i in glob('data_utils/*.py'):
    name = os.path.splitext(os.path.basename(i))[0]
    if name != '__init__':
        modules.append(name)
__all__ = modules