import os
import argparse


def parse_command_line():
    """
    parse command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-directory', '-b',
                        help    = "path to NonlinearRegression base directory",
                        default = "../",
                        type    = str,
                        dest    = "basedir")

    params = parser.parse_args()
    return params


# Get NonlinearRegression base directory
params  = parse_command_line()
basedir = params.basedir

# Make sure the site-packages modules load
VE_basedir = os.getenv('HOME') + '/noise_cancellation/lib/python2.7/site-packages/'
packages   = os.listdir(VE_basedir)
packages   = [p for p in packages if 'info' not in p
                                  if not p.startswith(('_', '.'))
                                  if not p.endswith(('pyc', 'pth', 'egg', 'so'))]

for index, mod in enumerate(packages):
    if mod.endswith('py'):
        mod = mod.split('.')[0]
        packages[index] = mod

packages = list(set(packages))

description = '''\'\'\'
This test module attempts to find all of the pip installed packages.
If an import fails, NonlinearRegression will likely fail as well. Be
sure to resolve all issues before moving on. Look in the site-packages
directory of the virtual environment for the package. It was probably
missed during the installation.
\'\'\'
'''

imports = '''import imp
import os\n\n
'''

block = '''
try:
    if os.path.isfile('{0}{1}/__init__.py'):
        imp.find_module('{1}')
except ImportError:
    print('{1} NOT FOUND')
    failed_imports += 1
'''

# Make sure the relative imports of NonlinearRegression load
tools_dir = basedir + '/NonlinearRegression/tools/'
modules   = [m.split('.')[0] for m in os.listdir(tools_dir)
             if not m.startswith('_') if not m.endswith('.pyc')]

repo_block = '''
try:
    main_info  = imp.find_module('NonlinearRegression')
    main       = imp.load_module('NonlinearRegression', *main_info)
    tools_info = imp.find_module('tools', main.__path__)
    tools      = imp.load_module('tools', *tools_info)
    helper     = imp.find_module('{0}', tools.__path__)
except ImportError:
    print('{0} NOT FOUND')
    failed_imports += 1
'''

end_block = '''
if failed_imports == 0:
    print('[+] All modules loaded correctly')
    print('[+] Relative paths set')
else:
    print('[-] WARNING: {} modules failed to load.'.format(failed_imports))
'''

with open(basedir + '/tests/test_imports.py', 'w') as ti:
    ti.write(description)
    ti.write(imports)
    ti.write('failed_imports = 0\n')
    ti.write("print('Checking imported modules...')\n")

    for p in packages:
        ti.write(block.format(VE_basedir, p))

    for mod in modules:
        ti.write(repo_block.format(mod))

    ti.write(end_block)
