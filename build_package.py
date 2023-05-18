import os
import subprocess
import sys
import shutil

def main():
    package_name = os.getenv('PYPI_PACKAGE_NAME', 'default-name')
    os.environ['PYPI_PACKAGE_NAME'] = package_name

    # Copy the temp_setup.py to setup.py
    shutil.copyfile('temp_setup.py', 'setup.py')

    # Run the build command
    subprocess.check_call([sys.executable, '-m', 'build'])

if __name__ == "__main__":
    main()
