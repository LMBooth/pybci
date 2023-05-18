import os
import subprocess
import sys

def main():
    package_name = os.getenv('PYPI_PACKAGE_NAME', 'pybci')
    os.environ['PYPI_PACKAGE_NAME'] = package_name

    # Run the build command
    subprocess.check_call([sys.executable, '-m', 'build'])

if __name__ == "__main__":
    main()