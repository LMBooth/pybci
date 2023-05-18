import os
import subprocess
import sys

def main():
    package_name = os.getenv('PYPI_PACKAGE_NAME', 'default-name')

    with open('setup.py', 'r') as f:
        setup_contents = f.read()

    setup_contents = setup_contents.replace("'default-name'", f"'{package_name}'")

    with open('temp_setup.py', 'w') as f:
        f.write(setup_contents)

    # Run the build command
    subprocess.check_call([sys.executable, '-m', 'build', '--setup-py', 'temp_setup.py'])

if __name__ == "__main__":
    main()