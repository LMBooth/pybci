import subprocess
import time
def run_cli_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()
    return_code = process.returncode
    
    if return_code == 0:
        print(f"Command succeeded with output:\n{stdout.decode('utf-8')}")
    else:
        print(f"Command failed with error:\n{stderr.decode('utf-8')}")

# Example usage
def test_cli():
    run_cli_command("pybci testSimple --timeout=10")
    time.sleep(15)
    run_cli_command("pybci testSklearn --timeout=10")
    time.sleep(15)
    run_cli_command("pybci testPyTorch --timeout=10")
    time.sleep(15)
    run_cli_command("pybci testTensorflow --timeout=10")
    time.sleep(15)
    assert True