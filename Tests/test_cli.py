
import subprocess
import time
import threading

def check_terminate(proc, timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    proc.terminate()

def test_cli_scripts():
    scripts = [
        'python /path/to/testPyTorch.py',
        'python /path/to/testSimple.py',
        'python /path/to/testSklearn.py',
        'python /path/to/testTensorflow.py',
        'python /path/to/cli.py'
    ]

    for script in scripts:
        print(f"Running {script}")
        proc = subprocess.Popen(script, shell=True)
        t = threading.Thread(target=check_terminate, args=(proc,))
        t.start()
        try:
            t.join()
        except KeyboardInterrupt:
            proc.terminate()
            break
