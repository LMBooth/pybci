from pybci.CliTests.testSimple import main as mainSimple
from pybci.CliTests.testSklearn import main as mainSklearn
from pybci.CliTests.testPyTorch import main as mainPyTorch
from pybci.CliTests.testTensorflow import main as mainTensorflow
from unittest.mock import patch

# Example usage
#def test_cli():
def test_cli_simple_timeout():
    with patch('builtins.input', return_value='stop'):
        timeout = 30  # timeout in seconds
        #def run_main():
        #    nonlocal my_bci_wrapper
        mainSimple(createPseudoDevice=True, min_epochs_train=1, min_epochs_test=2, timeout=timeout)

        #main_thread = threading.Thread(target=run_main)
        #main_thread.start()
        #main_thread.join()

def test_cli_sklearn_timeout():
    with patch('builtins.input', return_value='stop'):
        timeout = 30  # timeout in seconds
        #def run_main():
        #    nonlocal my_bci_wrapper
        mainSklearn(createPseudoDevice=True, min_epochs_train=1, min_epochs_test=2, timeout=timeout)

        #main_thread = threading.Thread(target=run_main)
        #main_thread.start()
        #main_thread.join()

def test_cli_pytorch_timeout():
    with patch('builtins.input', return_value='stop'):
        timeout = 30  # timeout in seconds

        #    nonlocal my_bci_wrapper
        mainPyTorch(createPseudoDevice=True, min_epochs_train=1, min_epochs_test=2, timeout=timeout)

        #main_thread = threading.Thread(target=run_main)
        #main_thread.start()
        #main_thread.join()

def test_cli_tensorflow_timeout():
    with patch('builtins.input', return_value='stop'):
        timeout = 30  # timeout in seconds
        mainTensorflow(createPseudoDevice=True, min_epochs_train=1, min_epochs_test=2, timeout=timeout)

        #main_thread = threading.Thread(target=run_main)
        #main_thread.start()
        #main_thread.join()