import time
from pybci.CliTests.testSimple import main as mainSimple
from pybci.CliTests.testSklearn import main as mainSklearn
from pybci.CliTests.testPyTorch import main as mainPyTorch
from pybci.CliTests.testTensorflow import main as mainTensorflow


# Example usage
def test_cli():
    mainSimple(min_epochs_train=1, min_epochs_test=2, timeout=10)
    time.sleep(15)
    mainSklearn(min_epochs_train=1, min_epochs_test=2, timeout=10)
    time.sleep(15)
    mainPyTorch(min_epochs_train=1, min_epochs_test=2, timeout=10)
    time.sleep(15)
    mainTensorflow(min_epochs_train=1, min_epochs_test=2, timeout=10)
    time.sleep(15)
    assert True
    