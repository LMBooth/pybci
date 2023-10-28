import time
from pybci.Utils.PseudoDevice import PseudoDeviceController

def test_run_pseudo():

    pd = PseudoDeviceController(execution_mode="thread")
    pd.BeginStreaming()
    time.sleep(5)
    pd.StopStreaming()
    time.sleep(5)

    pd = PseudoDeviceController(execution_mode="process")
    pd.BeginStreaming()
    time.sleep(5)
    pd.StopStreaming()
    time.sleep(5)
