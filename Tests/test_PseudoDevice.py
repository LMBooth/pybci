import time
from pybci.Utils.PseudoDevice import PseudoDeviceController

def test_run_pseudo():

    pd = PseudoDeviceController(is_multiprocessing=True)
    pd.BeginStreaming()
    time.sleep(5)
    pd.StopStreaming()
    time.sleep(5)

    pd = PseudoDeviceController(is_multiprocessing=False)
    pd.BeginStreaming()
    time.sleep(5)
    pd.StopStreaming()
    time.sleep(5)
