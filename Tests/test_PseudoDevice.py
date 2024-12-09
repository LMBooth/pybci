import time
from pybci.Utils.PseudoDevice import PseudoDeviceController

def test_run_pseudo():
    expected_duration = "~40 minutes"  # Adjust this to your expected time
    print(f"\n\n=== WARNING: The tests are expected to take {expected_duration} total! ===\n")
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
