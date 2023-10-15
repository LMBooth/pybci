import time
import pytest
from pybci import PyBCI
from pybci.Utils.PseudoDevice import PseudoDeviceController
# Test case using the fixture
#@pytest.mark.timeout(300)  # Extended timeout to 5 minutes
def test_run_pseudo():
    bci = PyBCI(minimumEpochsRequired=5, createPseudoDevice=True)
    while not bci.connected:
        bci.Connect()
        time.sleep(1)
    bci.TrainMode()
    marker_received = False
    while True:
        time.sleep(1)
        currentMarkers = bci.ReceivedMarkerCount() # check to see how many received epochs, if markers sent to close together will be ignored till done processing
        if len(currentMarkers) > 1:
            bci.StopThreads()
            marker_received = True
            break
    assert marker_received