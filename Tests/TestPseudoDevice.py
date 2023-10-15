import time
import pytest
from pybci.Utils.PseudoDevice import PseudoDeviceController
# Test case using the fixture
@pytest.mark.timeout(300)  # Extended timeout to 5 minutes
def test_run_pseudo():
    pseudoDevice = PseudoDeviceController()
    pseudoDevice.BeginStreaming()
    while True:
        time.sleep(1)
        if pseudoDevice.device.current_marker != None:
            assert True