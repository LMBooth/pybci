from pybci import PyBCI
import time
import pytest

# Test case using the fixture
def test_run_bci():
    bci = PyBCI(minimumEpochsRequired=5, createPseudoDevice=True)
    while not bci.connected:
        bci.Connect()
        time.sleep(1)
    bci.TrainMode()
    accuracy_achieved = False
    marker_received = False
    accuracy=0
    while True:
        currentMarkers = bci.ReceivedMarkerCount() # check to see how many received epochs, if markers sent to close together will be ignored till done processing
        time.sleep(0.5) # wait for marker updates
        print("Markers received: " + str(currentMarkers) +" Accuracy: " + str(round(accuracy,2)), end="         \r")
        if len(currentMarkers) > 1:  # check there is more then one marker type received
            marker_received = True
            if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired:
                classInfo = bci.CurrentClassifierInfo() # hangs if called too early
                accuracy = classInfo["accuracy"]###
                if accuracy > 0.75:
                    accuracy_achieved = True
                    bci.StopThreads()
                    break
            if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired+4:
                break
    assert accuracy_achieved and marker_received