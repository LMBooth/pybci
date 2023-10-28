from pybci import PyBCI, get_os
import time
from pybci.Utils.PseudoDevice import PseudoDeviceController

def test_run_bci():
    current_os = get_os()
    if current_os == "Windows":
        bci = PyBCI(minimumEpochsRequired = 3, createPseudoDevice=True)
    else:
        pdc = PseudoDeviceController(execution_mode="process")
        pdc.BeginStreaming()
        time.sleep(10)
        bci = PyBCI(minimumEpochsRequired = 3, createPseudoDevice=True, pseudoDeviceController=pdc)
    while not bci.connected:
        bci.Connect()
        time.sleep(1)
    bci.TrainMode()
    accuracy_achieved = False
    marker_received = False
    accuracy=None
    while True:
        currentMarkers = bci.ReceivedMarkerCount() # check to see how many received epochs, if markers sent to close together will be ignored till done processing
        time.sleep(0.5) # wait for marker updates
        #print("Markers received: " + str(currentMarkers) +" Accuracy: " + str(round(accuracy,2)), end="         \r")
        if len(currentMarkers) > 1:  # check there is more then one marker type received
            marker_received = True
            if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired:
                classInfo = bci.CurrentClassifierInfo() # hangs if called too early
                accuracy = classInfo["accuracy"]###
                if accuracy >= 0:
                    # set to above 0 to show some accuracy was retruend from model
                    accuracy_achieved = True
                    bci.StopThreads()
                    break
            #if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired+4:
            #    break
    assert accuracy_achieved and marker_received