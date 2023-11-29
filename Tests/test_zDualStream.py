import time
from pybci import PyBCI
from pybci.Utils.PseudoDevice import PseudoDeviceController

def test_run_dual():

    pd1 = PseudoDeviceController(is_multiprocessing=True, dataStreamName="dev1")
    pd1.BeginStreaming()

    pd2 = PseudoDeviceController(is_multiprocessing=True, dataStreamName="dev2", createMarkers=False)
    pd2.BeginStreaming()
    time.sleep(5)

    bci = PyBCI(minimumEpochsRequired = 2, createPseudoDevice=False)

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
                if accuracy > 0:
                    # set to above 0 to show some accuracy was retruend from model
                    accuracy_achieved = True
                    
                    pd1.StopStreaming()
                    pd2.StopStreaming()
                    bci.StopThreads()
                    time.sleep(1)
                    break
            #if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired+4:
            #    break
    assert accuracy_achieved and marker_received
