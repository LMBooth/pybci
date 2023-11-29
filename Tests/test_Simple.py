from pybci import PyBCI
import time
from pybci.Configuration.EpochSettings import GlobalEpochSettings

gs = GlobalEpochSettings()
gs.tmax = 1 # grab 1 second after marker
gs.tmin = 0 # grab 0 seconds before marker
gs.splitCheck = True # splits samples between tmin and tmax
gs.windowLength = 0.5 # 
gs.windowOverlap = 0.5 # windows overap by 50%, so for a total len

def test_run_bci():
    bci = PyBCI(minimumEpochsRequired = 3, createPseudoDevice=True, globalEpochSettings=gs, loggingLevel = "TIMING")

    while not bci.connected:
        bci.Connect()
        time.sleep(1)
    bci.TrainMode()
    accuracy_achieved = False
    marker_received = False
    in_test_mode = False
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
                    bci.StopThreads()
                    bci.TestMode()
                    break
            #if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired+4:
            #    break
    while True:
        markerGuess = bci.CurrentClassifierMarkerGuess() # when in test mode only y_pred returned
        guess = [key for key, value in currentMarkers.items() if value[0] == markerGuess]
        in_test_mode = True
        time.sleep(1) 
        break
        #print("Current marker estimation: " + str(guess), end="           \r")
    assert accuracy_achieved and marker_received and in_test_mode
