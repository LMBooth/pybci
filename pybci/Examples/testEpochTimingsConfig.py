import time
from pybci import PyBCI
from pybci.Configuration.EpochSettings import GlobalEpochSettings, IndividualEpochSetting

gs = GlobalEpochSettings()
gs.tmax = 0.5 # grab 1 second after marker
gs.tmin = 0.5 # grab 0.5 seconds before marker
gs.splitCheck = True # splits samples between tmin and tmax
gs.windowLength = 1 # 
gs.windowOverlap = 0.5 # windows overap by 50%, so for a total len

bci = PyBCI(globalEpochSettings=gs, minimumEpochsRequired=4) # create pybci object which auto scans for first available LSL marker and all accepted data streams

while not bci.connected: # check maker and data LSL streams available
    bci.Connect() # if not trr to reconnect
    time.sleep(1)
accuracy = 0
bci.TrainMode() # Now connected start bci training (defaults to sklearn SVM and all general feature settings, found in PyBCI.Configuration.FeatureSettings.GeneralFeatureChoices)
try:
    while(True):
        currentMarkers = bci.ReceivedMarkerCount() # check to see how many received epochs, if markers sent to close together will be ignored till done processing
        time.sleep(0.1) # wait for marker updates
        print("Markers received: " + str(currentMarkers) +" Class accuracy: " + str(accuracy), end="\r")
        if len(currentMarkers) > 1:  # check there is more then one marker type received
            #print(bci.CurrentFeaturesTargets())
            if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired:
                classInfo = bci.CurrentClassifierInfo() # hangs if called too early
                accuracy = classInfo["accuracy"]
            if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired+2:  
                bci.TestMode()
                break
    while True:
        markerGuess = bci.CurrentClassifierMarkerGuess() # when in test mode only y_pred returned
        guess = [key for key, value in currentMarkers.items() if value[0] == markerGuess]
        print("Current marker estimation: " + str(guess), end="\r")
        time.sleep(0.1)
except KeyboardInterrupt: # allow user to break while loop
    pass