import time
from pybci import PyBCI
from pybci.Configuration.EpochSettings import GlobalEpochSettings, IndividualEpochSetting

gs = GlobalEpochSettings()
gs.tmax = 1 # grab 1 second after marker
gs.tmin = 0.5 # grab 0.2 seconds before marker
gs.splitCheck = False # splits samples between tmin and tmax
gs.windowLength = 0.6 # 
gs.windowOverlap = 0.5 # windows overap by 50%, so for a total len

bci = PyBCI(globalEpochSettings=gs, minimumEpochsRequired=4) # create pybci object which auto scans for first available LSL marker and all accepted data streams

while not bci.connected: # check maker and data LSL streams available
    bci.Connect() # if not trr to reconnect
    time.sleep(1)

bci.TrainMode() # Now connected start bci training (defaults to sklearn SVM and all general feature settings, found in PyBCI.Configuration.FeatureSettings.GeneralFeatureChoices)
while(True):
    currentMarkers = bci.ReceivedMarkerCount()
    time.sleep(1) # wait for marker updates
    print("Markers received: " + str(bci.ReceivedMarkerCount()), end="\r")
    if len(currentMarkers) > 1:  # check there is more then one marker type received
        if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired:
            classInfo = bci.CurrentClassifierInfo() 
            print("\n Class accuracy: " + str(classInfo["accuracy"]) + " ")
        if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired+3:  
            print("we trying?")
            bci.TestMode()
            break
try:
    while True:
        time.sleep(1)
        classInfo = bci.CurrentClassifierInfo() # when in train mode only y_pred returned
        guess = [key for key, value in currentMarkers.items() if value[0] == classInfo["y_pred"]]
        print("Current marker estimation: " + str(guess), end="\r")
except KeyboardInterrupt: # allow user to break while loop
    pass