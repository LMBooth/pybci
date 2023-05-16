import time
from pybci import PyBCI
bci = PyBCI()
while not bci.connected:
    bci.Connect()
    time.sleep(1)
bci.TrainMode()
while(True):
    currentMarkers = bci.ReceivedMarkerCount()
    time.sleep(1) # wait for marker updates
    print("Markers received: " + str(bci.ReceivedMarkerCount()), end="\r")
    if len(currentMarkers) > 1:  # check there is more then one marker type received
        if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired:
            bci.TestMode()
            break
try:
    while True:
        classInfo = bci.CurrentClassifierInfo() # when in train mode only y_pred returned
        guess = [key for key, value in currentMarkers.items() if value[0] == classInfo["y_pred"]]
        print("Current marker estimation: " + str(guess), end="\r")
except KeyboardInterrupt: # allow user to break while loop
    pass