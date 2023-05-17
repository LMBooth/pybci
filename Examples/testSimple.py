import time
from pybci import PyBCI
bci = PyBCI()
while not bci.connected: # check to see if lsl marker and datastream are available
    bci.Connect()
    time.sleep(1)
bci.TrainMode() # now both marker and datastreams available start training on received epochs
while(True):
    currentMarkers = bci.ReceivedMarkerCount() # check to see how many received epochs, if markers sent to close together will be ignored till done processing
    time.sleep(1) # wait for marker updates
    print("Markers received: " + str(bci.ReceivedMarkerCount()), end="\r")
    if len(currentMarkers) > 1:  # check there is more then one marker type received
        if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired: # enough marker received, start testing!
            bci.TestMode()
            break
print("Started Testing")
try:
    while True:
        classInfo = bci.CurrentClassifierInfo() # when in test mode only y_pred returned
        guess = [key for key, value in currentMarkers.items() if value[0] == classInfo["y_pred"]]
        print("Current marker estimation: " + str(guess), end="\r")
except KeyboardInterrupt: # allow user to break while loop
    pass