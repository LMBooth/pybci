import time
from pybci import PyBCI
from pybci.Configuration.EpochSettings import GlobalEpochSettings

gs = GlobalEpochSettings()
gs.tmax = 1 # grab 1 second after marker
gs.tmin = 0 # grab 0 seconds before marker
gs.splitCheck = True # splits samples between tmin and tmax
gs.windowLength = 0.5 # 
gs.windowOverlap = 0.5 # windows overap by 50%, so for a total len

if __name__ == '__main__': # Note: this line is needed when calling pseudoDevice as by default runs in a multiprocessed operation
    bci = PyBCI(minimumEpochsRequired = 4, createPseudoDevice=True, globalEpochSettings=gs, loggingLevel = "TIMING")
    while not bci.connected: # check to see if lsl marker and datastream are available
        bci.Connect()
        time.sleep(1)
    bci.TrainMode() # now both marker and datastreams available start training on received epochs
    accuracy = 0
    try:
        while(True):
            currentMarkers = bci.ReceivedMarkerCount() # check to see how many received epochs, if markers sent to close together will be ignored till done processing
            time.sleep(0.5) # wait for marker updates
            print("Markers received: " + str(currentMarkers) +" Accuracy: " + str(round(accuracy,2)), end="         \r")
            if len(currentMarkers) > 1:  # check there is more then one marker type received
                if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired:
                    classInfo = bci.CurrentClassifierInfo() # hangs if called too early
                    accuracy = classInfo["accuracy"]
                if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired+10:  
                    bci.TestMode()
                    break
        while True:
            markerGuess = bci.CurrentClassifierMarkerGuess() # when in test mode only y_pred returned
            guess = [key for key, value in currentMarkers.items() if value[0] == markerGuess]
            print("Current marker estimation: " + str(guess), end="           \r")
            time.sleep(0.2)
    except KeyboardInterrupt: # allow user to break while loop
        print("\nLoop interrupted by user.")
        bci.StopThreads()
