import time
from pybci import PyBCI
from pybci.Utils.Logger import Logger
import numpy as np
from pybci.Utils.FeatureExtractor import GenericFeatureExtractor, GeneralFeatureChoices

#dropchs = range(20,60) #streamChsDropDict={"sendTest":dropchs}, 
featureChoices = GeneralFeatureChoices()
featureChoices.psdBand = False
featureChoices.meanPSD = False
featureChoices.slopeSignChange = True
featureChoices.waveformLength = True

featureChoices.zeroCross = True

streamCustomFeatureExtract = {"sendTest": GenericFeatureExtractor(featureChoices=featureChoices)}

bci = PyBCI(minimumEpochsRequired = 4, streamCustomFeatureExtract = streamCustomFeatureExtract)#, streamChsDropDict={"sendTest":dropchs})#, loggingLevel = Logger.NONE)
while not bci.connected: # check to see if lsl marker and datastream are available
    bci.Connect()
    time.sleep(1)
bci.TrainMode() # now both marker and datastreams available start training on received epochs
accuracy = 0
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
            if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired+10:  
                bci.TestMode()
                break
    while True:
        markerGuess = bci.CurrentClassifierMarkerGuess() # when in test mode only y_pred returned
        guess = [key for key, value in currentMarkers.items() if value[0] == markerGuess]
        print("Current marker estimation: " + str(guess), end="\r")
        time.sleep(0.1)
except KeyboardInterrupt: # allow user to break while loop
    pass