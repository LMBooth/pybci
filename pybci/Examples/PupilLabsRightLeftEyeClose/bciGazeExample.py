import time
from pybci import PyBCI
import numpy as np

class PupilGazeDecode():
    def __init__(self):
        super().__init__()
    def ProcessFeatures(self, epochData, sr, epochNum): # This is the required function name and variables that are passed to all 
        #print(epochData.shape)
        epochData = np.nan_to_num(epochData) # sklearn doesnt like nan
        if epochData.shape[0] ==  0:
            return np.array([0,0,0])
        else:
            rightmean = np.mean(epochData[:,20]) # channel 20 is 3d pupil diameter right, get mean
            leftmean = np.mean(epochData[:,21]) # channel 21 is 3d pupil diameter right, get mean
            bothmean = np.mean([(epochData[:,20][i] + epochData[:,21][i]) / 2 for i in range(len(epochData[:,20]))]) # mean of both eyes in 3d
            #print(np.nan_to_num([rightmean,leftmean,bothmean]))
            return np.nan_to_num([rightmean,leftmean,bothmean]) #  expects 1d
    
streamCustomFeatureExtract = {"pupil_capture" : PupilGazeDecode()}
dataStreamName = ["pupil_capture"]
# Recommended to probably drop alot of channels unused channels as Asynchronous data streams like pupil-gazr can be computationally more expensive then synchrnous slicing,
# if finding performance issues or markers not received  consider add to streamChsDropDict = {"pupil_capture" = range(20)} to PyBCI intiialise, then changeephcData[20] and [21] to 0 and 1 in PupilGazeDecode()
bci = PyBCI(dataStreams = dataStreamName,markerStream="TestMarkers", minimumEpochsRequired = 4, streamCustomFeatureExtract=streamCustomFeatureExtract)
while not bci.connected: # check to see if lsl marker and datastream are available
    bci.Connect()
    time.sleep(1)
bci.TrainMode() # now both marker and datastreams available start training on received epochs
accuracy = 0
try:
    while(True):
        currentMarkers = bci.ReceivedMarkerCount() # check to see how many received epochs, if markers sent to close together will be ignored till done processing
        time.sleep(1) # wait for marker updates
        print("Markers received: " + str(currentMarkers) +" Class accuracy: " + str(accuracy), end="\r")
        if len(currentMarkers) > 1:  # check there is more then one marker type received
            if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired:
                classInfo = bci.CurrentClassifierInfo() # hangs if called too early
                accuracy = classInfo["accuracy"]
            if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired+12:  
                bci.TestMode()
                break
    while True:
        markerGuess = bci.CurrentClassifierMarkerGuess() # when in test mode only y_pred returned
        guess = [key for key, value in currentMarkers.items() if value[0] == markerGuess]
        print("Current marker estimation: " + str(guess), end="\r")
        time.sleep(0.5)
except KeyboardInterrupt: # allow user to break while loop
    pass
