import time
from pybci import PyBCI
import numpy as np
import tensorflow as tf# bring in tf for custom model creation
from pybci.Utils.FeatureExtractor import GenericFeatureExtractor
# Define our model, must be 1 dimesional at output to flatten due to multi-modal devices
pupilFeatures = 3
eegChs = 2 # Fp1, Fp2
eegfeaturesPerCh = 17
totalFeatures = (eegfeaturesPerCh * eegChs) + pupilFeatures
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(totalFeatures,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # 2 class output
])
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # 2 class output
    metrics=['accuracy']
)
class PupilGazeDecode():
    def __init__(self):
        super().__init__()
    def ProcessFeatures(self, epochData, sr, epochNum): # This is the required function name and variables that are passed to all 
        epochData = np.nan_to_num(epochData) # sklearn doesnt like nan
        #print(epochData.shape)
        if len(epochData[0]) ==  0:
            return [0,0,0]
        else:
            rightmean = np.mean(epochData[0]) # channel 20 is 3d pupil diameter right, get mean
            leftmean = np.mean(epochData[1]) # channel 21 is 3d pupil diameter right, get mean
            bothmean = np.mean([(epochData[0][i] + epochData[1][i]) / 2 for i in range(len(epochData[0]))]) # mean of both eyes in 3d
            return np.nan_to_num([rightmean,leftmean,bothmean, len(epochData[0])]) #  expects 2d

hullUniEEGLSLStreamName = "EEGStream"
pupilLabsLSLName = "pupil_capture" 
markerstream = "TestMarkers" # using pupillabs rightleftmarkers example
streamCustomFeatureExtract = {pupilLabsLSLName: PupilGazeDecode(), hullUniEEGLSLStreamName: GenericFeatureExtractor()}
dataStreamNames = [pupilLabsLSLName, hullUniEEGLSLStreamName]
# to reduce overall computational complexity we are going to drop irrelevant channels
streamChsDropDict = {hullUniEEGLSLStreamName : [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,19,20,21,22,23], # for our device we have Fp1 and Fp2 on channels 18 and 19, so list values 17 and 18 removed
                     pupilLabsLSLName: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] # pupil labs we only wan left and right 3d pupil diameter, drop rest
                     } 
bci = PyBCI(dataStreams = dataStreamNames, markerStream=markerstream, minimumEpochsRequired = 4,
            streamChsDropDict = streamChsDropDict,
            streamCustomFeatureExtract=streamCustomFeatureExtract ) #model = model, 

while not bci.connected:
    bci.Connect()
    time.sleep(1)
print(bci.markerStream.info().name())
bci.TrainMode()
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
            if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired+1:  
                bci.TestMode()
                break
    while True:
        markerGuess = bci.CurrentClassifierMarkerGuess() # when in test mode only y_pred returned
        guess = [key for key, value in currentMarkers.items() if value[0] == markerGuess]
        print("Current marker estimation: " + str(guess), end="\r")
        time.sleep(0.5)
except KeyboardInterrupt: # allow user to break while loop
    pass
