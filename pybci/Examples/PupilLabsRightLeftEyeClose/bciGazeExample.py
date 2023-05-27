import time
from pybci import PyBCI
import numpy as np
import tensorflow as tf
num_chs = 1 # 8 channels re created in the PsuedoLSLGwnerator
num_feats = 3 # there are a total of 17 available features which are all enabled by default in Configurations.GeneralFeatureChoices (4 freq bands and 13 other metrics)
num_classes = 3 # number of different triggers (can include baseline) sent 
# Define the GRU model
model = tf.keras.Sequential()
#model.add(tf.keras.layers.Reshape((num_chs*num_feats, 1), input_shape=(num_chs*num_feats,)))
#model.add(tf.keras.layers.GRU(units=256))#, input_shape=num_chs*num_feats)) # maybe should show this example as 2d with toggleable timesteps disabled
input_shape = (None, 3)  # assuming the input is (number_of_timestamps, number_of_features)
n_units = 32 
model.add(tf.keras.layers.Reshape((num_chs*num_feats, 1), input_shape=(num_chs*num_feats,)))
model.add(tf.keras.layers.GRU(n_units))#, input_shape=input_shape))
#model.add(tf.keras.layers.GRU((32), input_shape=(3,)))
#model.add(tf.keras.layers.Dense(units=512, activation='relu'))
#model.add(tf.keras.layers.Flatten())#   )tf.keras.layers.Dense(units=128, activation='relu'))
#model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax')) # softmax as more then binary classification (sparse_categorical_crossentropy)
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # sigmoid as ninary classification (binary_crossentropy)
model.summary()
model.compile(loss='binary_crossentropy',# using sparse_categorical as we expect multi-class (>2) output, sparse because we encode targetvalues with integers
              optimizer='adam',
              metrics=['accuracy'])

class PupilGazeDecode():
    def __init__(self):
        super().__init__()
    def ProcessFeatures(self, epochData, sr, epochNum): # This is the required function name and variables that are passed to all 
        epochData = np.nan_to_num(epochData) # sklearn doesnt like nan
        if len(epochData[0]) ==  0:
            return np.array([0,0,0])
        else:
            rightmean = np.mean(epochData[20]) # channel 20 is 3d pupil diameter right, get mean
            leftmean = np.mean(epochData[21]) # channel 21 is 3d pupil diameter right, get mean
            bothmean = np.mean([(epochData[20][i] + epochData[21][i]) / 2 for i in range(len(epochData[20]))]) # mean of both eyes in 3d
            #print(np.nan_to_num([rightmean,leftmean,bothmean]))
            return np.nan_to_num([rightmean,leftmean,bothmean]) #  expects 2d
    
streamCustomFeatureExtract = {"pupil_capture" : PupilGazeDecode()}
dataStreamName = ["pupil_capture"]
# Recommended to probably drop alot of channels unused channels as Asynchronous data streams like pupil-gazr can be computationally more expensive then synchrnous slicing,
# if finding performance issues or markers not received  consider add to streamChsDropDict = {"pupil_capture" = range(20)} to PyBCI intiialise, then changeephcData[20] and [21] to 0 and 1 in PupilGazeDecode()
bci = PyBCI(dataStreams = dataStreamName, minimumEpochsRequired = 4, streamCustomFeatureExtract=streamCustomFeatureExtract, model = model)
while not bci.connected: # check to see if lsl marker and datastream are available
    bci.Connect()
    time.sleep(1)
bci.TrainMode() # now both marker and datastreams available start training on received epochs
accuracy = 0
try:
    while(True):
        currentMarkers = bci.ReceivedMarkerCount() # check to see how many received epochs, if markers sent to close together will be ignored till done processing
        time.sleep(1) # wait for marker updates
        print("Markers received: " + str(currentMarkers) +" Class accuracy: " + str(accuracy))#, end="\r")
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
        print("Current marker estimation: " + str(guess))#, end="\r")
        time.sleep(0.5)
except KeyboardInterrupt: # allow user to break while loop
    pass
