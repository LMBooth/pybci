import time
from pybci import PyBCI
import numpy as np
import tensorflow as tf# bring in tf for custom model creation

num_chs = 8 # 8 channels re created in the PsuedoLSLGwnerator
sum_samps = 250
num_classes = 3 # number of different triggers (can include baseline) sent 
model = tf.keras.Sequential()
model.add(tf.keras.layers.Reshape((num_chs,sum_samps, 1), input_shape=(num_chs,sum_samps)))
model.add(tf.keras.layers.Permute((2, 1, 3)))
model.add(tf.keras.layers.Reshape((num_chs*sum_samps, 1)))
model.add(tf.keras.layers.GRU(units=256))#, input_shape=num_chs*num_feats)) # maybe should show this example as 2d with toggleable timesteps disabled
model.add(tf.keras.layers.Dense(units=512, activation='relu'))
model.add(tf.keras.layers.Flatten())#   )tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax')) # softmax as more then binary classification (sparse_categorical_crossentropy)
#model.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # sigmoid as binary classification (binary_crossentropy)
model.summary()
model.compile(loss='sparse_categorical_crossentropy',# using sparse_categorical as we expect multi-class (>2) output, sparse because we encode targetvalues with integers
              optimizer='adam',
              metrics=['accuracy'])

class RawDecode():
    def ProcessFeatures(self, epochData, sr, epochNum): 
        #print(np.array(epochData).shape)
        return np.array(epochData) # tensorflow wants [1,chs,samps] for testing model

streamCustomFeatureExtract = {"sendTest" : RawDecode()} # we select EMG as that is the default type in the psuedolslgenerator example

bci = PyBCI(minimumEpochsRequired = 4, model = model, streamCustomFeatureExtract=streamCustomFeatureExtract )

while not bci.connected:
    bci.Connect()
    time.sleep(1)

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
