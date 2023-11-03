import time
from pybci import PyBCI
import tensorflow as tf# bring in tf for custom model creation
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
num_chs = 8 # 8 channels are created in the PseudoLSLGenerator
num_feats = 2 # default is mean freq and rms to keep it simple
num_classes = 4 # number of different triggers (can include baseline) sent, defines if we use softmax of binary
# Define the GRU model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Reshape((num_chs*num_feats, 1), input_shape=(num_chs*num_feats,)))
model.add(tf.keras.layers.GRU(units=256))#, input_shape=num_chs*num_feats)) # maybe should show this example as 2d with toggleable timesteps disabled
model.add(tf.keras.layers.Dense(units=512, activation='relu'))
model.add(tf.keras.layers.Flatten())#   )tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax')) # softmax as more then binary classification (sparse_categorical_crossentropy)
#model.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # sigmoid as ninary classification (binary_crossentropy)
model.summary()
model.compile(loss='sparse_categorical_crossentropy',# using sparse_categorical as we expect multi-class (>2) output, sparse because we encode targetvalues with integers
              optimizer='adam',
              metrics=['accuracy'])

if __name__ == '__main__': # Note: this line is needed when calling pseudoDevice as by default runs in a multiprocessed operation
    bci = PyBCI(minimumEpochsRequired = 4, createPseudoDevice=True, model = model)
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

