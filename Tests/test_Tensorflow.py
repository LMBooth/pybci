from pybci import PyBCI
import time
import tensorflow as tf
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

#@pytest.mark.timeout(300)  # Extended timeout to 5 minutes
def test_run_bci():
    bci = PyBCI(minimumEpochsRequired = 3, model = model, createPseudoDevice=True)

    while not bci.connected:
        bci.Connect()
        time.sleep(1)
    bci.TrainMode()
    accuracy_achieved = False
    marker_received = False
    accuracy=None
    while True:
        currentMarkers = bci.ReceivedMarkerCount() # check to see how many received epochs, if markers sent to close together will be ignored till done processing
        time.sleep(0.5) # wait for marker updates
        #print("Markers received: " + str(currentMarkers) +" Accuracy: " + str(round(accuracy,2)), end="         \r")
        if len(currentMarkers) > 1:  # check there is more then one marker type received
            marker_received = True
            if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired:
                classInfo = bci.CurrentClassifierInfo() # hangs if called too early
                accuracy = classInfo["accuracy"]###
                if accuracy > 0:
                    # set to above 0 to show some accuracy was retruend from model
                    accuracy_achieved = True
                    bci.StopThreads()
                    break
            #if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired+4:
            #   break
    assert accuracy_achieved and marker_received
