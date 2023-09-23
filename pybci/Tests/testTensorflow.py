import time, argparse
from pybci import PyBCI
from pybci.Configuration.EpochSettings import GlobalEpochSettings
import tensorflow as tf# bring in tf for custom model creation


def main(create_pseudo_device=True, min_epochs_train=4, min_epochs_test=10, num_chs = 8, num_feats = 2, num_classes = 4):
    if create_pseudo_device == True:
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
    bci = PyBCI(minimumEpochsRequired = min_epochs_train, createPseudoDevice=create_pseudo_device, model = model)
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
                if min([currentMarkers[key][1] for key in currentMarkers]) > min_epochs_test:  
                    bci.TestMode()
                    break
        while True:
            markerGuess = bci.CurrentClassifierMarkerGuess() # when in test mode only y_pred returned
            guess = [key for key, value in currentMarkers.items() if value[0] == markerGuess]
            print("Current marker estimation: " + str(guess), end="           \r")
            time.sleep(0.2)
    except KeyboardInterrupt: # allow user to break while loop
        print("\nLoop interrupted by user.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tensorflow GRU is used for model and pseudodevice generates 8 channels of 3 marker types and baseline. Similar to the testTensorflow.py in the examples folder.")
    parser.add_argument("--createPseudoDevice", default=True, type=bool, help="Set to True or False to enable or disable pseudo device creation.")
    parser.add_argument("--min_epochs_train", default=4, type=int, help='Minimum epochs to collect before model training commences, must be less than, min_epochs_test. If less than min_epochs_test defaults to min_epochs_test+1.')
    parser.add_argument("--min_epochs_test", default=10, type=int, help='Minimum epochs to collect before model testing commences, if less than min_epochs_test defaults to min_epochs_test+1.')
    parser.add_argument("--num_chs", default=8, type=int, help='Num of channels in data stream to configure tensorflow model, if PseudoDevice==True defaults to 8.')
    parser.add_argument("--num_classes", default=4, type=int, help='Num of classes in marker stream to configure tensorflow model, if PseudoDevice==True defaults to 4.')
    
    args = parser.parse_args()
    main(**vars(args))
