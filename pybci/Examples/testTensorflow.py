import time, sys
sys.path.append('../')  # add the parent directory of 'utils' to sys.path, whilst in beta build.
from pybci import PyBCI
from pybci.Configuration.EpochSettings import GlobalEpochSettings
import tensorflow as tf# bring in tf for custom model creation

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

num_chs = 8 # 8 channels re created in the PsuedoLSLGwnerator
num_feats = 15 # there are a total of 17 available features which are all enabled by default in Configurations.GeneralFeatureChoices (4 freq bands and 13 other metrics)
num_classes = 3 # number of different triggers (can include baseline) sent 
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

generalEpochSettings = GlobalEpochSettings() # get general epoch time window settings (found in Configuration.EpochSettings.GlobalEpochSettings)
generalEpochSettings.windowLength = 1 # == tmax+tmin if generalEpochSettings.splitcheck is False, splits specified epochs in customEpochSettings

bci = PyBCI(minimumEpochsRequired = 4, globalEpochSettings = generalEpochSettings, model = model)

while not bci.connected:
    bci.Connect()
    time.sleep(1)

bci.TrainMode()
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
