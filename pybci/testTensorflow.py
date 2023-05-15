import time, sys
sys.path.append('../')  # add the parent directory of 'utils' to sys.path, whilst in beta build.
from pybci import PyBCI
from Configuration.EpochSettings import GlobalEpochSettings
# bring in tf for custom model creation
import tensorflow as tf

num_chs = 8 # 8 channels re created in the PsuedoLSLGwnerator
num_feats = 17 # there are a total of 17 available features which are all enabled by default in Configurations.GeneralFeatureChoices (4 freq bands and 13 other metrics)
num_classes = 3 # number of different triggers (can include baseline) sent 
# Define the GRU model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Reshape((num_chs*num_feats, 1), input_shape=(num_chs*num_feats,)))
model.add(tf.keras.layers.GRU(units=256))#, input_shape=num_chs*num_feats)) # maybe should show this example as 2d with toggleable timesteps disabled
# Add additional layers if necessary
model.add(tf.keras.layers.Dense(units=512, activation='relu'))
#model.add(tf.keras.layers.Flatten())#   )tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax')) # softmax as more then binary classification
#model.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # sigmoid as ninary classification
model.summary()
model.compile(loss='categorical_crossentropy',#'binary_crossentropy',#categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

generalEpochSettings = GlobalEpochSettings() # get general epoch time window settings (found in Configuration.EpochSettings.GlobalEpochSettings)
generalEpochSettings.windowLength = 1 # == tmax+tmin if generalEpochSettings.splitcheck is False, splits specified epochs in customEpochSettings
bci = PyBCI(minimumEpochsRequired = 4, globalEpochSettings = generalEpochSettings, model = model)


while not bci.connected:
    bci.Connect()
    time.sleep(1)

bci.TrainMode()
while(True):
    currentMarkers = bci.ReceivedMarkerCount()
    print(currentMarkers)
    time.sleep(1)
    if len(currentMarkers) > num_classes:   
        print(currentMarkers)
        if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired + 3: # start training a bit after minimum required for training
            print("we're starting test mode!!")
            bci.TestMode()
            break
try:
    while True:
        # print current guess
        time.sleep(1)
except KeyboardInterrupt:
    pass
#### Other function examples
#bci.ConfigureDataStreamChannels(streamChsDropDict = streamChsDropDict)
#bci.ConfigureEpochWindowSettings(globalEpochSettings = generalEpochSettings, customEpochSettings = customEpochSettings)
#bci.ConfigureMachineLearning(clf = . model = model)
# need reuired num epochs setting example
# Need some way of relaying sufficient number of epochs ascertained in each version to notify testmode
#with PyBCI(dataStreams = validDataStreams, markerStream = validMarkerStream,streamChsDropDict = streamChsDropDict, 
#            globalEpochSettings = generalEpochSettings, customEpochSettings = wantedEpochs) as bci

