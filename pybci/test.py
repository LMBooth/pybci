import time, sys
sys.path.append('../')  # add the parent directory of 'utils' to sys.path, whilst in beta build.
from pybci import PyBCI

validMarkerStream = "UoHDataOffsetStream" #"Markers" "UoHDataOffsetStream"
validDataStreams = ["EEGStream"]#Liam's EMG device"]

bci = PyBCI(dataStreams = validDataStreams, markerStream = validMarkerStream)
streamChsDropDict = {"EEGStream": [19,20,21,22,23]}
bci.ConfigureDataStreamChannels(streamChsDropDict = streamChsDropDict)

while not bci.connected:
    bci.Connect()
    time.sleep(1)

bci.TrainMode()

# Need some way of relaying sufficient number of epochs ascertained in each version to notify testmode
#bci.TestMode()
