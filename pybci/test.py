import time, sys
sys.path.append('../')  # add the parent directory of 'utils' to sys.path, whilst in beta build.
from pybci import PyBCI
from Configuration.EpochSettings import GlobalEpochSettings, IndividualEpochSetting

validMarkerStream = "UoHDataOffsetStream" # Sets target marker stream, currently only works with 1 target LSL marker stream
validDataStreams = ["EEGStream"] # give a defined list of valid datastream names, should probably add dict to allow people to pass custom classes for feature processing
# ConfigureDataStreamChannels
streamChsDropDict = {"EEGStream": [19,20,21,22,23]} # drop selected channel on defined streams
#ConfigureEpochWindowSettings
generalEpochSettings = GlobalEpochSettings() # get general epoch time window settings (found in Configuration.EpochSettings.GlobalEpochSettings)
generalEpochSettings.windowLength = 1 # == tmax+tmin if generalEpochSettings.splitcheck is False, splits specified epochs in customEpochSettings
ind = IndividualEpochSetting() # Sets individual time windows (found in Configuration.EpochSettings.IndividualEpochSetting)
ind.tmax = 1
ind.tmin = 0
baselineInd = IndividualEpochSetting() # Set individual time windows (found in Configuration.EpochSettings.IndividualEpochSetting)
baselineInd.tmax = 1 # 30 # slice needs adding before differing tmin+tmaxs can be used
baselineInd.tmin = 0
baselineInd.splitCheck = True
customEpochSettings = {"Baseline": baselineInd,
                "Dropped 1 samples": ind,
                "Dropped 2 samples": ind,}

bci = PyBCI( streamChsDropDict = streamChsDropDict, #dataStreams = validDataStreams, markerStream = validMarkerStream,
            minimumEpochsRequired = 3, globalEpochSettings = generalEpochSettings)#, customEpochSettings = customEpochSettings)
# minimumEpochsRequired set how many epochs are required before classificaiton attempted, if testtrainsplit (sklearn) contains only one target value classification is
#    def __init__(self, dataStreams = None, markerStream= None, streamTypes = None, markerTypes = None, printDebug = True,
#                 globalEpochSettings = GlobalEpochSettings(), customEpochSettings = {}, streamChsDropDict = {},
#                 freqbands = [[1.0, 4.0], [4.0, 8.0], [8.0, 12.0], [12.0, 20.0]], featureChoices = FeatureChoices ()):
# could add marker and dataoutlet timeouts as options 
while not bci.connected:
    bci.Connect()
    time.sleep(1)

bci.TrainMode()
while(True):
    currentMarkers = bci.ReceivedMarkerCount()
    print(currentMarkers)
    time.sleep(1)
    if len(currentMarkers) > 1:   
        print(currentMarkers)
        if min([currentMarkers[key][1] for key in currentMarkers]) > 5:
            print("we're starting test mode!!")
            bci.TestMode()
            break

try:
    while True:
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

