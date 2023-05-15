import time, sys
sys.path.append('../')  # add the parent directory of 'utils' to sys.path, whilst in beta build.
from PyBCI.pybci import PyBCI
from PyBCI.Configuration.EpochSettings import GlobalEpochSettings, IndividualEpochSetting

bci = PyBCI() # create pybci object which auto scans for first available LSL marker and all accepted data streams

while not bci.connected: # check maker and data LSL streams available
    bci.Connect() # if not trr to reconnect
    time.sleep(1)

bci.TrainMode() # Now connected start bci training (defaults to sklearn SVM and all general feature settings, found in PyBCI.Configuration.FeatureSettings.GeneralFeatureChoices)
while(True):
    currentMarkers = bci.ReceivedMarkerCount() # gets current received training markers on marker stream
    time.sleep(1) # poll for 1 second
    
    if len(currentMarkers) > 1:   
        if min([currentMarkers[key][1] for key in currentMarkers]) > 10:
            bci.TestMode()
        break

while(True):
    # polls in testing
    time.sleep(1)
