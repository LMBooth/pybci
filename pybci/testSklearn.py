import time, sys
sys.path.append('../')  # add the parent directory of 'utils' to sys.path, whilst in beta build.
from pybci import PyBCI
from Configuration.EpochSettings import GlobalEpochSettings

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(max_iter = 1000, solver ="lbfgs")#solver=clf, alpha=alpha,hidden_layer_sizes=hid)

num_chs = 8 # 8 channels re created in the PsuedoLSLGwnerator
num_feats = 17 # there are a total of 17 available features which are all enabled by default in Configurations.GeneralFeatureChoices (4 freq bands and 13 other metrics)
num_classes = 3 # number of different triggers (can include baseline) sent 

generalEpochSettings = GlobalEpochSettings() # get general epoch time window settings (found in Configuration.EpochSettings.GlobalEpochSettings)
generalEpochSettings.windowLength = 1 # == tmax+tmin if generalEpochSettings.splitcheck is False, splits specified epochs in customEpochSettings
bci = PyBCI(minimumEpochsRequired = 4, globalEpochSettings = generalEpochSettings, clf = clf)

while not bci.connected:
    bci.Connect()
    time.sleep(1)

bci.TrainMode()
while(True):
    currentMarkers = bci.ReceivedMarkerCount()
    print(currentMarkers)
    time.sleep(1)
    if len(currentMarkers) >= num_classes:   
        if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired + 3: # start training a bit after minimum required for training
            print("we're starting test mode!!")
            bci.TestMode()
            break
try:
    while True:
        # print current guess from pybci
        time.sleep(1)
except KeyboardInterrupt:
    pass
