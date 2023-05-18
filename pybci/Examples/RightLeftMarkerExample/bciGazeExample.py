import time
from pybci import PyBCI
import numpy as np

bci = PyBCI(minimumEpochsRequired = 4)

class CustomChoices:
    TEPR = True
    PCPD = True
    mean = True
class PupilGazeDecode():
    '''pupil channels in order
    confidence: 1 channel
    norm_pos_x/y: 2 channels
    gaze_point_3d_x/y/z: 3 channels
    eye_center0_3d_x/y/z (right/left, x/y/z): 6 channels (3 channels for each eye)
    gaze_normal0/1_x/y/z (right/left, x/y/z): 6 channels (3 channels for each eye)
    norm_pos_x/y: 2 channels
    diameter0/1_2d (right/left): 2 channels
    diameter0/1_3d (right/left): 2 channels
    22 total
    '''
    def __init__(self, featureChoices = CustomChoices()):
        super().__init__()
        self.numFeatures = sum([self.featureChoices.appr_entropy,
            self.featureChoices.TEPR,
            self.featureChoices.PCPD,
            self.featureChoices.mean]
        )

    def ProcessFeatures(self, epoch):
        # first we  extract the channel we want (for us we just want diameter0/1_3d for left and right)
        # then we filter for potential blinks
        # pupil labs channel order: 
        features = np.zeros(self.numFeatures)
        return features 

while not bci.connected: # check to see if lsl marker and datastream are available
    bci.Connect()
    time.sleep(1)
bci.TrainMode() # now both marker and datastreams available start training on received epochs
try:
    while(True):
        currentMarkers = bci.ReceivedMarkerCount() # check to see how many received epochs, if markers sent to close together will be ignored till done processing
        time.sleep(1) # wait for marker updates
        print("Markers received: " + str(currentMarkers), end="\r")
        if len(currentMarkers) > 1:  # check there is more then one marker type received
            if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired:
                classInfo = bci.CurrentClassifierInfo() # hangs if called too early
                #print(classInfo)
                print("\nClass accuracy: " + str(classInfo["accuracy"]) + " ")
            if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired+3:  
                bci.TestMode()
                break
    while True:
        markerGuess = bci.CurrentClassifierMarkerGuess() # when in test mode only y_pred returned
        guess = [key for key, value in currentMarkers.items() if value[0] == markerGuess]
        print("Current marker estimation: " + str(guess), end="\r")
        time.sleep(1)
except KeyboardInterrupt: # allow user to break while loop
    pass