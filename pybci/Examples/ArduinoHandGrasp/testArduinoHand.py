import time
from pybci import PyBCI
import serial
from pybci.Configuration.EpochSettings import GlobalEpochSettings
from pybci.Utils.FeatureExtractor import GenericFeatureExtractor, GeneralFeatureChoices
from pybci.Utils.Logger import Logger

port = 'COM9'
baud_rate = 9600
ser = serial.Serial(port, baud_rate)

dropchs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,23] 
#sending = False
featureChoices = GeneralFeatureChoices()
featureChoices.psdBand =True
featureChoices.appr_entropy = False
featureChoices.perm_entropy = False
featureChoices.spec_entropy = True
featureChoices.svd_entropy = False
featureChoices.samp_entropy = False
featureChoices.zeroCross = True
featureChoices.waveformLength = True
featureChoices.slopeSignChange = True
streamCustomFeatureExtract = {"EEGStream": GenericFeatureExtractor(freqbands = [[1.0, 4.0], [4.0, 8.0], [8.0, 12.0], [12.0, 20.0],[20.0, 30.0]], featureChoices=featureChoices)}

gs = GlobalEpochSettings()
gs.tmax = 2.5 # grab 1 second after marker
gs.tmin = -0.5 # grab 0.5 seconds before marker
gs.splitCheck = True # splits samples between tmin and tmax
gs.windowLength = 0.7  # 
gs.windowOverlap = 0.5 # windows overap by 50%, so for a total len


from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(max_iter = 1000, solver ="lbfgs")#solver=clf, alpha=alpha,hidden_layer_sizes=hid)

bci = PyBCI(minimumEpochsRequired = 6, loggingLevel= Logger.INFO,clf=clf, globalEpochSettings=gs, markerStream="TestMarkers", streamCustomFeatureExtract=streamCustomFeatureExtract,streamChsDropDict={"EEGStream":dropchs})#, loggingLevel = Logger.NONE)
while not bci.connected: # check to see if lsl marker and datastream are available
    bci.Connect()
    time.sleep(1)

ser.write(b'0\r')  # Convert the string to bytes and send 

bci.TrainMode() # now both marker and datastreams available start training on received epochs
accuracy = 0
try:
    while(True):
        currentMarkers = bci.ReceivedMarkerCount() # check to see how many received epochs, if markers sent to close together will be ignored till done processing
        time.sleep(0.1) # wait for marker updates
        print("Markers received: " + str(currentMarkers) +" Class accuracy: " + str(accuracy), end="\r")
        if len(currentMarkers) > 1:  # check there is more then one marker type received
            #print(bci.CurrentFeaturesTargets())
            if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired:
                classInfo = bci.CurrentClassifierInfo() # hangs if called too early
                accuracy = classInfo["accuracy"]
            if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired+25:  
                time.sleep(2)
                bci.TestMode()
                break
    while True:
        markerGuess = bci.CurrentClassifierMarkerGuess() # when in test mode only y_pred returned
        guess = [key for key, value in currentMarkers.items() if value[0] == markerGuess]
        print("Current marker estimation: " + str(guess), end="\r")
        time.sleep(0.1)
        if len(guess)>0:
            if guess[0] == "open":
            #print("sending 0")
                ser.write(b"0\r")  # Convert the string to bytes and send 
            elif guess[0] == "fist":
            #print("sending 1")
                ser.write(b"1\r")  # Convert the string to bytes and send 
            elif guess[0] == "rock":
                ser.write(b"2\r")  # Convert the string to bytes and send 
            elif guess[0] == "peace":
                ser.write(b"3\r")  # Convert the string to bytes and send 
            elif guess[0] == "pinky":
                ser.write(b"4\r")  # Convert the string to bytes and send 
            elif guess[0] == "thumb":
                ser.write(b"5\r")  # Convert the string to bytes and send 

#["open", "fist", "rock"]#, "peace", "pinky"]
except KeyboardInterrupt: # allow user to break while loop
    pass