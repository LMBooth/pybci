import time
from pybci import PyBCI
from pybci.Configuration.EpochSettings import GlobalEpochSettings
from pybci.Utils.Logger import Logger
import numpy as np
from scipy.fft import fft
# We control the arduino via lsl python script which is responsible for COM connection
from pylsl import StreamInfo, StreamOutlet
# Set up the LSL stream info
info = StreamInfo('MarkerHandGrasps', 'Markers', 1, 0, 'string', 'myuniquemarkerid2023')
# Create the outlet
outlet = StreamOutlet(info)
gs = GlobalEpochSettings()
gs.tmax = 2.5 # grab 1 second after marker
gs.tmin = -0.5 # grab 0.5 seconds before marker
gs.splitCheck = True # splits samples between tmin and tmax
gs.windowLength = 0.5  # 
gs.windowOverlap = 0.5 # windows overap by 50%, so for a total len
#from sklearn.neural_network import MLPClassifier
#clf = MLPClassifier(max_iter = 1000, solver ="lbfgs")#solver=clf, alpha=alpha,hidden_layer_sizes=hid)
class EMGClassifier():
    def ProcessFeatures(self, epochData, sr, epochNum): # Every custom class requires a function with this name and structure to extract the featur data and epochData is always [Samples, Channels]
        print(epochData.shape)

        rmsCh1 = np.sqrt(np.mean(np.array(epochData[:,0])**2))
        rangeCh1 = max(epochData[:,0])-min(epochData[:,0])
        varCh1 = np.var(epochData[:,0])
        meanAbsCh1 = np.mean(np.abs(epochData[:,0]))
        zeroCrossCh1 = ((epochData[:,0][:-1] * epochData[:,0][1:]) < 0).sum()
        fft_result = fft(epochData[:,0])
        frequencies = np.fft.fftfreq(len(epochData[:,0]), 1/ 192) # approximate sample rate

        delta_mask = (frequencies >= 0.5) & (frequencies <= 2)
        delta_power = np.mean(np.abs(fft_result[delta_mask])**2)
        delta2_mask = (frequencies >= 2) & (frequencies <= 4)
        delta2_power = np.mean(np.abs(fft_result[delta2_mask])**2)

        theta_mask = (frequencies >= 4) & (frequencies <= 7)
        theta_power = np.mean(np.abs(fft_result[theta_mask])**2)
        
        alpha_mask = (frequencies >= 7) & (frequencies <= 10)
        alpha_power = np.mean(np.abs(fft_result[alpha_mask])**2)

        beta_mask = (frequencies >= 10) & (frequencies <= 15)
        beta_power = np.mean(np.abs(fft_result[beta_mask])**2)
        beta2_mask = (frequencies >= 15) & (frequencies <= 20)
        beta2_power = np.mean(np.abs(fft_result[beta2_mask])**2)

        gamma_mask = (frequencies >= 20) & (frequencies <= 25)
        gamma_power = np.mean(np.abs(fft_result[gamma_mask])**2)

        a = np.array([rmsCh1, varCh1,rangeCh1,  meanAbsCh1,  zeroCrossCh1, max(epochData[:,0]), min(epochData[:,0]), 
                         alpha_power, delta_power,delta2_power, theta_power, beta_power,beta2_power, gamma_power]).T
        
        return np.nan_to_num(a)
            #[rmsCh1, rmsCh2,varCh1,varCh2,rangeCh1, rangeCh2, meanAbsCh1, meanAbsCh2, zeroCrossCh1,zeroCrossCh2])
    
streamCustomFeatureExtract = {"ArduinoHandData":EMGClassifier()}
dataStreams = ["ArduinoHandData"]
bci = PyBCI(minimumEpochsRequired = 6, loggingLevel= Logger.INFO, globalEpochSettings=gs,dataStreams=dataStreams, markerStream="TestMarkers", streamCustomFeatureExtract=streamCustomFeatureExtract)#, loggingLevel = Logger.NONE)
while not bci.connected: # check to see if lsl marker and datastream are available
    bci.Connect()
    time.sleep(1)

#ser.write(b'0\r')  # Convert the string to bytes and send 

bci.TrainMode() # now both marker and datastreams available start training on received epochs
accuracy = 0
try:
    while(True):
        currentMarkers = bci.ReceivedMarkerCount() # check to see how many received epochs, if markers sent to close together will be ignored till done processing
        time.sleep(0.2) # wait for marker updates
        print("Markers received: " + str(currentMarkers) +" Class accuracy: " + str(accuracy), end="\r")
        if len(currentMarkers) > 1:  # check there is more then one marker type received
            #print(bci.CurrentFeaturesTargets())
            if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired:
                classInfo = bci.CurrentClassifierInfo() # hangs if called too early
                accuracy = classInfo["accuracy"]
            if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired+30:  
                #time.sleep(2)
                bci.TestMode()
                break
    while True:
        markerGuess = bci.CurrentClassifierMarkerGuess() # when in test mode only y_pred returned
        guess = [key for key, value in currentMarkers.items() if value[0] == markerGuess]
        print("Current marker estimation: " + str(guess), end="\r")
        time.sleep(0.2)
        if len(guess)>0:
            outlet.push_sample([str(markerGuess)])
            if guess[0] == "open":
            #print("sending 0")
                outlet.push_sample(["0\r"])
                #ser.write(b"0\r")  # Convert the string to bytes and send 
            elif guess[0] == "fist":
                outlet.push_sample(["1\r"])
                #ser.write(b"1\r")  # Convert the string to bytes and send 
            elif guess[0] == "rock":
                outlet.push_sample(["2\r"])
                #ser.write(b"2\r")  # Convert the string to bytes and send 
            elif guess[0] == "peace":
                outlet.push_sample(["3\r"])
                #ser.write(b"3\r")  # Convert the string to bytes and send 
            elif guess[0] == "pinky":
                outlet.push_sample(["4\r"])
                #ser.write(b"4\r")  # Convert the string to bytes and send 
            elif guess[0] == "thumb":
                outlet.push_sample(["5\r"])
                #ser.write(b"5\r")  # Convert the string to bytes and send 

#["open", "fist", "rock"]#, "peace", "pinky"]
except KeyboardInterrupt: # allow user to break while loop
    pass