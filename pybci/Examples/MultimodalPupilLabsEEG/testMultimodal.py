import time
from pybci import PyBCI
import numpy as np
from scipy.fft import fft

class PupilGazeDecode():
    # pupil-labs channels:
    #['confidence',  1
    # 'norm_pos_x', 'norm_pos_y', 'gaze_point_3d_x', 'gaze_point_3d_y', 'gaze_point_3d_z',  + 5
    # 'eye_center0_3d_x', 'eye_center0_3d_y', 'eye_center0_3d_z', 'eye_center1_3d_x', 'eye_center1_3d_y',  + 5
    # 'eye_center1_3d_z', # 'gaze_normal0_x', 'gaze_normal0_y', 'gaze_normal0_z', 'gaze_normal1_x',  + 5
    # 'gaze_normal1_y', 'gaze_normal1_z', 'diameter0_2d', 'diameter1_2d', 'diameter0_3d',  'diameter1_3d'] + 6 = 22 channels
    def __init__(self):
        super().__init__()
    def ProcessFeatures(self, epochData, sr, epochNum): # This is the required function name and variables that are passed to all 
        epochData = np.nan_to_num(epochData) # sklearn doesnt like nan
        #print(epochData.shape)
        if len(epochData[0]) ==  0:
            return [0,0,0]
        else:
            confidence = np.mean(epochData[0]) # channel 21 is 3d pupil diameter right, get mean
            rightmean = np.mean(epochData[1]) # channel 20 is 3d pupil diameter right, get mean
            leftmean = np.mean(epochData[2]) # channel 21 is 3d pupil diameter right, get mean
            bothmean = np.mean([(epochData[1][i] + epochData[2][i]) / 2 for i in range(len(epochData[1]))]) # mean of both eyes in 3d
            return np.nan_to_num([confidence, rightmean,leftmean,bothmean]) #  expects 2d
        

class EOGClassifier():
    # used Fp1 and Fp2 from io:bio EEG device
    def ProcessFeatures(self, epochData, sr, epochNum): # Every custom class requires a function with this name and structure to extract the featur data and epochData is always [Samples, Channels]
        #print(epochData.shape)
        rmsCh1 = np.sqrt(np.mean(np.array(epochData[:,0])**2))
        rangeCh1 = max(epochData[:,0])-min(epochData[:,0])
        varCh1 = np.var(epochData[:,0])
        meanAbsCh1 = np.mean(np.abs(epochData[:,0]))
        zeroCrossCh1 = ((epochData[:,0][:-1] * epochData[:,0][1:]) < 0).sum()
        fft_result = fft(epochData[:,0])
        frequencies = np.fft.fftfreq(len(epochData[:,0]), 1/ sr)
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

hullUniEEGLSLStreamName = "EEGStream"#EEGStream"
pupilLabsLSLName = "pupil_capture" 
markerstream = "TestMarkers" # using pupillabs rightleftmarkers example
streamCustomFeatureExtract = {pupilLabsLSLName: PupilGazeDecode(), hullUniEEGLSLStreamName: EOGClassifier()} #GenericFeatureExtractor
dataStreamNames = [pupilLabsLSLName, hullUniEEGLSLStreamName]
# to reduce overall computational complexity we are going to drop irrelevant channels
streamChsDropDict = {hullUniEEGLSLStreamName : [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,19,20,21,22,23],#0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,19,20,21,22,23], # for our device we have Fp1 and Fp2 on channels 18 and 19, so list values 17 and 18 removed
                     pupilLabsLSLName: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17, 20, 21] # pupil labs we only wan left and right 3d pupil diameter, drop rest
                     } 
bci = PyBCI(dataStreams = dataStreamNames, markerStream=markerstream, minimumEpochsRequired = 4,
            streamChsDropDict = streamChsDropDict,
            streamCustomFeatureExtract=streamCustomFeatureExtract ) #model = model, 

while not bci.connected:
    bci.Connect()
    time.sleep(1)
print(bci.markerStream.info().name())
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
            if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired+6:  
                bci.TestMode()
                break
    while True:
        markerGuess = bci.CurrentClassifierMarkerGuess() # when in test mode only y_pred returned
        guess = [key for key, value in currentMarkers.items() if value[0] == markerGuess]
        print("Current marker estimation: " + str(guess), end="\r")
        time.sleep(0.5)
except KeyboardInterrupt: # allow user to break while loop
    pass
