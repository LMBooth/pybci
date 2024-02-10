from pybci import PyBCI
from pybci.Configuration.FeatureSettings import GeneralFeatureChoices
from pybci.Configuration.EpochSettings import IndividualEpochSetting, GlobalEpochSettings
from pybci.Utils.FeatureExtractor import GenericFeatureExtractor
import time

# Test case using the fixture
#@pytest.mark.timeout(300)  # Extended timeout to 5 minutes
def test_run_bci():
    features = GeneralFeatureChoices
    features.psdBand = True
    #features.appr_entropy = True
    features.perm_entropy = True    
    features.spec_entropy = True
    features.svd_entropy = True
    features.rms = False
    features.meanPSD = False
    features.medianPSD = True
    features.variance = True
    features.meanAbs = True
    features.waveformLength = True
    features.zeroCross = True
    features.slopeSignChange = True

    markerSettings = {}
    markerSettings["baseline"] = IndividualEpochSetting()
    markerSettings["baseline"].splitCheck = False
    markerSettings["baseline"].tmin = 0      # time in seconds to capture samples before trigger
    markerSettings["baseline"].tmax=  2      # time in seconds to capture samples after trigger

    markerSettings["Marker1"] = IndividualEpochSetting()
    markerSettings["Marker1"].splitCheck = True
    markerSettings["Marker1"].tmin = 0      # time in seconds to capture samples before trigger
    markerSettings["Marker1"].tmax=  2      # time in seconds to capture samples after trigger

    extractor = GenericFeatureExtractor(featureChoices=features)



    bci = PyBCI(minimumEpochsRequired = 2, createPseudoDevice= True, customEpochSettings=markerSettings,  streamCustomFeatureExtract={"PyBCIPseudoDataStream":extractor},
                markerStream= "PyBCIPseudoMarkers", dataStreams=["PyBCIPseudoDataStream"]) 
    # set new config settings after instantiation
    bci.ConfigureEpochWindowSettings(globalEpochSettings = GlobalEpochSettings(), customEpochSettings = markerSettings)
    while not bci.connected:
        bci.Connect()
        time.sleep(1)
    bci.TrainMode()
    marker_received = False
    in_test_mode = False
    accuracy_achieved = False
    accuracy= 0
    while True:
        currentMarkers = bci.ReceivedMarkerCount() # check to see how many received epochs, if markers sent to close together will be ignored till done processing
        time.sleep(0.5) # wait for marker updates
        print("Markers received: " + str(currentMarkers) +" Accuracy: " + str(round(accuracy,2)), end="         \r")
        if len(currentMarkers) > 1:  # check there is more then one marker type received
            marker_received = True
            if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired:
                classInfo = bci.CurrentClassifierInfo() # hangs if called too early
                accuracy = classInfo["accuracy"]###
                if accuracy > 0:
                    # set to above 0 to show some accuracy was retruend from model
                    accuracy_achieved = True
                    bci.TestMode()
                    break
            #if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired+4:
            #    break
    while True:
        
        markerGuess = bci.CurrentClassifierMarkerGuess() # when in test mode only y_pred returned
        print(markerGuess)
        bci.CurrentFeaturesTargets()
        in_test_mode = True
        time.sleep(1) 
        bci.StopThreads()
        break
        #print("Current marker estimation: " + str(guess), end="           \r")
    assert accuracy_achieved and marker_received and in_test_mode
