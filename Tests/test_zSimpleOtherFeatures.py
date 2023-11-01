from pybci import PyBCI, get_os
from pybci.Configuration.FeatureSettings import GeneralFeatureChoices
from pybci.Utils.FeatureExtractor import GenericFeatureExtractor
from pybci.Utils.PseudoDevice import PseudoDeviceController
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

    extractor = GenericFeatureExtractor(featureChoices=features)
    current_os = get_os()
    if current_os == "Windows":
        bci = PyBCI(minimumEpochsRequired = 3, createPseudoDevice= True, streamCustomFeatureExtract={"PyBCIPseudoDataStream":extractor}) 
    else:
        pdc = PseudoDeviceController(execution_mode="process")
        pdc.BeginStreaming()
        time.sleep(10)
        bci = PyBCI(minimumEpochsRequired = 3, createPseudoDevice=True, pseudoDeviceController=pdc, streamCustomFeatureExtract={"PyBCIPseudoDataStream":extractor})
    #while not bci.connected:
    #bci = PyBCI(minimumEpochsRequired=1, createPseudoDevice=True, streamCustomFeatureExtract={"PyBCIPseudoDataStream":extractor})
    while not bci.connected:
        bci.Connect()
        time.sleep(1)
    bci.TrainMode()
    marker_received = False
    while True:
        currentMarkers = bci.ReceivedMarkerCount() # check to see how many received epochs, if markers sent to close together will be ignored till done processing
        time.sleep(0.5) # wait for marker updates
        #print("Markers received: " + str(currentMarkers) +" Accuracy: " + str(round(accuracy,2)), end="         \r")
        if len(currentMarkers) > 2:  # check there is more then one marker type received
            marker_received = True
            bci.StopThreads()
            break
            #if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired+4:
            #    break
    assert marker_received
