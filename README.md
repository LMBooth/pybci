# PyBCI
A Python interface to create a BCI with the [Lab Streaming Layer](https://github.com/sccn/labstreaminglayer), [scikit-learn](https://scikit-learn.org/stable/#) and [TensorFlow](https://www.tensorflow.org/install) packages, leveraging packages like [Antropy](https://github.com/raphaelvallat/antropy), [SciPy](https://scipy.org/) and [NumPy](https://numpy.org/) for time and/or frequency based feature extraction.

## Background Information
PyBCI is a python based brain computer interface software designed to receive a varying number, be it singular or multiple, [Lab Streaming Layer](https://github.com/sccn/labstreaminglayer) enabled physiological sensors data streams. 

[ReadTheDocs available here!](https://pybci.readthedocs.io/en/latest/) (In development)

## Installation
```
pip install --index-url https://test.pypi.org/simple/ pybci
```
(currently can only install with test.pypi due to name similarities with another package on pypi)

## Basic implementation
```python
import time
from pybci import PyBCI
bci = PyBCI()
while not bci.connected:
    bci.Connect()
    time.sleep(1)
bci.TrainMode()
while(True):
    currentMarkers = bci.ReceivedMarkerCount()
    time.sleep(1) # wait for marker updates
    if len(currentMarkers) > 1:  # check there is more then one marker type received
        if min([currentMarkers[key][1] for key in currentMarkers]) > 10:
            bci.TestMode()
            break 
try:
    while True:
        classInfo = bci.CurrentClassifierInfo() # when in train mode only y_pred returned
        guess = [key for key, value in currentMarkers.items() if value[0] == classInfo["y_pred"]]
        print("Current marker estimation: " + str(guess), end="\r")
except KeyboardInterrupt: # allow user to break while loop
    pass
```
If you have no LSL available hardware, a psuedo time-series signal can be created with the script found in [mainSend.py PsuedoLSLStreamGenerator folder](https://github.com/LMBooth/pybci/blob/main/pybci/Examples/PsuedoLSLStreamGenerator/mainSend.py). 


## Theory of Operation
1. Requirements Prior Initialising with ```python bci = PyBCI() ```
    - The bci must have >=1 LSL datastream with an accepted dataType ("EEG", "EMG", "Gaze") {hopefully configurable in the future t pass custom fature decoding class}
    - The bci must have ==1 LSL markerstream selected (if more then one LSL marker stream on system set the desired ML training marker stream with PyBCI(markerStream="yourMarkerStream")). Warning: If None set picks first available in list.
2. Initialising BCI software
    - Optional Configurations for ```python PyBCI() ```
        - dataStreams: list(string) - list of target data streams.
        - markerStream: string - target marker training stream.
        - streamTypes: list(string) - list of target data stream types if no specified streams set with dataStreams.
        - markerTypes: list(string) - list of target data marker types if no specified marker set with markerStream.
        - printDebug: bool - (default True) Sets whether PyBCI debug messages are printed.
        - globalEpochSettings: GlobalEpochSettings() - (default ) can be found in pybci.Configurations folder, sets global epoch timing settings
            - splitCheck: bool (default False) - Checks whether or not subdivide epochs.
            - tmin: int (default 0) - Time in seconds to capture samples before marker.
            - tmax: int (default 1) - Time in seconds to capture samples after marker.
            - windowLength: float (default 0.5) - If splitcheck true - time in seconds to split epoch. 
            - windowOverlap: float(default 0.5) - If splitcheck true  percentage value > 0 and < 1, example if epoch has tmin of 0 and tmax of 1 with window.
        - customEpochSettings: dict(str:IndividualEpochSetting()) - Each key in the dict specifies the target marker received on the marker stream and sets if the target epoch should have its time window cut up. 
            IndividualEpochSetting
            - splitCheck: bool (default False) Checks whether or not subdivide epochs. (Note: If True, divides epoch based on window global overlap and length as all have to be uniform to match with testmode window size)
            - tmin: int (default 0) Time in seconds to capture samples before marker.
            - tmax: int (default 1) Time in seconds to capture samples after marker.
        - streamChsDropDict: dict(str:list(int)) - Each key specifies the datastream and the list of indicies specifies which channels to drop in that keys stream.
        - freqbands: list(list()) - (default [[1.0, 4.0], [4.0, 8.0], [8.0, 12.0], [12.0, 20.0]]) 2D list of frequency bands for feature extraction where 1st dimension is m extensible and 2nd must have a length of 2 [lowerFc, higherFc].
        - featureChoices: GeneralFeatureChoices() - (default GeneralFeatureChoices()) - Sets trget features for decoding from time series data (pybci.utils.FeatureExtractor) 
        - minimumEpochsRequired: Int (default 10) minimum number of required epochs before model compiling begins (Warning: too low an suffer from inadequate test train epoch splitting for accuracy validation)
        - clf: sklearn.base.ClassifierMixin() - (default SVM) allows user sklearn clf to be passed, if no model or clf is passed then defaults to sklearn SVM with rbf kernel.
        - model: tf.keras.Model() - allows user tensorflow model to be passed, if no model or clf is passed then defaults to sklearn SVM with rbf kernel.
    - Once configuration settings are set various threads are created.
        - The marker stream has its own thread which recieves markers from the target LSL marker stream and when in train mode pushes this marker to the available datastreams. 
        - Each data stream has its own thread created responsible for pipleining received data on FIFO's and slicing approprialey based on globalEpochSettings and customEpochSettings.
        - FeaturesExtractor
        - Classifier
    
3. ML Training:
  
4. ML Testing:

## ToDo!
- 
### Future Work
- Add simple gaze decoding from pupil-labs
- Create example to output classification on LSL and make test train configurable on pure LSL markers 
- custom class passable for data decoding, use general and gaze for inpiration. 

## Curently in Alpha, come back in a few days for updates!
