# PyBCI
A Python interface to create a BCI with the [Lab Streaming Layer](https://github.com/sccn/labstreaminglayer), [scikit-learn](https://scikit-learn.org/stable/#) and [TensorFlow](https://www.tensorflow.org/install) packages, leveraging packages like [Antropy](https://github.com/raphaelvallat/antropy), [SciPy](https://scipy.org/) and [NumPy](https://numpy.org/) for time and/or frequency based feature extraction.

## Background Information
PyBCI is a python based brain computer interface software designed to receive a varying number, be it singular or multiple, [Lab Streaming Layer](https://github.com/sccn/labstreaminglayer) enabled physiological sensors data streams. 

## Theory of Operation
1. Initialise BCI software
    - Operational Requirements
        - The bci must have an available LSL datastream with accepted dataType ("EEG", "EMG", "Gaze") {hopefully configurable in the future t pass custom fature decoding class}
        - The bci software must have a singular LSL marker stream selected (if more then one LSL marker stream on system set the desired ML training marker stream with PyBCI(markerStream="yourMarkerStream")). Warning: If None set picks first available in list.
    - Optional Configurations
        - We can configure epoch time window sizes based so many seconds before and after a marker has been received on PyBCI.LSLScanner.markerStream.
        -
    
2. ML Training:
  
3. TestMode

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

## ToDo!
- 
### Future Work
- Add simple gaze decoding from pupil-labs
- Create example to output classification on LSL and make test train configurable on pure LSL markers 
- custom class passable for data decoding, use general and gaze for inpiration. 

## Curently in Alpha, come back in a few days for updates!
