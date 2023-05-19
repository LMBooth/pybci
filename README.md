[![PyPI - Downloads](https://img.shields.io/pypi/dm/install-pybci)](https://pypi.org/project/install-pybci)  [![Documentation Status](https://readthedocs.org/projects/pybci/badge/?version=latest)](https://pybci.readthedocs.io/en/latest/?badge=latest)

![Alt Text](https://github.com/LMBooth/pybci/blob/main/docs/Images/pyBCITitle.png)

A Python interface to create a Brain Computer Interface (BCI) with the [Lab Streaming Layer](https://github.com/sccn/labstreaminglayer), [scikit-learn](https://scikit-learn.org/stable/#) and [TensorFlow](https://www.tensorflow.org/install) packages, leveraging packages like [AntroPy](https://github.com/raphaelvallat/antropy), [SciPy](https://scipy.org/) and [NumPy](https://numpy.org/) for time and/or frequency based feature extraction.

The goal of PyBCI is to help create quick iteration pipelines for testing potential human machine and brain computer interfaces, namely applied machine learning models and feature extraction techniques.

## Installation
For stable releases use: ```pip install install-pybci```
(currently can only install-pybci due to pybci name similarities with another package on pypi)

For unstable dev installations and up-to-date git pushes use: ```pip install --index-url https://test.pypi.org/simple/ install-pybci```


[ReadTheDocs available here!](https://pybci.readthedocs.io/en/latest/) (In development)

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

## Background Information
PyBCI is a python based brain computer interface software designed to receive a varying number, be it singular or multiple, [Lab Streaming Layer](https://github.com/sccn/labstreaminglayer) enabled physiological sensor data streams. An understanding of time-series data analysis, the lab streaming layer protocol, and machine learning techniques are a must to integrate innovative ideas with this interface.
An LSL marker stream is required to train the model, where a received marker [epochs](https://www.google.com/search?q=epochs+definition&rlz=1C1CHBF_en-GBGB921GB921&sxsrf=APwXEddAlMkYQ6MqziIvXbvsCxl3SjySNA%3A1684343462996&ei=pgplZJK2PJiigAaErrWwCw&oq=epochs+def&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAxgAMg0IABCKBRCRAhBGEPkBMgUIABCABDIFCAAQgAQyBQgAEIAEMggIABAWEB4QDzIICAAQFhAeEA8yBggAEBYQHjIGCAAQFhAeMgYIABAWEB4yCQgAEBYQHhDxBDoHCCMQsAMQJzoKCAAQRxDWBBCwAzoHCCMQigUQJzoHCAAQigUQQzoNCAAQigUQsQMQgwEQQzoICAAQigUQkQJKBAhBGABQmAZYwQhgzw5oAXABeACAAYYBiAHBA5IBAzEuM5gBAKABAcgBCsABAQ&sclient=gws-wiz-serp) the data received on the accepted datastreams based on a configurable time window around certain markers - where custom marker strings can optionally have its epoch timewindow split and overlapped to count for more then one marker, example: a baseline marker may have one marker sent for a 60 second window, where as target actions may only be ~0.5s long, so to conform when testing the model and giving a standardised window length would be desirable to split the 60s window after the received baseline marker in to ~0.5s windows. By overlapping windows we try to account for potential missed signal patterns/aliasing, as a rule of thumb it would be advised when testing a model to have an overlap >= than 50%, [see Shannon nyquist criterion](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem).

Once the data has been epoched it is sent for feature extraction, there is a general feature extraction class which can be configured for general time and/or frequency analysis based features, data streams types like "EEG" and "EMG". (DevNOTE: looking to write class for basic pupil labs example to be passed, or solely samples + channels, or other custom classes passed to selected marker streams >:] )

Finally a passable, customisable sklearn or tensorflow classifier can be giving to the bci class, once a defined number of epochs have been obtained for each received epoch/marker type the classifier can begin to fit the model. It's advised to use bci.ReceivedMarkerCount() to get the number of received training epochs received, once the min num epochs received of each type is >= pybci.minimumEpochsRequired (default 10 of each epoch) the mdoel will begin to fit. Once fit classifier info can be queried with CurrentClassifierInfo, when a desired accuracy is met or number of epochs TestMode() can be called. Once in test mode you can query (sould change function to own function and queue for quering testthread) what pybci estimates the current bci epoch is(typically bseline is used for no state).

If you have no LSL available hardware, a psuedo time-series signal can be created with the script found in [mainSend.py PsuedoLSLStreamGenerator folder](https://github.com/LMBooth/pybci/tree/main/pybci/Examples/PsuedoLSLStreamGenerator/mainSend.py). 


## Theory of Operation
1. Requirements Prior Initialising with ```python bci = PyBCI() ```
    - The bci must have >=1 LSL datastream with an accepted dataType ("EEG", "EMG", "Gaze") {hopefully configurable in the future t pass custom fature decoding class}
    - The bci must have ==1 LSL markerstream selected (if more then one LSL marker stream on system set the desired ML training marker stream with PyBCI(markerStream="yourMarkerStream")). Warning: If None set picks first available in list.
2. Initialising BCI software
    - [Optional Configurations](https://pybci.readthedocs.io/en/latest/api/PyBCI.html) for ```python PyBCI() ```
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
