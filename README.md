# PyBCI
A Python interface to create a BCI with the [Lab Streaming Layer](https://github.com/sccn/labstreaminglayer), [scikit-learn](https://scikit-learn.org/stable/#) and [TensorFlow](https://www.tensorflow.org/install) packages, leveraging packages like [Antropy](https://github.com/raphaelvallat/antropy), [SciPy](https://scipy.org/) and [NumPy](https://numpy.org/) for time and/or frequency based feature extraction.

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
while(True):
    time.sleep(1)
```
If you have no LSL available hardware, a psuedo time-series signal across 8 channels on an LSL stream with a StreamType "EMG" can be created with the script found in [mainSend.py PsuedoLSLStreamGenerator folder](https://github.com/LMBooth/pybci/blob/main/pybci/Examples/PsuedoLSLStreamGenerator/mainSend.py). 

## ToDo!
- Implement split epoch window settings to allow one marker to signify long period of data before and after marker which can be otionally split in to multiple epochs for training.
- Add LSL classification output marker stream + config.
- Capture data before and after maker relative to timestamp for data streams without sample rates or 0 sample rate.
- Add functions to get current accuracy, clf+model and assumed epoch num/string from classifier in pybci class (queues or callbacks)
### Future Work
- Add simple gaze decoding from pupil-labs
- Have custom class passable for data decoding, use general and gaze for inpiration. 
## Curently in Alpha, come back in a few days for updates!
