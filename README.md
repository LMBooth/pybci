# PyBCI
A Python interface to create a BCI with the [Lab Streaming Layer](https://github.com/sccn/labstreaminglayer), [scikit-learn](https://scikit-learn.org/stable/#) and [TensorFlow](https://www.tensorflow.org/install) packages, leveraging packages like [Antropy](https://github.com/raphaelvallat/antropy), [SciPy](https://scipy.org/) and [NumPy](https://numpy.org/) for time and/or frequency based feature extraction.

## Installation
pip install --index-url https://test.pypi.org/simple/ pybci
(currently can only install with testPyPi due to name isimilarities with another package on normal pip)

## Basic implementation:

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

For an example with no required LSL available hardware, a psuedo time-series signal across 8 channels on an LSL stream with a StreamType "EMG" can be created with the script found in [PsuedoLSLStreamGenerator folder](https://github.com/LMBooth/pybci/tree/main/PsuedoLSLStreamGenerator). You can send a trigger marker for 5 different signal types (all modified slightly in the __init__) and a baseline marker with stream data plotted in a pyqt/pyqtgraph application.


[ReadTheDocs available here!](https://pybci.readthedocs.io/en/latest/) (In development)


ToDo!
Implement split epoch window settings to allow one marker to signify long period of data which can be split in to multiple epochs for training.

## Curently in Alpha, come back in a few days for updates!
