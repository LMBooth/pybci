# PyBCI
A Python interface to create a BCI with the [Lab Streaming Layer](https://github.com/sccn/labstreaminglayer), [scikit-learn](https://scikit-learn.org/stable/#) and [TensorFlow](https://www.tensorflow.org/install) packages, leveraging packages like [Antropy](https://github.com/raphaelvallat/antropy), [SciPy](https://scipy.org/) and [NumPy](https://numpy.org/) for time and/or frequency based feature extraction.

Basic implementation:
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


[ReadTheDocs!](https://pybci.readthedocs.io/en/latest/) (In development)


ToDo!
Implement split epoch window settings to allow one marker to signify long period of data which can be split in to multiple epochs for training.

## Curently in Alpha, come back in a few days for updates!
