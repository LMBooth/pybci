# PyBCI
A Python interface to create a BCI with the [Lab Streaming Layer](https://github.com/sccn/labstreaminglayer), [scikit-learn](https://scikit-learn.org/stable/#) and [TensorFlow](https://www.tensorflow.org/install) packages, leveraging packages like [Antropy](https://github.com/raphaelvallat/antropy), [SciPy](https://scipy.org/) and [NumPy](https://numpy.org/) for time and/or frequency based feature extraction.

Basic implementation:
```python
from PyBCI.pybci import PyBCI
from PyBCI.Configuration.EpochSettings import GlobalEpochSettings, IndividualEpochSetting

bci = PyBCI() # create pybci object which auto scans for first available LSL marker and all accepted data streams

while not bci.connected: # check maker and data LSL streams avaialble
    bci.Connect() # if not trr to reconnect
    time.sleep(1)

bci.TrainMode() # Now connected start bci training (defaults to sklearn SVM and all general feature settings, found in PyBCI.Configuration.FeatureSettings.GeneralFeatureChoices)
while(True):
    currentMarkers = bci.ReceivedMarkerCount() # gets current received training markers on marker stream
    time.sleep(1) # poll for 1 second
    
    if len(currentMarkers) > 1:   
        if min([currentMarkers[key][1] for key in currentMarkers]) > 10:
        bci.TestMode()
        break

while(True):
    # polls in training, needs function adding to pull guessed received markers
    time.sleep(1)

```


[ReadTheDocs!](https://pybci.readthedocs.io/en/latest/) (In development)


ToDo!
Implement split epoch window settings to allow one marker to signify long period of data which can be split in to multiple epochs for training.

## Curently in Alpha, come back in a few days for updates!
