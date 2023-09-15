[![PyPI - Downloads](https://img.shields.io/pypi/dm/install-pybci)](https://pypi.org/project/install-pybci) [![PyPI - version](https://img.shields.io/pypi/v/install-pybci)](https://pypi.org/project/install-pybci)  [![Documentation Status](https://readthedocs.org/projects/pybci/badge/?version=latest)](https://pybci.readthedocs.io/en/latest/?badge=latest)

[![pybci](https://raw.githubusercontent.com/LMBooth/pybci/main/docs/Images/pyBCITitle.svg)](https://github.com/LMBooth/pybci)

A Python package to create real-time Brain Computer Interfaces (BCI's). Data synchronisation and pipelining handled by the [Lab Streaming Layer](https://github.com/sccn/labstreaminglayer), machine learning with [Pytorch](https://pytorch.org/), [scikit-learn](https://scikit-learn.org/stable/#) or [TensorFlow](https://www.tensorflow.org/install), leveraging packages like [AntroPy](https://github.com/raphaelvallat/antropy), [SciPy](https://scipy.org/) and [NumPy](https://numpy.org/) for generic time and/or frequency based feature extraction or optionally have the users own custom feature extraction class used.

The goal of PyBCI is to enable quick iteration when creating pipelines for testing human machine and brain computer interfaces, namely testing applied data processing and feature extraction techniques on custom machine learning models. Training the BCI requires LSL enabled devices and an LSL marker stream for timing stimuli. 

All the  [examples](https://github.com/LMBooth/pybci/tree/main/pybci/Examples) found on the github not in a dedicated folder have a pseudo LSL data generator enabled by default, `createPseudoDevice=True` so the examples can run without the need of LSL capable hardware. Any generic LSLViewer can be used to view the generated data, [example viewers found on this link.](https://labstreaminglayer.readthedocs.io/info/viewers.html)

# Installation
For stable releases use: ```pip install pybci-package```

For development versions use: ```pip install git+https://github.com/LMBooth/pybci.git``` or 
```
git clone https://github.com/LMBooth/pybci.git
cd pybci
pip install -e .
```
## Optional: Virtual Environment
Or optionally, install and run in a virtual environment:

Windows:
```
python -m venv my_env
.\my_env\Scripts\Activate
pip install pybci-package  # For stable releases
# OR
pip install git+https://github.com/LMBooth/pybci.git  # For development version
```
Linux/MaxOS:
```
python3 -m venv my_env
source my_env/bin/activate
pip install pybci-package  # For stable releases
# OR
pip install git+https://github.com/LMBooth/pybci.git  # For development version
```


## Prerequisite for Non-Windows Users
If you are not using windows then there is a prerequisite stipulated on the [pylsl repository](https://github.com/labstreaminglayer/pylsl) to obtain a liblsl shared library. See the [liblsl repo documentation](https://github.com/sccn/liblsl) for more information. 
Once the liblsl library has been downloaded ```pip install install-pybci``` should work.

(currently using install-pybci due to pybci having name too similar with another package on pypi, [issue here.](https://github.com/pypi/support/issues/2840))

[ReadTheDocs available here!](https://pybci.readthedocs.io/en/latest/)      [Examples found here!](https://github.com/LMBooth/pybci/tree/main/pybci/Examples)

[Examples of supported LSL hardware here!](https://labstreaminglayer.readthedocs.io/info/supported_devices.html)

## Python Package Dependencies Version Minimums
The following package versions define the minimum supported by PyBCI, also defined in setup.py:

    "pylsl>=1.16.1",
    "scipy>=1.11.1",
    "numpy>=1.24.3",
    "antropy>=0.1.6",
    "tensorflow>=2.13.0",
    "scikit-learn>=1.3.0",
    "torch>=2.0.1"
    
Earlier packages may work but are not guaranteed to be supported.

## Basic implementation
```python
import time
from pybci import PyBCI
if __name__ == '__main__':
    bci = PyBCI(createPseudoDevice=True) # set default epoch timing, looks for first available lsl marker stream and all data streams
    while not bci.connected: # check to see if lsl marker and datastream are available
        bci.Connect()
        time.sleep(1)
    bci.TrainMode() # now both marker and datastreams available start training on received epochs
    accuracy = 0
    try:
        while(True):
            currentMarkers = bci.ReceivedMarkerCount() # check to see how many received epochs, if markers sent to close together will be ignored till done processing
            time.sleep(0.5) # wait for marker updates
            print("Markers received: " + str(currentMarkers) +" Accuracy: " + str(round(accuracy,2)), end="         \r")
            if len(currentMarkers) > 1:  # check there is more then one marker type received
                if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired:
                    classInfo = bci.CurrentClassifierInfo() # hangs if called too early
                    accuracy = classInfo["accuracy"]
                if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired+10:  
                    bci.TestMode()
                    break
        while True:
            markerGuess = bci.CurrentClassifierMarkerGuess() # when in test mode only y_pred returned
            guess = [key for key, value in currentMarkers.items() if value[0] == markerGuess]
            print("Current marker estimation: " + str(guess), end="           \r")
            time.sleep(0.2)
    except KeyboardInterrupt: # allow user to break while loop
        print("\nLoop interrupted by user.")
```

## Background Information
PyBCI is a python brain computer interface software designed to receive a varying number, be it singular or multiple, Lab Streaming Layer enabled data streams. An understanding of time-series data analysis, the lab streaming layer protocol, and machine learning techniques are a must to integrate innovative ideas with this interface. An LSL marker stream is required to train the model, where a received marker epochs the data received on the accepted datastreams based on a configurable time window around certain markers - where custom marker strings can optionally have its epoch timewindow split and overlapped to count as more then one marker, example: in training mode a baseline marker may have one marker sent for a 60 second window, whereas target actions may only be ~0.5s long,  when testing the model and data is constantly analysed it would be desirable to standardise the window length, we do this by splitting the 60s window after the received baseline marker in to ~0.5s windows. PyBCI allows optional overlapping of time windows to try to account for potential missed signal patterns/aliasing - as a rule of thumb it would be advised when testing a model to have a time window overlap >= 50% (Shannon-Nyquist criterion). [See here for more information on epoch timing](https://pybci.readthedocs.io/en/latest/BackgroundInformation/Epoch_Timing.html).

Once the data has been epoched it is sent for feature extraction, there is a general feature extraction class which can be configured for general time and/or frequency analysis based features, ideal for data stream types like "EEG" and "EMG". Since data analysis, preprocessing and feature extraction trechniques can vary greatly between device data inputs, a custom feature extraction class can be created for each data stream maker type. [See here for more information on feature extraction](https://pybci.readthedocs.io/en/latest/BackgroundInformation/Feature_Selection.html).

Finally a passable pytorch, sklearn or tensorflow classifier can be given to the bci class, once a defined number of epochs have been obtained for each received epoch/marker type the classifier can begin to fit the model. It's advised to use bci.ReceivedMarkerCount() to get the number of received training epochs received, once the min num epochs received of each type is >= pybci.minimumEpochsRequired (default 10 of each epoch) the model will begin to fit. Once fit the classifier info can be queried with CurrentClassifierInfo, this returns the model used and accuracy. If enough epochs are received or high enough accuracy is obtained TestMode() can be called. Once in test mode you can query what pybci estimates the current bci epoch is(typically baseline is used for no state). [Review the examples for sklearn and model implementations](https://pybci.readthedocs.io/en/latest/BackgroundInformation/Examples.html).

## All issues, recommendations, pull-requests and suggestions are welcome and encouraged!
