
## Description
The PsuedoLSLSreamGenerator found in mainSend.py in this directory can generate multiple channels on a given stream type at a given sample rate.

A baseline signal is generated on an LSL stream outlet and a PyQt button can be pressed to signify this baseline signal on a separate LSL marker stream. 
There are a number of markers that can be sent on the LSL marker stream which alters the data stream outlet signal for a given amount of time.

To run this script requires installation of PyQt5 for the button interface and PyQtGraph for the data plotting.

In it's current configuration you can send a trigger marker for 5 different signal types (all modified slightly in the __init__) and a baseline marker with stream data plotted in a pyqt/pyqtgraph application.

### Note:
To change the signal for each marker type you can alter the PsuedoEMGDataConfig properties in init for each marker respectively.

Default config:
```python
class PsuedoEMGDataConfig:
    duration = 1.0 
    noise_level = 0.1
    amplitude = 0.2
    frequency = 1.0
```

## ToDo

