import time, sys
sys.path.append('../')  # add the parent directory of 'utils' to sys.path, whilst in beta build.
from pybci import PyBCI

validMarkerStream = ["Markers"]
validDataStreams = ["UoHExGStream"]#Liam's EMG device"]

bci = PyBCI()   #dataStreams = validDataStreams, markerStream = validMarkerStream)
if bci.lslScanner.CheckAvailableLSL():
    bci.StartTraining()

#with PyBCI() as bci:
    #time.sleep(1)
    #if bci.lslScanner.CheckAvilableLSL():
    #    bci.StartTraining()