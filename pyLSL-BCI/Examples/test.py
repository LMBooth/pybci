import time, sys
sys.path.append('../')  # add the parent directory of 'utils' to sys.path, whilst in beta build.
from pylslbci import PyLSLBCI

validMarkerStream = ["Markers"]
validDataStreams = ["UoHExGStream"]#Liam's EMG device"]

bci = PyLSLBCI()   #dataStreams = validDataStreams, markerStream = validMarkerStream)
if bci.lslScanner.CheckAvailableLSL():
    bci.StartTraining()

#with PyLSLBCI() as bci:
    #time.sleep(1)
    #if bci.lslScanner.CheckAvilableLSL():
    #    bci.StartTraining()