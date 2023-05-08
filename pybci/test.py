import time, sys
sys.path.append('../')  # add the parent directory of 'utils' to sys.path, whilst in beta build.
from pybci import PyBCI

validMarkerStream = ["Markers"]
validDataStreams = ["UoHExGStream"]#Liam's EMG device"]

bci = PyBCI()   #dataStreams = validDataStreams, markerStream = validMarkerStream)
connected = False
#while(not connected):
#    if bci.lslScanner.CheckAvilableLSL():
#        connected = True

#while(connected):

#with PyBCI() as bci:
    #time.sleep(1)
    #if bci.lslScanner.CheckAvilableLSL():
    #    bci.StartTraining()