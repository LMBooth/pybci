import time, sys
sys.path.append('../')  # add the parent directory of 'utils' to sys.path, whilst in beta build.
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
        time.sleep(1)
except KeyboardInterrupt: # allow user to break while loop
    pass