import threading

class MarkerThread(threading.Thread):
    """Receives Marker on chosen lsl Marker outlet. Pushes marker to data threads for framing epochs, 
    also sends markers to featureprocessing thread for epoch counting and multiple device synchronisation.
    
    """
    def __init__(self,closeEvent, trainTestEvent, markerStreamInlet, dataThreads, featureThreads):#, lock):
        super().__init__()
        self.trainTestEvent = trainTestEvent
        self.closeEvent = closeEvent
        self.markerStreamInlet = markerStreamInlet
        self.dataThreads = dataThreads
        self.featureThreads= featureThreads

    def run(self):
        while not self.closeEvent.is_set():
            marker, timestamp = self.markerStreamInlet.pull_sample(timeout = 10)
            if self.trainTestEvent.is_set(): # We're training!
                if marker != None:
                    marker = marker[0]
                    for thread in self.dataThreads:
                        thread.ReceiveMarker(marker, timestamp)
                    for thread in self.featureThreads:
                        thread.ReceiveMarker(marker, timestamp)
                    #self.featureThread.ReceiveMarker(marker, timestamp)
                else:
                    pass
                    # add levels of debug 
                    # print("PyBCI: LSL pull_sample timed out, no marker on stream...")
