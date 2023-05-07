from collections import deque
import threading
from FeatureExtractor.FeatureExtractor import UniversalFeatureProcessor 

class FeatureProcessorThread(threading.Thread):
    def __init__(self, dataQueue, lock, freqbands = None, featureChoices = None):
        super().__init__()
        self.dataQueue = dataQueue
        self.lock = lock
        self.ufp = UniversalFeatureProcessor(freqbands = None, featureChoices = None)

    # need to add configurable number of desired epochs of each condition before including, if not set defaults from minimum viable (2???)
    def run(self):
        while True:
            data = self.dataQueue.get()
            with self.lock:
                print(data)
                self.ufp.ProcessEpoch(data, target, sr)

class DataReceiverThread(threading.Thread):
    """Responsible for receiving data from accepted LSL outlet, slices samples based on tmin+tmax basis, 
    starts counter for received samples after marker is received in ReceiveMarker.
    """
    startCounting = False
    currentMarker = ""
    def __init__(self, markerQueue, dataQueue, dataStreamInlet,  customWindowSettings, globalWindowSettings,  streamChsDropDict = []):
        super().__init__()
        self.markerQueue = markerQueue
        self.dataQueue = dataQueue
        self.dataStreamInlet = dataStreamInlet
        self.customWindowSettings = customWindowSettings
        self.globalWindowSettings = globalWindowSettings
        self.streamChsDropDict = streamChsDropDict
        self.sr = dataStreamInlet.info().nominal_srate()

    def run(self):
        posCount = 0
        chCount = self.dataStreamInlet.info().channel_count()
        if len(self.customWindowSettings.keys())>0:
            maxTime = max([self.customWindowSettings[x][2] + self.customWindowSettings[x][3] for x in self.customWindowSettings])
        else:
            maxTime = self.globalWindowSettings[2] + self.globalWindowSettings[3]
        fifoLength = int(self.dataStreamInlet.info().nominal_srate()*maxTime)
        dataFIFOs = [deque(maxlen=fifoLength) for ch in range(chCount)]
        while True:
            sample, timestamp = self.dataStreamInlet.pull_sample()
            for index in sorted(self.streamChsDropDict, reverse=True):
                del sample[index] # remove the desired channels from the sample
            for i,fifo in enumerate(dataFIFOs):
                fifo.append(sample[i])
            if self.startCounting:
                posCount+=1
                if posCount >= self.desiredCount:
                    # slice data fifo based on currentMarker tmin + tmax times    
                    if len(self.customWindowSettings.keys())>0: #  custom marker received
                        dataFIFOs = [d[int((self.customWindowSettings[self.currentMarker][2]+self.customWindowSettings[self.currentMarker][3]) * self.sr):]for d in dataFIFOs]
                    else:
                        dataFIFOs = [d[int((self.globalWindowSettings[2]+self.globalWindowSettings[3]) * self.sr):]for d in dataFIFOs]
                    self.dataQueue.put([dataFIFOs, self.currentMarker, self.sr, epochCount])
                    # reset flags and counters
                    self.startCounting = False
                    posCount = 0
    
    def ReceiveMarker(self, marker):
        if self.startCounting == False: # only one marker at a time allow, other in windowed timeframe ignored
            self.currentMarker = marker[0]
            if len(self.customWindowSettings.keys())>0: #  custom marker received
                if marker[0] in self.customWindowSettings.keys():
                    self.desiredCount = int(self.customWindowSettings[marker][3] * self.sr) # find number of samples after tmax to finish counting
                    self.startCounting = True
            else: # no custom markers set, use global settings
                self.desiredCount = int(self.globalWindowSettings[3] * self.sr) # find number of samples after tmax to finish counting
                self.startCounting = True

class MarkerReceiverThread(threading.Thread):
    """Receives Marker on chosen lsl Marker outlet. Pushes marker to 
    
    """
    def __init__(self, markerQueue,  markerStreamInlet,dataThreads):#, lock):
        super().__init__()
        self.markerQueue = markerQueue
        self.markerStreamInlet = markerStreamInlet
        self.dataThreads = dataThreads

    def run(self):
        while True:
            marker, timestamp = self.markerStreamInlet.pull_sample()
            self.markerQueue.put(marker)
            for thread in self.dataThreads:
                thread.ReceiveMarker(marker)