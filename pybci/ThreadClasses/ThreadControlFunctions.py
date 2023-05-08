from collections import deque
import itertools
import threading
from Utils.FeatureExtractor import FeatureExtractor
from Utils.Classifier import Classifier 
import numpy as np
import queue

# need to add configurable number of desired epochs of each condition before including, if not set defaults from minimum viable (2???)
# self.epochCounts has total counts of each epoch available
class ClassifierThread(threading.Thread):
    features = []
    targets = []
    mode = "train"
    
    def __init__(self, closeEvent,trainTestEvent, featureQueue, minRequiredEpochs = 10, clf = None, model = None):
        super().__init__()
        self.trainTestEvent = trainTestEvent # responsible for tolling between train and test mode
        self.closeEvent = closeEvent # responsible for cosing threads
        self.featureQueue = featureQueue # gets feature data from feature processing thread
        self.classifier = Classifier(clf = clf, model = model) # sets classifier class, if clf and model passed, defaults to clf and sklearn
        self.minRequiredEpochs = minRequiredEpochs # the minimum number of epochs required for classifier attempt

    def run(self):
        while not self.closeEvent.is_set():
            if self.trainTestEvent.is_set(): # We're training!
                featuresSingle, target, epochCounts = self.featureQueue.get() #[dataFIFOs, self.currentMarker, self.sr, self.dataType]
                self.targets.append(target)
                self.features.append(featuresSingle)
                print(epochCounts)
                minNumKeyEpochs = min([epochCounts[key][1] for key in epochCounts])
                if minNumKeyEpochs < self.minRequiredEpochs:
                    pass
                elif minNumKeyEpochs == self.minRequiredEpochs:
                    self.classifier.CompileModel(self.features, self.targets)
                else:
                    self.classifier.UpdateModel(featuresSingle,target)
            else: # We're testing!
                featuresSingle = self.featureQueue.get() #[dataFIFOs, self.currentMarker, self.sr, self.dataType]
                self.classifier.TestModel(featuresSingle)


class FeatureProcessorThread(threading.Thread):
    def __init__(self, closeEvent, trainTestEvent, dataQueue, featureQueue,  totalDevices, lock, customWindowSettings = {}, freqbands = None, featureChoices = None):
        super().__init__()
        self.trainTestEvent = trainTestEvent
        self.closeEvent = closeEvent
        self.dataQueue = dataQueue
        self.featureQueue = featureQueue
        self.lock = lock
        self.ufp = FeatureExtractor(freqbands = None, featureChoices = None)
        self.totalDevices = totalDevices
        self.epochCounts = {}
        self.customWindowSettings = customWindowSettings
        
    def run(self):
        while not self.closeEvent.is_set():
            if self.trainTestEvent.is_set(): # We're training!
                try:

                    dataFIFOs, currentMarker, sr, dataType = self.dataQueue.get_nowait() #[dataFIFOs, self.currentMarker, self.sr, self.dataType]

                    # This is where epoch slice setting should be implemented

                    with self.lock:
                        print([currentMarker, sr, dataType])
                        print(np.array(dataFIFOs).shape)
                        print(dataType)
                    target = self.epochCounts[currentMarker][0]
                    # could maybe allow custom dataType dict to select epoch processing pipeline. This is where new libraries will be added
                    if (dataType == "EEG" or dataType == "EMG"): # found the same can be used for EMG
                        features = self.ufp.ProcessGeneralEpoch(dataFIFOs, sr)
                    elif (dataType == "ECG"):
                        features = self.ufp.ProcessECGFeatures(dataFIFOs, sr)
                    elif (dataType == "Gaze"):
                        features = self.ufp.ProcessPupilFeatures(dataFIFOs)
                    # add logic to ensure all devices epoch data has been received (totalDevices)
                    #self.epochCountsdata[1]
                    self.featureQueue.put( [features, target, self.epochCounts] )
                except queue.Empty:
                    break
            else:
                try:
                    dataFIFOs, sr, dataType = self.dataQueue.get_nowait() #[dataFIFOs, self.currentMarker, self.sr, self.dataType]
                    self.featureQueue.put( [features] )
                except self.dataQueue.Empty:
                    break

    def ReceiveMarker(self, marker):
        """ Tracks count of epoch markers in dict self.epochCounts - used for syncing data between multiple devices in function self.run() """
        if len(self.customWindowSettings.keys())>0: #  custom marker received
            if marker[0] in self.customWindowSettings.keys():
                if marker in self.epochCounts:
                    self.epochCounts[marker][1] += 1
                else:
                    self.epochCounts[marker] = [len(self.epochCounts.keys()),1]
        else: # no custom markers set, use global settings
            if marker in self.epochCounts:
                self.epochCounts[marker][1] += 1
            else:
                self.epochCounts[marker] = [len(self.epochCounts.keys()),1]

class DataReceiverThread(threading.Thread):
    """Responsible for receiving data from accepted LSL outlet, slices samples based on tmin+tmax basis, 
    starts counter for received samples after marker is received in ReceiveMarker.
    """
    startCounting = False
    currentMarker = ""
    def __init__(self, closeEvent, trainTestEvent, dataQueue, dataStreamInlet,  customWindowSettings, globalWindowSettings,  streamChsDropDict = []):
        super().__init__()
        self.trainTestEvent = trainTestEvent
        self.closeEvent = closeEvent
        self.dataQueue = dataQueue
        self.dataStreamInlet = dataStreamInlet
        self.customWindowSettings = customWindowSettings
        self.globalWindowSettings = globalWindowSettings
        self.streamChsDropDict = streamChsDropDict
        self.sr = dataStreamInlet.info().nominal_srate()
        self.dataType = dataStreamInlet.info().type()
        
    def run(self):
        posCount = 0
        chCount = self.dataStreamInlet.info().channel_count()
        if len(self.customWindowSettings.keys())>0:
            maxTime = max([self.customWindowSettings[x][2] + self.customWindowSettings[x][3] for x in self.customWindowSettings])
        else:
            maxTime = self.globalWindowSettings[2] + self.globalWindowSettings[3]
        fifoLength = int(self.dataStreamInlet.info().nominal_srate()*maxTime)
        dataFIFOs = [deque(maxlen=fifoLength) for ch in range(chCount - len(self.streamChsDropDict))]
        while not self.closeEvent.is_set():
            sample, timestamp = self.dataStreamInlet.pull_sample()
            for index in sorted(self.streamChsDropDict, reverse=True):
                del sample[index] # remove the desired channels from the sample
            for i,fifo in enumerate(dataFIFOs):
                fifo.append(sample[i])
            if self.trainTestEvent.is_set(): # We're training!
                if self.startCounting:
                    posCount+=1
                    if posCount >= self.desiredCount:
                        # slice data fifo based on currentMarker tmin + tmax times    
                        if len(self.customWindowSettings.keys())>0: #  custom marker received
                            sliceDataFIFOs = [list(itertools.islice(d, fifoLength - int((self.customWindowSettings[self.currentMarker][2]+self.customWindowSettings[self.currentMarker][3]) * self.sr), fifoLength))for d in dataFIFOs]
                        else:
                            sliceDataFIFOs = [list(itertools.islice(d, fifoLength - int((self.globalWindowSettings[2]+self.globalWindowSettings[3]) * self.sr), fifoLength)) for d in dataFIFOs]
                        self.dataQueue.put([sliceDataFIFOs, self.currentMarker, self.sr, self.dataType])
                        # reset flags and counters
                        self.startCounting = False
                        posCount = 0
            else:
                # ooooooo this is gonna be interesting, how do i slice... i think i need a universal window size...
                
                self.dataQueue.put([sliceDataFIFOs, self.sr, self.dataType])

    def ReceiveMarker(self, marker):
        if self.startCounting == False: # only one marker at a time allow, other in windowed timeframe ignored
            self.currentMarker = marker
            if len(self.customWindowSettings.keys())>0: #  custom marker received
                if marker in self.customWindowSettings.keys():
                    self.desiredCount = int(self.customWindowSettings[marker][3] * self.sr) # find number of samples after tmax to finish counting
                    self.startCounting = True
            else: # no custom markers set, use global settings
                self.desiredCount = int(self.globalWindowSettings[3] * self.sr) # find number of samples after tmax to finish counting
                self.startCounting = True

class MarkerReceiverThread(threading.Thread):
    """Receives Marker on chosen lsl Marker outlet. Pushes marker to data threads for framing epochs, 
    also sends markers to featureprocessing thread for epoch counting and multiple device synchronisation.
    
    """
    def __init__(self,closeEvent, trainTestEvent, markerStreamInlet, dataThreads, featureThread):#, lock):
        super().__init__()
        self.trainTestEvent = trainTestEvent
        self.closeEvent = closeEvent
        self.markerStreamInlet = markerStreamInlet
        self.dataThreads = dataThreads
        self.featureThread = featureThread

    def run(self):
        while not self.closeEvent.is_set():
            marker, timestamp = self.markerStreamInlet.pull_sample()
            marker = marker[0]
            for thread in self.dataThreads:
                thread.ReceiveMarker(marker)
            self.featureThread.ReceiveMarker(marker)