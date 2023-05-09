from collections import deque
import itertools
import threading
from Utils.FeatureExtractor import FeatureExtractor
from Utils.Classifier import Classifier 
import numpy as np
import queue
from Configuration.EpochSettings import GlobalEpochSettings, IndividualEpochSetting
from Configuration.FeatureSettings import FeatureChoices
# need to add configurable number of desired epochs of each condition before including, if not set defaults from minimum viable (2???)
# self.epochCounts has total counts of each epoch available
class ClassifierThread(threading.Thread):
    features = []
    targets = []
    mode = "train"
    
    def __init__(self, closeEvent,trainTestEvent, featureQueue, lock, minRequiredEpochs = 10, clf = None, model = None):
        super().__init__()
        self.trainTestEvent = trainTestEvent # responsible for tolling between train and test mode
        self.closeEvent = closeEvent # responsible for cosing threads
        self.featureQueue = featureQueue # gets feature data from feature processing thread
        self.classifier = Classifier(clf = clf, model = model) # sets classifier class, if clf and model passed, defaults to clf and sklearn
        self.minRequiredEpochs = minRequiredEpochs # the minimum number of epochs required for classifier attempt
        self.lock = lock

    def run(self):
        while not self.closeEvent.is_set():
            if self.trainTestEvent.is_set(): # We're training!
                try:
                    featuresSingle, target, epochCounts = self.featureQueue.get_nowait() #[dataFIFOs, self.currentMarker, self.sr, self.dataType]
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
                        self.classifier.TestModel(featuresSingle) # maybe make this toggleable?
                except queue.Empty:
                    pass
            else: # We're testing!
                try:
                    featuresSingle = self.featureQueue.get_nowait() #[dataFIFOs, self.currentMarker, self.sr, self.dataType]
                    self.classifier.TestModel(featuresSingle)
                except queue.Empty:
                    pass


class FeatureProcessorThread(threading.Thread):
    def __init__(self, closeEvent, trainTestEvent, dataQueue, featureQueue,  totalDevices,markerCountRetrieveEvent,markerCountQueue, customEpochSettings = {}, 
                 globalEpochSettings = GlobalEpochSettings(),freqbands = [[1.0, 4.0], [4.0, 8.0], [8.0, 12.0], [12.0, 20.0]], featureChoices = FeatureChoices()):
        super().__init__()
        self.markerCountQueue = markerCountQueue
        self.trainTestEvent = trainTestEvent
        self.closeEvent = closeEvent
        self.dataQueue = dataQueue
        self.featureQueue = featureQueue
        self.ufp = FeatureExtractor(freqbands = freqbands, featureChoices = featureChoices)
        self.totalDevices = totalDevices
        self.markerCountRetrieveEvent = markerCountRetrieveEvent
        self.epochCounts = {}
        self.customEpochSettings = customEpochSettings
        self.globalWindowSettings = GlobalEpochSettings
        
    def run(self):
        while not self.closeEvent.is_set():
            if self.markerCountRetrieveEvent.is_set():
                self.markerCountQueue.put(self.epochCounts)
            if self.trainTestEvent.is_set(): # We're training!
                try:
                    dataFIFOs, currentMarker, sr, dataType = self.dataQueue.get_nowait() #[dataFIFOs, self.currentMarker, self.sr, self.dataType]
                    target = self.epochCounts[currentMarker][0]
                    # could maybe allow custom dataType dict to select epoch processing pipeline. This is where new libraries will be added
                    if (dataType == "EEG" or dataType == "EMG"): # found the same can be used for EMG
                        features = self.ufp.ProcessGeneralEpoch(dataFIFOs, sr)
                    elif (dataType == "ECG"):
                        features = self.ufp.ProcessECGFeatures(dataFIFOs, sr)
                    elif (dataType == "Gaze"):
                        features = self.ufp.ProcessPupilFeatures(dataFIFOs)
                    # add logic to ensure all devices epoch data has been received (totalDevices)
                    if self.totalDevices == 1:
                        #self.epochCountsdata[1]
                        self.featureQueue.put( [features, target, self.epochCounts] )
                except queue.Empty:
                    pass
            else:
                try:
                    dataFIFOs, sr, dataType = self.dataQueue.get_nowait() #[dataFIFOs, self.currentMarker, self.sr, self.dataType]
                    if (dataType == "EEG" or dataType == "EMG"): # found the same can be used for EMG
                        features = self.ufp.ProcessGeneralEpoch(dataFIFOs, sr)
                    elif (dataType == "ECG"):
                        features = self.ufp.ProcessECGFeatures(dataFIFOs, sr)
                    elif (dataType == "Gaze"):
                        features = self.ufp.ProcessPupilFeatures(dataFIFOs)
                    self.featureQueue.put( [features] )
                except queue.Empty:
                    pass

    def ReceiveMarker(self, marker):
        """ Tracks count of epoch markers in dict self.epochCounts - used for syncing data between multiple devices in function self.run() """
        if len(self.customEpochSettings.keys())>0: #  custom marker received
            if marker in self.customEpochSettings.keys():
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
    def __init__(self, closeEvent, trainTestEvent, dataQueue, dataStreamInlet,  customEpochSettings, globalEpochSettings,  streamChsDropDict = []):
        super().__init__()
        self.trainTestEvent = trainTestEvent
        self.closeEvent = closeEvent
        self.dataQueue = dataQueue
        self.dataStreamInlet = dataStreamInlet
        self.customEpochSettings = customEpochSettings
        self.globalEpochSettings = globalEpochSettings
        self.streamChsDropDict = streamChsDropDict
        self.sr = dataStreamInlet.info().nominal_srate()
        self.dataType = dataStreamInlet.info().type()
        
    def run(self):
        posCount = 0
        chCount = self.dataStreamInlet.info().channel_count()
        maxTime = (self.globalEpochSettings.tmin + self.globalEpochSettings.tmax)
        if len(self.customEpochSettings.keys())>0:
            if  max([self.customEpochSettings[x].tmin + self.customEpochSettings[x].tmax for x in self.customEpochSettings]) > maxTime:
                maxTime = max([self.customEpochSettings[x].tmin + self.customEpochSettings[x].tmax for x in self.customEpochSettings])
        fifoLength = int(self.dataStreamInlet.info().nominal_srate()*maxTime)
        #print(fifoLength)
        dataFIFOs = [deque(maxlen=fifoLength) for ch in range(chCount - len(self.streamChsDropDict))]
        while not self.closeEvent.is_set():
            sample, timestamp = self.dataStreamInlet.pull_sample(timeout = 1)
            if sample != None:
                for index in sorted(self.streamChsDropDict, reverse=True):
                    del sample[index] # remove the desired channels from the sample
                for i,fifo in enumerate(dataFIFOs):
                    fifo.append(sample[i])
                if self.trainTestEvent.is_set(): # We're training!
                    if self.startCounting:
                        posCount+=1
                        if posCount >= self.desiredCount:
                            ##############################
                            # Update required here!!!
                            # slice data fifo based on currentMarker tmin + tmax times    
                            if len(self.customEpochSettings.keys())>0: #  custom marker received
                                sliceDataFIFOs = [list(itertools.islice(d, fifoLength - int((self.customEpochSettings[self.currentMarker].tmax+self.customEpochSettings[self.currentMarker].tmin) * self.sr), fifoLength))for d in dataFIFOs]
                            else:
                                sliceDataFIFOs = [list(itertools.islice(d, fifoLength - int((self.globalEpochSettings.tmin+self.globalEpochSettings.tmax) * self.sr), fifoLength)) for d in dataFIFOs]
                            self.dataQueue.put([sliceDataFIFOs, self.currentMarker, self.sr, self.dataType])
                            # reset flags and counters
                            self.startCounting = False
                            posCount = 0
                else:
                    posCount+=1
                    if posCount >= int(self.globalEpochSettings.windowLength * self.sr):
                        posCount = 0
                    # ooooooo this is gonna be interesting, how do i slice... i think i need a universal window size...
                    # relates to required update above too
                        self.dataQueue.put([sliceDataFIFOs, self.sr, self.dataType])
            else:
                print("PyBCI: LSL pull_sample timed out, no data on stream...")

    def ReceiveMarker(self, marker):
        #print(marker)
        if self.startCounting == False: # only one marker at a time allow, other in windowed timeframe ignored
            self.currentMarker = marker
            if len(self.customEpochSettings.keys())>0: #  custom marker received
                if marker in self.customEpochSettings.keys():
                    self.desiredCount = int(self.customEpochSettings[marker].tmax * self.sr) # find number of samples after tmax to finish counting
                    self.startCounting = True
            else: # no custom markers set, use global settings
                self.desiredCount = int(self.globalEpochSettings.tmax * self.sr) # find number of samples after tmax to finish counting
                self.startCounting = True
            #print(self.desiredCount)


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
            marker, timestamp = self.markerStreamInlet.pull_sample(timeout = 10)
            if marker != None:
                marker = marker[0]
                for thread in self.dataThreads:
                    thread.ReceiveMarker(marker)
                self.featureThread.ReceiveMarker(marker)
            else:
                print("PyBCI: LSL pull_sample timed out, no marker on stream...")