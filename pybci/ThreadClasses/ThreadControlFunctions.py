from collections import deque
import itertools
import threading
from ..Utils.FeatureExtractor import GenericFeatureExtractor
from ..Utils.Classifier import Classifier 
import numpy as np
import queue
from ..Configuration.EpochSettings import GlobalEpochSettings, IndividualEpochSetting
from ..Configuration.FeatureSettings import GeneralFeatureChoices
class ClassifierThread(threading.Thread):
    features = []
    targets = []
    mode = "train"
    guess = None
    epochCounts = {} 
    def __init__(self, closeEvent,trainTestEvent, featureQueueTest,featureQueueTrain, classifierInfoQueue, classifierInfoRetrieveEvent, 
                 classifierGuessMarkerQueue, classifierGuessMarkerEvent,
                 minRequiredEpochs = 10, clf = None, model = None):
        super().__init__()
        self.trainTestEvent = trainTestEvent # responsible for tolling between train and test mode
        self.closeEvent = closeEvent # responsible for cosing threads
        self.featureQueueTest = featureQueueTest # gets feature data from feature processing thread
        self.featureQueueTrain = featureQueueTrain # gets feature data from feature processing thread
        self.classifier = Classifier(clf = clf, model = model) # sets classifier class, if clf and model passed, defaults to clf and sklearn
        self.minRequiredEpochs = minRequiredEpochs # the minimum number of epochs required for classifier attempt
        self.classifierInfoRetrieveEvent = classifierInfoRetrieveEvent
        self.classifierInfoQueue = classifierInfoQueue
        self.classifierGuessMarkerQueue = classifierGuessMarkerQueue
        self.classifierGuessMarkerEvent = classifierGuessMarkerEvent
    def run(self):
        while not self.closeEvent.is_set():
            if self.trainTestEvent.is_set(): # We're training!
                try:
                    featuresSingle, target, self.epochCounts = self.featureQueueTrain.get_nowait() #[dataFIFOs, self.currentMarker, self.sr, self.dataType]
                    self.targets.append(target)
                    self.features.append(featuresSingle)
                    if len(self.epochCounts) > 1: # check if there is more then one test condition
                        minNumKeyEpochs = min([self.epochCounts[key][1] for key in self.epochCounts]) # check minimum viable number of training eochs have been obtained
                        #print("minNumKeyEpochs"+str(minNumKeyEpochs))
                        if minNumKeyEpochs < self.minRequiredEpochs:
                            pass
                        else: 
                            self.classifier.TrainModel(self.features, self.targets)
                    if self.classifierGuessMarkerEvent.is_set():
                        self.classifierGuessMarkerQueue.put(None)
                except queue.Empty:
                    pass
            else: # We're testing!
                try:
                    featuresSingle = self.featureQueueTest.get_nowait() #[dataFIFOs, self.currentMarker, self.sr, self.dataType]
                    self.guess = self.classifier.TestModel(featuresSingle)
                    if self.classifierGuessMarkerEvent.is_set():
                        self.classifierGuessMarkerQueue.put(self.guess)
                except queue.Empty:
                    pass
            if self.classifierInfoRetrieveEvent.is_set():
                classdata = {
                    "clf":self.classifier.clf,
                    "model":self.classifier.model,
                    "accuracy":self.classifier.accuracy
                    }
                self.classifierInfoQueue.put(classdata) 
            

class FeatureProcessorThread(threading.Thread):
    tempDeviceEpochLogger = []
    def __init__(self, closeEvent, trainTestEvent, dataQueueTrain,dataQueueTest,
                featureQueueTest,featureQueueTrain,  totalDevices,markerCountRetrieveEvent,markerCountQueue, customEpochSettings = {}, 
                globalEpochSettings = GlobalEpochSettings(),
                featureExtractor = GenericFeatureExtractor()):
        super().__init__()
        self.markerCountQueue = markerCountQueue
        self.trainTestEvent = trainTestEvent
        self.closeEvent = closeEvent
        self.dataQueueTrain = dataQueueTrain
        self.dataQueueTest = dataQueueTest
        self.featureQueueTrain = featureQueueTrain
        self.featureQueueTest = featureQueueTest
        self.featureExtractor = featureExtractor
        self.totalDevices = totalDevices
        self.markerCountRetrieveEvent = markerCountRetrieveEvent
        self.epochCounts = {}
        self.customEpochSettings = customEpochSettings
        self.globalWindowSettings = globalEpochSettings
        self.tempDeviceEpochLogger = [0 for x in range(self.totalDevices)]
        
    def run(self):
        while not self.closeEvent.is_set():
            if self.markerCountRetrieveEvent.is_set():
                self.markerCountQueue.put(self.epochCounts)
            if self.trainTestEvent.is_set(): # We're training!
                try:
                    dataFIFOs, currentMarker, sr, dataType, qNumber = self.dataQueueTrain.get_nowait() #[dataFIFOs, self.currentMarker, self.sr, self.dataType]
                    lastDevice = self.ReceiveMarker(currentMarker, qNumber)
                    target = self.epochCounts[currentMarker][0]
                    # could maybe allow custom dataType dict to select epoch processing pipeline. This is where new libraries will be added

                    features = self.featureExtractor.ProcessFeatures(dataFIFOs, sr)

                    # add logic to ensure all devices epoch data has been received (totalDevices)
                    if lastDevice:
                        self.featureQueueTrain.put( [features, target, self.epochCounts] )
                    else:
                        # needs logic to append features together across devices (requires flattening of channels to 1d array of features)
                        # same for test mode
                        pass
                except queue.Empty:
                    pass
            else:
                try:
                    #print("we ain't dataQueueTesting: FeatureProcessorThread")
                    dataFIFOs, sr, dataType, qNumber = self.dataQueueTest.get_nowait() #[dataFIFOs, self.currentMarker, self.sr, self.dataType]
                    if (dataType == "EEG" or dataType == "EMG"): # found the same can be used for EMG
                        features = self.ufp.ProcessGeneralEpoch(dataFIFOs, sr)
                    elif (dataType == "ECG"):
                        features = self.ufp.ProcessECGFeatures(dataFIFOs, sr)
                    elif (dataType == "Gaze"):
                        features = self.ufp.ProcessPupilFeatures(dataFIFOs)
                    self.featureQueueTest.put(features)
                except queue.Empty:
                    pass

    def ReceiveMarker(self, marker, qNumber):
        """ Tracks count of epoch markers in dict self.epochCounts - used for syncing data between multiple devices in function self.run() """
        if self.totalDevices == 1:
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
            return True # alays true as only one device to receive
        else:
            if all(element == 0 for element in self.tempDeviceEpochLogger): # first device of epoch received so add to epochCounts
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
            self.tempDeviceEpochLogger[qNumber] = 1
            if all(element == 1 for element in self.tempDeviceEpochLogger): # first device of epoch received
                self.tempDeviceEpochLogger = [0 for x in range(self.totalDevices)]
                return True
            else:
                return False
            # we need to log each type of device and make sure the same number is received of each

class DataReceiverThread(threading.Thread):
    """Responsible for receiving data from accepted LSL outlet, slices samples based on tmin+tmax basis, 
    starts counter for received samples after marker is received in ReceiveMarker.
    """
    startCounting = False
    currentMarker = ""
    def __init__(self, closeEvent, trainTestEvent, dataQueueTrain,dataQueueTest, dataStreamInlet,  customEpochSettings, globalEpochSettings,devCount,  streamChsDropDict = []):
        super().__init__()
        self.trainTestEvent = trainTestEvent
        self.closeEvent = closeEvent
        self.dataQueueTrain = dataQueueTrain
        self.dataQueueTest = dataQueueTest
        self.dataStreamInlet = dataStreamInlet
        self.customEpochSettings = customEpochSettings
        self.globalEpochSettings = globalEpochSettings
        self.streamChsDropDict = streamChsDropDict
        self.sr = dataStreamInlet.info().nominal_srate()
        self.dataType = dataStreamInlet.info().type()
        self.devCount = devCount # used for tracking which device is sending data to feature extractor
        
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
                    
                    if self.startCounting: # we received a marker
                        posCount+=1
                        if posCount >= self.desiredCount:  # enough samples are in FIFO, chop up and put in dataqueue
                            if len(self.customEpochSettings.keys())>0: #  custom marker received
                                if self.customEpochSettings[self.currentMarker].splitCheck: # slice epochs in to overlapping time windows
                                    window_samples =int(self.customEpochSettings[self.currentMarker].windowLength * self.sr) #number of samples in each window
                                    increment = int((1-self.customEpochSettings[self.currentMarker].windowOverlap)*window_samples) # if windows overlap each other by how many samples
                                    while posCount - window_samples > 0:
                                        sliceDataFIFOs = [list(itertools.islice(d, posCount - window_samples, posCount)) for d in dataFIFOs]
                                        self.dataQueueTrain.put([sliceDataFIFOs, self.currentMarker, self.sr, self.dataType, self.devCount])
                                        posCount-=increment
                                else: # don't slice just take tmin to tmax time
                                    sliceDataFIFOs = [list(itertools.islice(d, fifoLength - int((self.customEpochSettings[self.currentMarker].tmax+self.customEpochSettings[self.currentMarker].tmin) * self.sr), fifoLength))for d in dataFIFOs]
                                    self.dataQueueTrain.put([sliceDataFIFOs, self.currentMarker, self.sr, self.dataType, self.devCount])
                            else:
                                if self.globalEpochSettings.splitCheck: # slice epochs in to overlapping time windows
                                    window_samples =int(self.globalEpochSettings.windowLength * self.sr) #number of samples in each window
                                    increment = int((1-self.globalEpochSettings.windowOverlap)*window_samples) # if windows overlap each other by how many samples
                                    startCount = self.desiredCount + int(self.globalEpochSettings.tmin * self.sr)
                                    while startCount - window_samples > 0:
                                        sliceDataFIFOs = [list(itertools.islice(d, startCount - window_samples, startCount)) for d in dataFIFOs]
                                        self.dataQueueTrain.put([sliceDataFIFOs, self.currentMarker, self.sr, self.dataType,self.devCount])
                                        startCount-=increment
                                else: # don't slice just take tmin to tmax time
                                    sliceDataFIFOs = [list(itertools.islice(d, fifoLength - int((self.globalEpochSettings.tmin+self.globalEpochSettings.tmax) * self.sr), fifoLength)) for d in dataFIFOs]
                                    self.dataQueueTrain.put([sliceDataFIFOs, self.currentMarker, self.sr, self.dataType, self.devCount])
                            # reset flags and counters
                            posCount = 0
                            self.startCounting = False
                else: # in Test mode
                    posCount+=1
                    if self.globalEpochSettings.splitCheck:
                        window_samples = int(self.globalEpochSettings.windowLength * self.sr) #number of samples in each window
                    else:
                        window_samples = int((self.globalEpochSettings.tmin+self.globalEpochSettings.tmax) * self.sr)
                    if posCount >= window_samples:
                        sliceDataFIFOs = [list(itertools.islice(d, fifoLength-window_samples, fifoLength)) for d in dataFIFOs]
                        if self.globalEpochSettings.splitCheck:
                            posCount = int((1-self.globalEpochSettings.windowOverlap)*window_samples) # offset poscoutn based on window overlap 
                        else:
                            posCount = 0
                            
                        self.dataQueueTest.put([sliceDataFIFOs, self.sr, self.dataType, self.devCount])
            else:
                pass
                # add levels of debug 
                # print("PyBCI: LSL pull_sample timed out, no data on stream...")

    def ReceiveMarker(self, marker, timestamp): # timestamp will be used for non sample rate specific devices (pupil-labs gazedata)
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
            if self.trainTestEvent.is_set(): # We're training!
                if marker != None:
                    marker = marker[0]
                    for thread in self.dataThreads:
                        thread.ReceiveMarker(marker, timestamp)
                    #self.featureThread.ReceiveMarker(marker, timestamp)
                else:
                    pass
                    # add levels of debug 
                    # print("PyBCI: LSL pull_sample timed out, no marker on stream...")
