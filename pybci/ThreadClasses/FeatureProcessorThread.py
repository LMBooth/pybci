import threading, queue
from ..Utils.FeatureExtractor import GenericFeatureExtractor
from ..Configuration.EpochSettings import GlobalEpochSettings

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
                    #print(len(self.dataQueueTrain.get_nowait()))
                    dataFIFOs, currentMarker, sr, devCount = self.dataQueueTrain.get_nowait() #[sliceDataFIFOs, self.currentMarker, self.sr, self.devCount
                    #lastDevice = self.ReceiveMarker(currentMarker, devCount)
                    if currentMarker in self.epochCounts:
                        self.epochCounts[currentMarker][1] += 1
                    else:
                        self.epochCounts[currentMarker] = [len(self.epochCounts.keys()),1]
                    target = self.epochCounts[currentMarker][0]
                    features = self.featureExtractor.ProcessFeatures(dataFIFOs, sr, target) # allows custom epoch class to be passed
                    # add logic to ensure all devices epoch data has been received (totalDevices)
                    #if lastDevice:
                    self.featureQueueTrain.put( [features, devCount, target, self.epochCounts] )
                    #else:
                        # needs logic to append features together across devices (requires flattening of channels to 1d array of features)
                        # same for test mode
                    #    pass
                except queue.Empty:
                    pass
            else:
                try:
                    #print("we ain't dataQueueTesting: FeatureProcessorThread")
                    dataFIFOs, sr, devCount = self.dataQueueTest.get_nowait() #[dataFIFOs, self.currentMarker, self.sr, ]
                    features = self.featureExtractor.ProcessFeatures(dataFIFOs, sr, None)
                    self.featureQueueTest.put([features, devCount])
                except queue.Empty:
                    pass

    '''
    def ReceiveMarker(self, marker, devCount):
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
            self.tempDeviceEpochLogger[devCount] = 1
            if all(element == 1 for element in self.tempDeviceEpochLogger): # first device of epoch received
                self.tempDeviceEpochLogger = [0 for x in range(self.totalDevices)]
                return True
            else:
                return False
            # we need to log each type of device and make sure the same number is received of each
    '''