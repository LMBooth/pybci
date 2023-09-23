import threading, queue, time
from ..Utils.FeatureExtractor import GenericFeatureExtractor
from ..Utils.Logger import Logger
from ..Configuration.EpochSettings import GlobalEpochSettings
import copy

class FeatureProcessorThread(threading.Thread):
    tempDeviceEpochLogger = []
    def __init__(self, closeEvent, trainTestEvent, dataQueueTrain,dataQueueTest,
                featureQueueTest,featureQueueTrain,  totalDevices,markerCountRetrieveEvent,markerCountQueue, customEpochSettings = {}, 
                globalEpochSettings = GlobalEpochSettings(),logger = Logger(Logger.INFO), 
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
        self.logger = logger
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
                    dataFIFOs, currentMarker, sr, devCount = self.dataQueueTrain.get_nowait() #[sliceDataFIFOs, self.currentMarker, self.sr, self.devCount
                    if currentMarker in self.epochCounts:
                        self.epochCounts[currentMarker][1] += 1
                    else:
                        self.epochCounts[currentMarker] = [len(self.epochCounts.keys()),1]
                    target = self.epochCounts[currentMarker][0]
                    start = time.time()
                    features = self.featureExtractor.ProcessFeatures(dataFIFOs, sr, target) # allows custom epoch class to be passed
                    if (self.logger.level == Logger.TIMING):
                        end = time.time()
                        self.logger.log(Logger.TIMING, f" Feature Extraction time {end - start}")
                        if (end-start) >self.globalWindowSettings.windowLength:
                            self.logger.log(Logger.WARNING, f" Feature Extraction time > globalEpochSetting.windowLength, will create lag in classification output. Recommended to reduce channels, smapling rate, and features or reduce feature computational complexity.")
                    self.featureQueueTrain.put( [features, devCount, target, dict(self.epochCounts)] )
                except queue.Empty:
                    pass
            else:
                try:
                    dataFIFOs, sr, devCount = self.dataQueueTest.get_nowait() #[dataFIFOs, self.currentMarker, self.sr, ]
                    start = time.time()
                    features = self.featureExtractor.ProcessFeatures(dataFIFOs, sr, None)
                    if (self.logger.level == Logger.TIMING):
                        end = time.time()
                        self.logger.log(Logger.TIMING, f" Feature Extraction time {end - start}")
                        if (end-start) >self.globalWindowSettings.windowLength:
                            self.logger.log(Logger.WARNING, f" Feature Extraction time > globalEpochSetting.windowLength, will create lag in classification output. Recommended to reduce channels, smapling rate, and features or reduce feature computational complexity.")
                    self.featureQueueTest.put([features, devCount])
                except queue.Empty:
                    pass
