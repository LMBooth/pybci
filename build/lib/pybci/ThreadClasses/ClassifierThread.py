from ..Utils.Classifier import Classifier 
from ..Utils.Logger import Logger
import queue,threading, time
import numpy as np

class ClassifierThread(threading.Thread):
    features = np.array([])#[]
    targets = np.array([])
    mode = "train"
    guess = " "
    #epochCountsc = {} 
    def __init__(self, closeEvent,trainTestEvent, featureQueueTest,featureQueueTrain, classifierInfoQueue, classifierInfoRetrieveEvent, 
                 classifierGuessMarkerQueue, classifierGuessMarkerEvent, queryFeaturesQueue, queryFeaturesEvent,
                 logger = Logger(Logger.INFO), numStreamDevices = 1,
                 minRequiredEpochs = 10, clf = None, model = None, torchModel = None):
        super().__init__()
        self.trainTestEvent = trainTestEvent # responsible for tolling between train and test mode
        self.closeEvent = closeEvent # responsible for cosing threads
        self.featureQueueTest = featureQueueTest # gets feature data from feature processing thread
        self.featureQueueTrain = featureQueueTrain # gets feature data from feature processing thread
        self.classifier = Classifier(clf = clf, model = model, torchModel = torchModel) # sets classifier class, if clf and model passed, defaults to clf and sklearn
        self.minRequiredEpochs = minRequiredEpochs # the minimum number of epochs required for classifier attempt
        self.classifierInfoRetrieveEvent = classifierInfoRetrieveEvent
        self.classifierInfoQueue = classifierInfoQueue
        self.classifierGuessMarkerQueue = classifierGuessMarkerQueue
        self.classifierGuessMarkerEvent = classifierGuessMarkerEvent
        self.queryFeaturesQueue = queryFeaturesQueue
        self.queryFeaturesEvent = queryFeaturesEvent
        self.numStreamDevices = numStreamDevices
        self.logger = logger

    def run(self):
        epochCountsc={}
        if self.numStreamDevices > 1:
            tempdatatrain = {}
            tempdatatest = {}
        while not self.closeEvent.is_set():
            if self.trainTestEvent.is_set(): # We're training!
                if self.featureQueueTrain.empty():
                    if len(epochCountsc) > 1: # check if there is more then one test condition
                        minNumKeyEpochs = min([epochCountsc[key][1] for key in epochCountsc]) # check minimum viable number of training eochs have been obtained
                        if minNumKeyEpochs < self.minRequiredEpochs:
                            pass
                        else: 
                            start = time.time()
                            self.classifier.TrainModel(self.features, self.targets)
                            if (self.logger.level == Logger.TIMING):
                                end = time.time()
                                self.logger.log(Logger.TIMING, f" classifier training time {end - start}")
                    if self.classifierGuessMarkerEvent.is_set():
                        self.classifierGuessMarkerQueue.put(self.guess)
                else:
                    try:
                        featuresSingle, devCount, target, epochCountsc = self.featureQueueTrain.get_nowait() #[dataFIFOs, self.currentMarker, self.sr, self.dataType]
                        if self.numStreamDevices > 1: # Collects multiple data strems feature sets and synchronise here
                            tempdatatrain[devCount] = featuresSingle
                            if len(tempdatatrain) == self.numStreamDevices:
                                flattened_list = np.array([item for sublist in tempdatatrain.values() for item in sublist])
                                tempdatatrain = {}
                                self.targets = np.append(self.targets, [target], axis = 0)
                                #self.features = np.append(self.features, [flattened_list], axis = 0)
                                if self.features.shape[0] == 0:
                                    self.features = self.features.reshape((0,) + flattened_list.shape)
                                self.features = np.append(self.features, [flattened_list], axis=0)
                            # need to check if all device data is captured, then flatten and append  
                        else: # Only one device to collect from
                            if self.features.shape[0] == 0:
                                self.features = self.features.reshape((0,) + featuresSingle.shape)
                            self.targets = np.append(self.targets, [target], axis = 0)
                            self.features = np.append(self.features, [featuresSingle], axis = 0)
                    except queue.Empty:
                        pass
            else: # We're testing!
                try:
                    featuresSingle, devCount = self.featureQueueTest.get_nowait() #[dataFIFOs, self.currentMarker, self.sr, self.dataType]
                    if self.numStreamDevices > 1:
                        tempdatatest[devCount] = featuresSingle
                        if len(tempdatatest) == self.numStreamDevices:
                            flattened_list = []
                            flattened_list = np.array([item for sublist in tempdatatest.values() for item in sublist])
                            tempdatatest = {}
                            start = time.time()
                            self.guess = self.classifier.TestModel(flattened_list)
                            if (self.logger.level == Logger.TIMING):
                                end = time.time()
                                self.logger.log(Logger.TIMING, f" classifier testing time {end - start}")
                    else:
                        start = time.time()
                        self.guess = self.classifier.TestModel(featuresSingle)
                        if (self.logger.level == Logger.TIMING):
                            end = time.time()
                            self.logger.log(Logger.TIMING, f" classifier testing time {end - start}")
                    if self.classifierGuessMarkerEvent.is_set():
                        self.classifierGuessMarkerQueue.put(self.guess)
                except queue.Empty:
                    pass
            if self.classifierInfoRetrieveEvent.is_set():
                a = self.classifier.accuracy
                classdata = {
                    "clf":self.classifier.clf,
                    "model":self.classifier.model,
                    "torchModel":self.classifier.torchModel,
                    "accuracy":a
                    }
                self.classifierInfoQueue.put(classdata) 
            if self.queryFeaturesEvent.is_set():
                featureData = {
                    "features":self.features,
                    "targets":self.targets
                }
                self.queryFeaturesQueue.put(featureData) 