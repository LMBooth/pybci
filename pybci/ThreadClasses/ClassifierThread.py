from ..Utils.Classifier import Classifier 
import queue,threading, time

class ClassifierThread(threading.Thread):
    features = []
    targets = []
    mode = "train"
    guess = None
    epochCounts = {} 
    def __init__(self, closeEvent,trainTestEvent, featureQueueTest,featureQueueTrain, classifierInfoQueue, classifierInfoRetrieveEvent, 
                 classifierGuessMarkerQueue, classifierGuessMarkerEvent, printDebug = True, numStreamDevices = 1,
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
        self.numStreamDevices = numStreamDevices
        self.printDebug = printDebug

    def run(self):
        if self.numStreamDevices > 1:
            tempdatatrain = {}
            tempdatatest = {}
        while not self.closeEvent.is_set():
            if self.trainTestEvent.is_set(): # We're training!
                try:
                    featuresSingle, devCount, target, self.epochCounts = self.featureQueueTrain.get_nowait() #[dataFIFOs, self.currentMarker, self.sr, self.dataType]
                    if self.numStreamDevices > 1:
                        tempdatatrain[devCount] = featuresSingle
                        if len(tempdatatrain) == self.numStreamDevices:
                            #print(tempdatatrain)
                            #flattened_list = []
                            #for value in tempdatatrain.values():
                                #flattened_list.extend(value)
                            flattened_list = [item for sublist in tempdatatrain.values() for item in sublist]
                            tempdatatrain = {}
                            #print(flattened_list)
                            self.targets.append(target)
                            self.features.append(flattened_list)

                        # need to check if all device data is captured, then flatten and append
                            if len(self.epochCounts) > 1: # check if there is more then one test condition
                                minNumKeyEpochs = min([self.epochCounts[key][1] for key in self.epochCounts]) # check minimum viable number of training eochs have been obtained
                                #print("minNumKeyEpochs"+str(minNumKeyEpochs))
                                if minNumKeyEpochs < self.minRequiredEpochs:
                                    pass
                                else: 
                                    start = time.time()
                                    self.classifier.TrainModel(self.features, self.targets)
                                    if (self.printDebug):
                                        end = time.time()
                                        print(f"PyBCI: Info - classifier training time {end - start}")
                            if self.classifierGuessMarkerEvent.is_set():
                                self.classifierGuessMarkerQueue.put(None)
                    else:
                        self.targets.append(target)
                        self.features.append(featuresSingle)
                        if len(self.epochCounts) > 1: # check if there is more then one test condition
                            minNumKeyEpochs = min([self.epochCounts[key][1] for key in self.epochCounts]) # check minimum viable number of training eochs have been obtained
                            #print("minNumKeyEpochs"+str(minNumKeyEpochs))
                            if minNumKeyEpochs < self.minRequiredEpochs:
                                pass
                            else: 
                                start = time.time()
                                self.classifier.TrainModel(self.features, self.targets)
                                if (self.printDebug):
                                    end = time.time()
                                    print(f"PyBCI: Info - classifier training time {end - start}")
                        if self.classifierGuessMarkerEvent.is_set():
                            self.classifierGuessMarkerQueue.put(None)
                except queue.Empty:
                    pass
            else: # We're testing!
                try:
                    featuresSingle, devCount = self.featureQueueTest.get_nowait() #[dataFIFOs, self.currentMarker, self.sr, self.dataType]
                    if self.numStreamDevices > 1:
                        tempdatatest[devCount] = featuresSingle
                        if len(tempdatatest) == self.numStreamDevices:
                            flattened_list = []
                            for value in tempdatatest.values():
                                flattened_list.extend(value)
                            tempdatatest = {}
                            #self.features.append(flattened_list)
                            start = time.time()
                            self.guess = self.classifier.TestModel(flattened_list)
                            if (self.printDebug):
                                end = time.time()
                                print(f"PyBCI: Info - classifier testing time {end - start}")
                    else:
                        start = time.time()
                        self.guess = self.classifier.TestModel(featuresSingle)
                        if (self.printDebug):
                            end = time.time()
                            print(f"PyBCI: Info - classifier testing time {end - start}")
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