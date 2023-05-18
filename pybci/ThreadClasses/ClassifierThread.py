from ..Utils.Classifier import Classifier 
import queue,threading

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