from .Utils.LSLScanner import LSLScanner
from .Utils.Logger import Logger
from .Utils.PseudoDevice import PseudoDeviceController
from .ThreadClasses.FeatureProcessorThread import FeatureProcessorThread
#from .ThreadClasses.DataReceiverThread import DataReceiverThread
#from .ThreadClasses.AsyncDataReceiverThread import AsyncDataReceiverThread
from .ThreadClasses.OptimisedDataReceiverThread import OptimisedDataReceiverThread
from .ThreadClasses.MarkerThread import MarkerThread
from .ThreadClasses.ClassifierThread import ClassifierThread
from .Configuration.EpochSettings import GlobalEpochSettings, IndividualEpochSetting
from .Configuration.FeatureSettings import GeneralFeatureChoices
import queue, threading, copy
import tensorflow as tf
#import torch
import torch.nn as nn

#tf.get_logger().setLevel('ERROR')

class PyBCI:
    globalEpochSettings = GlobalEpochSettings()
    customEpochSettings = {}
    minimumEpochsRequired = 10
    markerThread = []
    dataThreads = []
    streamChsDropDict= {}
    dataStreams = []
    markerStream = None
    connected = False
    epochCounts = {} # holds markers received, their target ids and number received of each
    classifierInformation = []
    clf= None
    model = None 
    torchModel = None
    def __init__(self, dataStreams = None, markerStream= None, streamTypes = None, markerTypes = None, loggingLevel = Logger.INFO,
                 globalEpochSettings = GlobalEpochSettings(), customEpochSettings = {}, streamChsDropDict = {},
                 streamCustomFeatureExtract = {},minimumEpochsRequired = 10, 
                 createPseudoDevice=False, pseudoDeviceArgs=None,
                 clf= None, model = None, torchModel = None):
        """
        The PyBCI object stores data from available lsl time series data streams (EEG, pupilometry, EMG, etc.)
        and holds a configurable number of samples based on lsl marker strings.
        If no marker strings are available on the LSL the class will close and return an error.
        Parameters
        ----------
        dataStreams: List[str] 
            Allows user to set custom acceptable EEG stream definitions, if None defaults to streamTypes scan
        markerStream: List[str] 
            Allows user to set custom acceptable Marker stream definitions, if None defaults to markerTypes scan
        streamTypes: List[str] 
            Allows user to set custom acceptable EEG type definitions, ignored if dataStreams not None
        markerTypes: List[str] 
            Allows user to set custom acceptable Marker type definitions, ignored if markerStream not None
        loggingLevel: string 
            Sets PyBCI print level, ('INFO' prints all statements, 'WARNING' is only warning messages, 'TIMING' gives estimated time for feature extraction, and  classifier training or testing,  'NONE' means no prints from PyBCI)
        globalEpochSettings (GlobalEpochSettings): 
            Sets global timing settings for epochs.
        customEpochSettings: dict 
            Sets individual timing settings for epochs. {markerstring1:IndividualEpochSettings(),markerstring2:IndividualEpochSettings()}
        streamChsDropDict: dict 
            Keys for dict should be respective datastreams with corresponding list of which channels to drop.  {datastreamstring1: list(ints), datastreamstring2: list(ints)}
        streamCustomFeatureExtract: dict
            Allows dict to be passed of datastream with custom feature extractor class for analysing data.  {datastreamstring1: customClass1(), datastreamstring2: customClass1(),}
        minimumEpochsRequired: int 
            Minimm number of required epochs before model fitting begins, must be of each type of received markers and mroe then 1 type of marker to classify.
        createPseudoDevice : bool
            Sets whether a pseudodevice should be generated.
        pseudoDevice : pybci.Utils.PseudoDevice.PseudoDeviceController 
            Allows custom pseudoDevice to be passed with own marker and signal information.
        clf: sklearn.base.ClassifierMixin 
            Allows custom Sklearn model to be passed.
        model: tf.keras.model
            Allows custom tensorflow model to be passed.
        torchmodel:  [torchModel(), torch.nn.Module]
            Currently a list where first item is torchmodel analysis function, second is torch model, check pytorch example - likely to change in future updates.
        """
        self.streamCustomFeatureExtract = streamCustomFeatureExtract
        self.globalEpochSettings = globalEpochSettings
        self.customEpochSettings = customEpochSettings
        self.streamChsDropDict = streamChsDropDict
        self.loggingLevel = loggingLevel
        self.logger = Logger(self.loggingLevel)
        self.lslScanner = LSLScanner(self, dataStreams, markerStream,streamTypes, markerTypes, logger =self.logger)
        self.ConfigureMachineLearning(minimumEpochsRequired,  clf, model, torchModel) # configure first, connect second
        self.Connect()
        if createPseudoDevice:
            if isinstance(pseudoDeviceArgs,dict):
                pseudoDevice = PseudoDeviceController(pseudoDeviceArgs)
                pseudoDevice.BeginStreaming()
            else:
                pseudoDevice = PseudoDeviceController()
                pseudoDevice.BeginStreaming()
        self.pseudoDevice = pseudoDevice

       
    def __enter__(self, dataStreams = None, markerStream= None, streamTypes = None, markerTypes = None, loggingLevel = Logger.INFO,
                 globalEpochSettings = GlobalEpochSettings(), customEpochSettings = {}, streamChsDropDict = {},
                 streamCustomFeatureExtract = {},
                 minimumEpochsRequired = 10, clf= None, model = None, torchModel = None): # with bci
        """
        Please look at PyBCI.__init__ (same parameters, setup and description)
        """
        self.streamCustomFeatureExtract = streamCustomFeatureExtract
        self.globalEpochSettings = globalEpochSettings
        self.customEpochSettings = customEpochSettings
        self.streamChsDropDict = streamChsDropDict
        self.lslScanner = LSLScanner(self, dataStreams, markerStream,streamTypes, markerTypes)
        self.loggingLevel = loggingLevel
        self.ConfigureMachineLearning(minimumEpochsRequired,  clf, model, torchModel) # configure first, connect second
        self.Connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.StopThreads()

    def Connect(self): # Checks valid data and markers streams are present, controls dependant functions by setting self.connected
        if self.lslScanner.CheckAvailableLSL():
            self.__StartThreads()
            self.connected = True
            return True # uses return statements so user can check if connected with bool returned
        else:
            self.connected = False
            return False

    # set test and train boolean for changing  thread operation 
    def TrainMode(self):
        """
         Starts BCI training If PyBCI is connected to valid LSL data and marker streams, if not tries to scan and connect.
        """
        if self.connected:
            self.logger.log(Logger.INFO," Started training...")
            self.trainTestEvent.set()
        else:
            self.Connect()

    def TestMode(self):
        """
         Starts BCI testing If PyBCI is connected to valid LSL data and marker streams, if not tries to scan and connect.
         (Need to check if invalid number of epochs are obtained and this is set)
        """
        if self.connected:
            self.logger.log(Logger.INFO," Started testing...")
            self.trainTestEvent.clear()
        else:
            self.Connect()

    # Get data from threads
    def CurrentClassifierInfo(self):
        """
        Gets dict with current clf, model, torchModel and accuracy. Accuracy will be 0 if not fiting has occurred.
        Returns
        -------
        classInfo: dict
            dict of "clf", "model, "torchModel"" and "accuracy" where accuracy is 0 if no model training/fitting has occurred. If mode not used corresponding value is None.
            If not connected returns {"Not Connected": None}
        """
        if self.connected:
            self.classifierInfoRetrieveEvent.set()
            classInfo = self.classifierInfoQueue.get()
            self.classifierInfoRetrieveEvent.clear()
            return classInfo
        else:
            self.Connect()
            return {"Not Connected": None}

    def CurrentClassifierMarkerGuess(self):
        """
        Gets classifier current marker guess and targets.
        Returns
        -------
        classGuess: int | None
            Returned int correlates to value of key from dict from ReceivedMarkerCount() when in testmode. 
            If in trainmode returns None.
        """
        if self.connected:
            # probably needs check that we're in test mode, maybe debu print if not?
            self.classifierGuessMarkerEvent.set()
            classGuess = self.classifierGuessMarkerQueue.get()
            self.classifierGuessMarkerEvent.clear()
            return classGuess
        else:
            self.Connect()
            return None

    def CurrentFeaturesTargets(self):
        """
        Gets classifier current features and targets.
        Returns
        -------
        featureTargets: dict
            dict of "features" and "targets" where features is 2d list of feature data and targets is a 1d list of epoch targets as ints.
            If not connected returns {"Not Connected": None}
        """
        if self.connected:
            self.queryFeaturesEvent.set()
            featureTargets = self.queryFeaturesQueue.get()
            self.queryFeaturesEvent.clear() # still needs coding
            return featureTargets
        else:
            self.Connect()
            return {"Not Connected": None}

    def ReceivedMarkerCount(self):
        """
        Gets number of received training marker, their strings and their respective values to correlate with CurrentClassifierMarkerGuess().
        Returns
        -------
        markers: dict
            Every key is a string received on the selected LSL marker stream, the value is a list where the first item is the marker id value, 
            use with CurrentClassifierMarkerGuess() the second value is a received count for that marker type. Will be empty if no markers received.
        """
        if self.connected:
            self.markerCountRetrieveEvent.set()
            markers = self.markerCountQueue.get()
            self.markerCountRetrieveEvent.clear()
            return markers
        else:
            self.Connect()

    def __StartThreads(self):
        self.featureQueueTrain = queue.Queue()
        self.featureQueueTest = queue.Queue()
        self.classifierInfoQueue = queue.Queue()
        self.markerCountQueue = queue.Queue()
        self.classifierGuessMarkerQueue = queue.Queue()
        self.classifierGuessMarkerEvent = threading.Event()
        self.closeEvent = threading.Event() # used for closing threads
        self.trainTestEvent = threading.Event()
        self.markerCountRetrieveEvent = threading.Event()
        self.classifierInfoRetrieveEvent = threading.Event()

        self.queryFeaturesQueue = queue.Queue()
        self.queryFeaturesEvent  = threading.Event() # still needs coding

        self.trainTestEvent.set() # if set we're in train mode, if not we're in test mode, always start in train...
        self.logger.log(Logger.INFO," Starting threads initialisation...")
        # setup data thread
        self.dataThreads = []
        self.featureThreads = []
        for stream in self.dataStreams:
            self.dataQueueTrain = queue.Queue()
            self.dataQueueTest = queue.Queue()

            #if stream.info().nominal_srate() == 0:
            #    if stream.info().name() in self.streamChsDropDict.keys():
            #        dt = AsyncDataReceiverThread(self.closeEvent, self.trainTestEvent, self.dataQueueTrain,self.dataQueueTest, stream,  self.customEpochSettings, 
            #                                self.globalEpochSettings, len(self.dataThreads), streamChsDropDict=self.streamChsDropDict[stream.info().name()])
            #    else:
            #        dt = AsyncDataReceiverThread(self.closeEvent, self.trainTestEvent, self.dataQueueTrain,self.dataQueueTest, stream,  self.customEpochSettings, 
            #                                self.globalEpochSettings, len(self.dataThreads))
            #else: # cold be desirable to capture samples only relative to timestammps with async, so maybe make this configurable?
            #if stream.info().nominal_srate() == 0:
            print(len(self.dataThreads))
            print(stream.info().name())
            if stream.info().name() in self.streamChsDropDict.keys(): ## all use optimised now (pull_chunk and timestamp relative)
                #print(self.streamChsDropDict[stream.info().name()])
                dt = OptimisedDataReceiverThread(self.closeEvent, self.trainTestEvent, self.dataQueueTrain,self.dataQueueTest, stream,  self.customEpochSettings, 
                                        self.globalEpochSettings, len(self.dataThreads), streamChsDropDict=self.streamChsDropDict[stream.info().name()])
            else:
                dt = OptimisedDataReceiverThread(self.closeEvent, self.trainTestEvent, self.dataQueueTrain,self.dataQueueTest, stream,  self.customEpochSettings, 
                                        self.globalEpochSettings, len(self.dataThreads))
            #else:
            #    if stream.info().name() in self.streamChsDropDict.keys(): ## all use optimised now (pull_chunk and timestamp relative)
            #        #print(self.streamChsDropDict[stream.info().name()])
            #        dt = OptimisedDataReceiverThread(self.closeEvent, self.trainTestEvent, self.dataQueueTrain,self.dataQueueTest, stream,  self.customEpochSettings, 
            #                                self.globalEpochSettings, len(self.dataThreads), streamChsDropDict=self.streamChsDropDict[stream.info().name()], maxExpectedSampleRate = stream.info().nominal_srate())
            #    else:
            #        dt = OptimisedDataReceiverThread(self.closeEvent, self.trainTestEvent, self.dataQueueTrain,self.dataQueueTest, stream,  self.customEpochSettings, 
            #                                self.globalEpochSettings, len(self.dataThreads),maxExpectedSampleRate = stream.info().nominal_srate())
            dt.start()
            self.dataThreads.append(dt)
            if stream.info().name() in self.streamCustomFeatureExtract.keys():
                self.ft = FeatureProcessorThread(self.closeEvent,self.trainTestEvent, self.dataQueueTrain, self.dataQueueTest,
                                                self.featureQueueTest,self.featureQueueTrain, len(self.dataStreams),
                                                self.markerCountRetrieveEvent, self.markerCountQueue,logger=self.logger,
                                                featureExtractor = self.streamCustomFeatureExtract[stream.info().name()],
                                                globalEpochSettings = self.globalEpochSettings, customEpochSettings = self.customEpochSettings)
            else:
                self.ft = FeatureProcessorThread(self.closeEvent,self.trainTestEvent, self.dataQueueTrain, self.dataQueueTest,
                                                self.featureQueueTest,self.featureQueueTrain, len(self.dataStreams),
                                                self.markerCountRetrieveEvent, self.markerCountQueue,logger=self.logger,
                                                globalEpochSettings = self.globalEpochSettings, customEpochSettings = self.customEpochSettings)
            self.ft.start()
            self.featureThreads.append(dt)
        # marker thread requires data and feature threads to push new markers too
        self.markerThread = MarkerThread(self.closeEvent,self.trainTestEvent, self.markerStream,self.dataThreads, self.featureThreads)
        self.markerThread.start()
        self.classifierThread = ClassifierThread(self.closeEvent,self.trainTestEvent, self.featureQueueTest,self.featureQueueTrain,
                                                 self.classifierInfoQueue, self.classifierInfoRetrieveEvent,
                                                 self.classifierGuessMarkerQueue, self.classifierGuessMarkerEvent, self.queryFeaturesQueue, self.queryFeaturesEvent,
                                                 logger = self.logger, numStreamDevices = len(self.dataThreads), minRequiredEpochs = self.minimumEpochsRequired,
                                                clf = self.clf, model = self.model, torchModel = self.torchModel)
        self.classifierThread.start()
        self.logger.log(Logger.INFO," Threads initialised.")


    def StopThreads(self):
        """
        Stops all PyBCI threads.
        """
        self.closeEvent.set()
        self.markerThread.join()
        # wait for all threads to finish processing, probably worth pulling out finalised classifier information stored for later use.
        for dt in self.dataThreads:
            dt.join()
        for ft in self.featureThreads:
            ft.join()
        self.classifierThread.join()
        self.connected = False
        self.logger.log(Logger.INFO," Threads stopped.")

    def ConfigureMachineLearning(self, minimumEpochsRequired = 10, clf = None, model = None, torchModel = None):
        from sklearn.base import ClassifierMixin
        self.minimumEpochsRequired = minimumEpochsRequired
        
        if isinstance(clf, ClassifierMixin):
            self.clf = clf
        else:
            self.clf = None
            self.logger.log(Logger.INFO," Invalid or no sklearn classifier passed to clf. Checking tensorflow model... ")
            if isinstance(model, tf.keras.Model):
                self.model = model
            else:
                self.model = None
                self.logger.log(Logger.INFO," Invalid or no tensorflow model passed to model.  Checking pytorch torchModel...")
                if callable(torchModel): # isinstance(torchModel, torch.nn.Module):
                    self.torchModel = torchModel
                else:
                    self.torchModel = None
                    self.logger.log(Logger.INFO," Invalid or no PyTorch model passed to model. Defaulting to SVM by SkLearn")
    

    # Could move all configures to a configuration class, might make options into more descriptive classes?
    def ConfigureEpochWindowSettings(self, globalEpochSettings = GlobalEpochSettings(), customEpochSettings = {}):
        """
        Allows globalWindowSettings to be modified, customWindowSettings is a dict with value names for marker strings which will appear on avalable markerStreams. 
        """
        valid = False
        for key in customEpochSettings.keys():
            if isinstance(customEpochSettings[key], IndividualEpochSetting):
                valid = True
            else:
                valid = False
                self.logger.log(Logger.WARNING," Invalid datatype passed for customWindowSettings, create dict of wanted markers \
                        using class bci.IndividualEpochSetting() as value to configure individual epoch window settings.")
                break
        #if isinstance(customWindowSettings[key], GlobalEpochSettings()):
        if valid:   
            self.customEpochSettings = customEpochSettings
            if globalEpochSettings.windowLength > globalEpochSettings.tmax + globalEpochSettings.tmin:
                self.logger.log(Logger.WARNING," windowLength < (tmin+tmax), pass vaid settings to ConfigureEpochWindowSettings")
            else:
                self.globalWindowglobalEpochSettingsSettings = globalEpochSettings
                self.ResetThreadsAfterConfigs()

    def ConfigureDataStreamChannels(self,streamChsDropDict = {}):
        # potentially should move configuration to generic class which can be used for both test and train
        self.streamChsDropDict = streamChsDropDict 
        self.ResetThreadsAfterConfigs()

    def ResetThreadsAfterConfigs(self):
        if self.connected:
            self.logger.log(Logger.INFO,"Resetting threads after BCI reconfiguration...")
            self.StopThreads()
            self.Connect()
