from Utils.LSLScanner import LSLScanner
from ThreadClasses.ThreadControlFunctions import DataReceiverThread, MarkerReceiverThread, FeatureProcessorThread
import queue
import threading

class PyBCI:
    """The PyBCI object stores data from available lsl time series data streams (EEG, pupilometry, EMG, etc.)
     and holds a configurable number of samples based on lsl marker strings.
     If no marker strings are available on the LSL the class will close and return an error.
     Optional Inputs:
        dataStreams = List of strings, allows user to set custom acceptable EEG stream definitions, if None defaults to streamTypes scan
        markerStream = List of strings, allows user to set custom acceptable Marker stream definitions, if None defaults to markerTypes scan
        streamTypes = List of strings, allows user to set custom acceptable EEG type definitions, ignored if dataStreams not None
        markerTypes = List of strings, allows user to set custom acceptable Marker type definitions, ignored if markerStream not None
        printDebug = boolean, if true prints LSLScanner debug information
    """
    printDebug = True   # boolean, used to toggle print statements from LSLScanner class
    globalWindowSettings = [
        True,   #addcheck,  checks whether or not to include the epoch
        False,  #splitcheck, checks whether or not subdivide epochs
        0,      #tmins, time in seconds to capture samples before trigger
        1,      #tmaxs, time in seconds to capture samples after trigger
        0.5,    #window_lengths, if splitcheck true - time in seconds to split epoch
        0.5     #overlaps, percentage value > 0 and < 1, example if epoch has tmin of 0 and tmax of 1 with window 
        # length of 0.5 we have 1 epoch between t 0 and t0.5 another at 0.25 to 0.75, 0.5 to 1
     ] 
    markerThread = []
    dataThreads = []
    streamChsDropDict= {}
    dataStreams = []
    markerStream = None
    connected = False

    def __init__(self, dataStreams = None, markerStream= None, streamTypes = None, markerTypes = None, printDebug = True):
        self.lslScanner = LSLScanner(self, dataStreams, markerStream,streamTypes, markerTypes)
        self.printDebug = printDebug
        self.ConfigureEpochWindowSettings()
        self.Connect()

    def __enter__(self): # with bci
        self.ConfigureEpochWindowSettings()
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

    def TrainMode(self):
        if self.connected:
            if self.printDebug:
                print("PyBCI: Started training...")
            self.trainTestEvent.set()
        else:
            self.Connect()
            #self.trainer.StartTraining()

    def TestMode(self):
        # probably need to add check that enough valid epochs are present
        if self.connected:
            if self.printDebug:
                print("PyBCI: Started testing...")
            self.trainTestEvent.clear()
        else:
            self.Connect()

    def __StartThreads(self):
        self.featureQueue = queue.Queue()
        self.dataQueue = queue.Queue()
        totalDevices = len(self.dataStreams)
        lock = threading.Lock() # used for printing in threads
        self.closeEvent = threading.Event() # used for closing threads
        self.trainTestEvent = threading.Event()
        self.trainTestEvent.set() # if set we're in train mode, if not we're in test mode
        if self.printDebug:
            print("PyBCI: Starting threads initialisation...")
        # setup data thread
        self.dataThreads = []
        for stream in self.dataStreams:
            if stream.info().name() in self.streamChsDropDict.keys():
                dt = DataReceiverThread(self.closeEvent, self.trainTestEvent, self.dataQueue, stream,  self.customWindowSettings, self.globalWindowSettings, self.streamChsDropDict[stream.info().name()])
            else:
                dt = DataReceiverThread(self.closeEvent, self.trainTestEvent, self.dataQueue, stream,  self.customWindowSettings, self.globalWindowSettings)
            dt.start()
            self.dataThreads.append(dt)
        # setup feature processing thread, reduces time series data down to statisitical features
        self.featureThread = FeatureProcessorThread(self.closeEvent,self.trainTestEvent, self.dataQueue, self.featureQueue, totalDevices, lock, self.customWindowSettings)
        self.featureThread.start()
        # marker thread requires data and feature threads to push new markers too
        self.markerThread = MarkerReceiverThread(self.closeEvent,self.trainTestEvent, self.markerStream,self.dataThreads, self.featureThread)
        self.markerThread.start()

    def StopThreads(self):
        self.closeEvent.set()
        self.markerThread.join()
        # wait for all threads to finish processing, probably worth pulling out finalised classifier information stored for later use.
        for dt in self.dataThreads:
            dt.join()
        self.featureThread.join()
        self.connected = False
        if self.printDebug:
            print("PyBCI: Threads stopped.")

    # Could move all configures to a configuration class, might make options into more descriptive classes?
    def ConfigureEpochWindowSettings(self, globalWindowSettings = None, customWindowSettings = {}):
        """allows globalWindowSettings to be modified, customWindowSettings is a dict with value names for marker strings which will appear on avalable markerStreams """
        self.customWindowSettings = customWindowSettings
        if globalWindowSettings != None:
            self.globalWindowSettings = globalWindowSettings
        self.ResetThreadsAfterConfigs()

    def ConfigureFeatures(self,freqbands = None, featureChoices = None ):
        # potentially should move configuration to generic class which can be used for both test and train
        if freqbands != None:
            self.featureThread.freqbands = freqbands
        if featureChoices != None:    
            self.featureThread.featureChoices = featureChoices
        self.ResetThreadsAfterConfigs()

    def ConfigureDataStreamChannels(self,streamChsDropDict = {}):
        # potentially should move configuration to generic class which can be used for both test and train
        self.streamChsDropDict = streamChsDropDict 
        self.ResetThreadsAfterConfigs()

    def ResetThreadsAfterConfigs(self):
        if self.connected:
            if self.printDebug:
                print("PyBCI: Resetting threads after BCI reconfiguration...")
            self.StopThreads()
            self.Connect()