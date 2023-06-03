from pylsl import StreamInlet, resolve_stream  
from ..Utils.Logger import Logger

class LSLScanner:
    streamTypes = ["EEG", "ECG", "EMG", "Gaze"] # list of strings, holds desired LSL stream types
    markerTypes = ["Markers"] # list of strings, holds desired LSL marker types
    dataStreams = []    # list of data StreamInlets, available on LSL as chosen by streamTypes
    markerStream = []  # list of marker StreamInlets, available on LSL as chosen by markerTypes
    markerStreamPredefined = False
    dataStreamPredefined = False

    def __init__(self,parent, dataStreamsNames = None, markerStreamName = None, streamTypes = None, markerTypes = None, logger = Logger(Logger.INFO)):
        """
        Intiialises LSLScanner, accepts custom data and marker stream strings to search for, if valid can be obtained after scans with LSLScanner.dataStreams and LSLScanner.makerStream.
        Parameters:
        streamTypes (List of strings): allows user to set custom acceptable EEG stream definitions, if None defaults to streamTypes scan
        markerTypes (List of strings): allows user to set custom acceptable Marker stream definitions, if None defaults to markerTypes scan
        streamTypes (List of strings): allows user to set custom acceptable EEG type definitions, ignored if streamTypes not None
        markerTypes (List of strings): allows user to set custom acceptable Marker type definitions, ignored if markerTypes not None
        logger (pybci.Logger): Custom Logger class or PyBCI, defaults to logger.info if not set, which prints all pybci messages.
        """
        self.parent = parent
        if streamTypes != None:
            self.streamTypes = streamTypes
        if markerTypes != None:
            self.markerTypes = markerTypes
        self.logger = logger
        if dataStreamsNames != None:
            self.dataStreamPredefined = True
            self.dataStreamsNames = dataStreamsNames
        else:
            self.ScanDataStreams()
        if markerStreamName != None:
            self.markerStreamPredefined = True
            self.markerStreamName = markerStreamName
        else:
            self.ScanMarkerStreams()

    def ScanStreams(self):
        """Scans LSL for both data and marker channels."""
        self.ScanDataStreams()
        self.ScanMarkerStreams()

    def ScanDataStreams(self):
        """Scans available LSL streams and appends inlet to self.dataStreams"""
        streams = resolve_stream()
        dataStreams = []
        self.dataStreams = []
        for stream in streams:
            if stream.type() in self.streamTypes:
                dataStreams.append(StreamInlet(stream))
        if self.dataStreamPredefined:
            for s in dataStreams:
                name = s.info().name()
                if name not in self.dataStreamsNames:
                    self.logger.log(Logger.WARNING," Predefined LSL Data Stream name not present.")
                    self.logger.log(Logger.WARNING, " Available Streams: "+str([s.info().name() for s in dataStreams]))
                else:
                    self.dataStreams.append(s)
        else: # just add all datastreams as none were specified
            self.dataStreams = dataStreams
    
    def ScanMarkerStreams(self):
        """Scans available LSL streams and appends inlet to self.markerStreams"""
        streams = resolve_stream()
        markerStreams = []
        self.markerStream = None
        for stream in streams:
            if stream.type() in self.markerTypes:
                markerStreams.append(StreamInlet(stream))
        if self.markerStreamPredefined:
            if len(markerStreams) > 1:
                self.logger.log(Logger.WARNING," Too many Marker streams available, set single desired markerStream in  bci.lslScanner.markerStream correctly.")
            for s in markerStreams:
                name = s.info().name()
                if name != self.markerStreamName:
                    self.logger.log(Logger.WARNING," Predefined LSL Marker Stream name not present.")
                    self.logger.log(Logger.WARNING, " Available Streams: "+str([s.info().name() for s in markerStreams]))
                else:
                    self.markerStream = s
        else:
            if len(markerStreams) > 0:   
                self.markerStream = markerStreams[0] # if none specified grabs first avaialble marker stream

    def CheckAvailableLSL(self):
        """Checks streaminlets available, 
        Returns
        -------
        bool :
            True if 1 marker stream present and available datastreams are present.
            False if no datastreams are present and/or more or less then one marker stream is present, requires hard selection or markser stream if too many.
        """
        self.ScanStreams()
        if self.markerStream == None:
            self.logger.log(Logger.WARNING," No Marker streams available, make sure your accepted marker data Type have been set in bci.lslScanner.markerTypes correctly.")
        if len(self.dataStreams) == 0:
            self.logger.log(Logger.WARNING," No data streams available, make sure your streamTypes have been set in bci.lslScanner.dataStream correctly.")
        if len(self.dataStreams) > 0 and self.markerStream !=None:
            self.logger.log(Logger.INFO," Success - "+str(len(self.dataStreams))+" data stream(s) found, 1 marker stream found")
        
        if len(self.dataStreams) > 0 and self.markerStream != None:
            self.parent.dataStreams = self.dataStreams
            self.parent.markerStream = self.markerStream
            return True
        else:
            return False