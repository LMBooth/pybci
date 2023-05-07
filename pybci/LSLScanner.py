from pylsl import StreamInlet, resolve_stream  
class LSLScanner:
    streamTypes = ["EEG", "ECG", "EMG", "pupil_capture"] # list of strings, holds desired LSL stream types
    markerTypes = ["Markers"] # list of strings, holds desired LSL marker types
    dataStreams = []    # list of data StreamInlets, available on LSL as chosen by streamTypes
    markerStreams = []  # list of marker StreamInlets, available on LSL as chosen by markerTypes

    def __init__(self,parent, dataStreams = None, markerStreams= None, streamTypes = None, markerTypes = None, printDebug = True):
        """Intiialises LSLSCanner 
            Optional Inputs:
                streamTypes = List of strings, allows user to set custom acceptable EEG stream definitions, if None defaults to streamTypes scan
                markerTypes = List of strings, allows user to set custom acceptable Marker stream definitions, if None defaults to markerTypes scan
                streamTypes = List of strings, allows user to set custom acceptable EEG type definitions, ignored if streamTypes not None
                markerTypes = List of strings, allows user to set custom acceptable Marker type definitions, ignored if markerTypes not None
                printDebug = boolean, if true prints LSLScanner debug information
        """
        self.parent = parent
        if streamTypes != None:
            self.streamTypes = streamTypes
        if markerTypes != None:
            self.markerTypes = markerTypes
        if printDebug == False:
            self.printDebug = False
        if dataStreams != None:
            self.dataStreams = dataStreams
        else:
            self.ScanDataStreams()
        if markerStreams != None:
            self.markerStreams = markerStreams
        else:
            self.ScanMarkerStreams()
        self.CheckAvailableLSL()

    def ScanStreams(self):
        """Scans LSL for both data and marker channels."""
        self.ScanDataStreams()
        self.ScanMarkerStreams()

    def ScanDataStreams(self):
        """Scans available LSL streams and appends inlet to self.dataStreams"""
        self.streams = resolve_stream()
        dataStreams = []
        for stream in self.streams:
            if stream.type() in self.streamTypes:
                dataStreams.append(StreamInlet(stream))
        self.dataStreams = dataStreams
    
    def ScanMarkerStreams(self):
        """Scans available LSL streams and appends inlet to self.markerStreams"""
        self.streams = resolve_stream()
        markerStreams = []
        for stream in self.streams:
            if stream.type() in self.markerTypes:
                markerStreams.append(StreamInlet(stream))
        self.markerStreams = markerStreams

    def CheckAvailableLSL(self):
        """Checks streaminlets available, prints if printDebug  
        Returns:
            True = 1 marker stream present and available datastreams are present
            False = If no datastreams are present and/or more or less then one marker stream is present, requires hard selection or markser stream if too many.
        """
        self.ScanStreams()
        if (self.parent.printDebug):
            if len(self.markerStreams) == 0:
                print("PyLSL-BCI: No Markers streams available, make sure your streamTypes have been set in bci.lslScanner.dataStreams correctly")
            elif len(self.markerStreams) > 1:
                print("PyLSL-BCI: Too many Marker streams available, set markerStreams in  bci.lslScanner.dataStreams correctly")
            if len(self.dataStreams) == 0:
                print("PyLSL-BCI: No data streams available, make sure your streamTypes have been set in bci.lslScanner.markerStreams correctly")
        if len(self.dataStreams) > 0 and len(self.markerStreams) == 1:
            return True
        else:
            return False