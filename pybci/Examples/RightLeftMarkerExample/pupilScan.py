from pylsl import StreamInlet, resolve_stream  
def ScanDataStreams():
    """Scans available LSL streams and appends inlet to self.dataStreams"""
    streams = resolve_stream()
    dataStreams = []
    for stream in streams:
        s = StreamInlet(stream)
        print(s.info().name())
        print(s.info().type())
        print(s.info().nominal_srate())
    return dataStreams

ScanDataStreams()