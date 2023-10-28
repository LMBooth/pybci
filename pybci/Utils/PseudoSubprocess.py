
#from pybci import PyBCI
from pybci.Utils.PseudoDevice import PseudoDeviceController
import sys

class PseudoSubprocess:
    def __init__(self):
        print("initializing PseudoSubprocess")
        self.pd = PseudoDeviceController(execution_mode="thread")
        self.pd.BeginStreaming()

    def begin_subprocess_pseudo(self):
        print("attempting to begin streaming")
        self.pd.BeginStreaming()

    def stop_subprocess_pseudo(self):
        print("attempting to stop streaming")
        self.pd.StopStreaming()

if __name__ == '__main__':

    ps = PseudoSubprocess()

    for line in sys.stdin:
        command = line.strip()
        if command == 'begin':
            ps.begin_subprocess_pseudo()
        elif command == 'stop':
            ps.stop_subprocess_pseudo()
        elif command == 'terminate':
            ps.stop_subprocess_pseudo()
            break