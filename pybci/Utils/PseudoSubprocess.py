from pybci.Utils.PseudoDevice import PseudoDeviceController
import sys

class PseudoSubprocess:
    def __init__(self):
        self.pd = PseudoDeviceController(execution_mode="thread")

    def begin_subprocess_pseudo(self):
        self.pd.BeginStreaming()

    def stop_subprocess_pseudo(self):
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
            break