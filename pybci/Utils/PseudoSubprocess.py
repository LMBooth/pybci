from pybci.Utils.PseudoDevice import PseudoDeviceController
import os

class PseudoSubprocess:
    def __init__(self):
        self.pd = PseudoDeviceController(execution_mode="thread")

    def begin_subprocess_pseudo(self):
        self.pd.BeginStreaming()

    def stop_subprocess_pseudo(self):
        self.pd.StopStreaming()

fifo_path = "/tmp/my_fifo"

if __name__ == '__main__':
    try:
        os.mkfifo(fifo_path)
    except FileExistsError:
        pass

    ps = PseudoSubprocess()

    while True:
        with open(fifo_path, "r") as fifo:
            command = fifo.read().strip()
            if command == "begin":
                ps.begin_subprocess_pseudo()
            elif command == "stop":
                ps.stop_subprocess_pseudo()
            elif command == "terminate":
                break