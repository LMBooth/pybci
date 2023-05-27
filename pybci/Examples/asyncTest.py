from pylsl import StreamInlet, resolve_stream  
import threading
import time
class MyInlet(threading.Thread):
    def run(self):
        streams = resolve_stream()
        for stream in streams:
            print(stream.name())
            if stream.name() == "pupil_capture":
                my_inlet = StreamInlet(stream)
        # ... setup inlet for an irregular stream
        while True:
            data, timestamps = my_inlet.pull_chunk(timeout=5.0)
            print(f"Received {len(timestamps)} samples.", time.time())
class waiter(threading.Thread):
    def run(self):
        while True:
            print("waiting...", time.time())
            time.sleep(1)
def run():
    MyInlet().start()
    time.sleep(1.0)
    waiter().start()
run()