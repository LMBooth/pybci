################ PsuedoDevice.py ########################
# used for creating fake LSL device data and markers    #
# Please note! sample rate is not exact, expect         #
# some drop over time! linked to overall cpu strain     #
# Written by Liam Booth 19/08/2023                      #
#########################################################
from ..Utils.Logger import Logger
from ..Configuration.PseudoDeviceSettings import PseudoDataConfig, PseudoMarkerConfig
import  time
import threading
import pylsl
import queue
from multiprocessing import Process, Queue, Event
import multiprocessing
import numpy as np

class PseudoDeviceController:
    log_queue = None

    def __init__(self, execution_mode='process', *args, **kwargs):
        self.execution_mode = execution_mode
        self.args = args
        self.kwargs = kwargs
        self.stop_signal = Event()

        if self.execution_mode == 'process':
            self.command_queue = Queue()
            self.worker = Process(target=self._run_device)
        elif self.execution_mode == 'thread':
            self.worker = threading.Thread(target=self._run_device)
        else:
            raise ValueError(f"Unsupported execution mode: {execution_mode}")

        self.worker.start()

    def _run_device(self):
        device = PseudoDevice(*self.args, **self.kwargs, stop_signal=self.stop_signal)
        while not self.stop_signal.is_set():
            if self.execution_mode == 'process':
                try:
                    command = self.command_queue.get_nowait()
                    if command == "BeginStreaming":
                        device.BeginStreaming()
                except queue.Empty:
                    pass
            elif self.execution_mode == 'thread':
                device.update()

            time.sleep(0.01)

    def BeginStreaming(self):
        if self.execution_mode == 'process':
            self.command_queue.put("BeginStreaming")
        else:
            self.worker.BeginStreaming()

    def StopStreaming(self):
        self.stop_signal.set()
        self.worker.join(timeout=5)
        if self.worker.is_alive():
            self.worker.terminate()

def precise_sleep(duration):
    end_time = time.time() + duration
    while time.time() < end_time:
        pass

class PseudoDevice:
    samples_generated = 0
    chunkCount = 0
    #markerOccurred = False
    current_marker = None
    def __init__(self,stop_signal, is_multiprocessing=True, markerConfigStrings = ["Marker1", "Marker2", "Marker3"], 
                 pseudoMarkerDataConfigs = None, 
                 pseudoMarkerConfig = PseudoMarkerConfig,
                 dataStreamName = "PyBCIPseudoDataStream" , dataStreamType="EMG", 
                 sampleRate= 250, channelCount= 8, logger = Logger(Logger.INFO),log_queue=None):
        #self.currentMarker_lock = threading.Lock()
        self.markerQueue = queue.Queue()
        self.is_multiprocessing = is_multiprocessing 
        self.stop_signal = stop_signal
        self.logger = logger
        self.log_queue = log_queue
        self.lock = threading.Lock()  # Lock for thread safety
        self.markerConfigStrings = markerConfigStrings
        self.baselineConfig = pseudoMarkerConfig.baselineConfig
        self.baselineMarkerString = pseudoMarkerConfig.baselineMarkerString
        self.currentMarker = markerConfigStrings[0]
        if pseudoMarkerDataConfigs is None:
            pseudoMarkerDataConfigs = [PseudoDataConfig(), PseudoDataConfig(), PseudoDataConfig()]
            pseudoMarkerDataConfigs[0].amplitude =  5
            pseudoMarkerDataConfigs[1].amplitude =  6
            pseudoMarkerDataConfigs[2].amplitude =  7
            pseudoMarkerDataConfigs[0].frequency =  8
            pseudoMarkerDataConfigs[1].frequency =  10
            pseudoMarkerDataConfigs[2].frequency =  12
        self.pseudoMarkerDataConfigs = pseudoMarkerDataConfigs
        self.pseudoMarkerConfig = pseudoMarkerConfig
        self.sampleRate = sampleRate
        self.channelCount = channelCount
        markerInfo = pylsl.StreamInfo(pseudoMarkerConfig.markerName, pseudoMarkerConfig.markerType, 1, 0, 'string', 'Dev')
        self.markerOutlet = pylsl.StreamOutlet(markerInfo)
        info = pylsl.StreamInfo(dataStreamName, dataStreamType, self.channelCount, self.sampleRate, 'float32', 'Dev')
        chns = info.desc().append_child("channels")
        for label in range(self.channelCount):
            ch = chns.append_child("channel")
            ch.append_child_value("label", str(label+1))
            ch.append_child_value("type", dataStreamType)
        self.outlet = pylsl.StreamOutlet(info)
        self.last_update_time = time.time()
        self.phase_offset = 0.0

    # Pre-compute the full signal for a given marker index
    def precompute_marker_signal(self, config):
        total_samples_required = int(self.sampleRate * config.duration)
        times = np.linspace(0, config.duration, total_samples_required)
        full_signal = config.amplitude * np.sin(2 * np.pi * config.frequency * times)
        full_signal += np.random.normal(0, config.noise_level, total_samples_required)
        return np.tile(full_signal, (self.channelCount, 1)).T

    def log_message(self, level='INFO', message = ""):
        if self.log_queue is not None and isinstance(self.log_queue, type(multiprocessing.Queue)):
            self.log_queue.put(f'PyBCI: [{level}] - {message}')
        else:
            self.logger.log(level, message)
    
    def _should_stop(self):
        if isinstance(self.stop_signal, multiprocessing.synchronize.Event):
            return self.stop_signal.is_set()
        else:  # boolean flag for threads
            return self.stop_signal

    def update(self):
        with self.lock:
            if not self._should_stop():
                current_time = time.time()
                delta_time = current_time - self.last_update_time

                if self.current_marker is None and not self.markerQueue.empty():
                    self.current_marker = self.markerQueue.get()
                    self.samples_generated = 0

                if self.current_marker is not None:
                    for i, command in enumerate(self.markerConfigStrings):
                        if self.current_marker == command:
                            if self.samples_generated == 0:
                                self.precomputed_signal = self.precompute_marker_signal(self.pseudoMarkerDataConfigs[i])
                            
                            remaining_samples = len(self.precomputed_signal) - self.samples_generated
                            chunk_size = min(int(self.sampleRate * delta_time), remaining_samples)
                            num = self.precomputed_signal[self.samples_generated:self.samples_generated+chunk_size]
                            self.samples_generated += chunk_size
                            if self.samples_generated >= len(self.precomputed_signal):
                                self.current_marker = None
                                self.samples_generated = 0
                else:
                    # For baseline
                    if self.samples_generated == 0:
                        self.precomputed_signal = self.precompute_marker_signal(self.baselineConfig)
                    remaining_samples = len(self.precomputed_signal) - self.samples_generated
                    chunk_size = min(int(self.sampleRate * delta_time), remaining_samples)
                    num = self.precomputed_signal[self.samples_generated:self.samples_generated+chunk_size]
                    self.samples_generated += chunk_size
                    if self.samples_generated >= len(self.precomputed_signal):
                        self.samples_generated = 0  # Reset so that the baseline signal can loop
                self.outlet.push_chunk(num.tolist())
                self.last_update_time = current_time

    # Make sure this method is available in the PseudoDevice class
    def StopStreaming(self):
        if self.is_multiprocessing:
            self.stop_signal.set()
        else:  # For threading
            self.stop_signal = True

        if hasattr(self, 'thread'):
            self.thread.join()  # Wait for the thread to finish

    def BeginStreaming(self):
        if self.is_multiprocessing:
            # For multiprocessing, we assume the worker process is already running
            self.stop_signal.clear()
        else:
            self.stop_signal = False
        #else:  # For threading
        self.thread = threading.Thread(target=self._generate_signal)
        self.thread.start()
        if self.pseudoMarkerConfig.autoplay:
            self.StartMarkers()
        self.log_message(Logger.INFO, " PseudoDevice - Begin streaming.")

    def _generate_signal(self):
        while not self._should_stop():
            start_time = time.time()
            self.update()
            sleep_duration = max(0, (1.0 / 10) - (start_time - time.time()))
            time.sleep(sleep_duration)
#            precise_sleep(sleep_duration)

    def _should_stop(self):
        if self.is_multiprocessing:
            return self.stop_signal.is_set()
        else:
            return self.stop_signal
        
    def StartMarkers(self):
        self.marker_thread = threading.Thread(target=self._maker_timing)
        self.marker_thread.start()

    def _maker_timing(self):
        marker_iterations = 0
        baseline_iterations = 0
        while not (self.stop_signal.is_set() if self.is_multiprocessing else self.stop_signal):
            if marker_iterations < self.pseudoMarkerConfig.number_marker_iterations:
                for marker in self.markerConfigStrings:
                    self.markerOutlet.push_sample([marker])  
                    self.markerQueue.put(marker)  # Put the marker into the queue
                    self.log_message(Logger.INFO," PseudoDevice - sending marker " + marker)
                    time.sleep(self.pseudoMarkerConfig.seconds_between_markers)
            if baseline_iterations < self.pseudoMarkerConfig.num_baseline_markers:
                self.markerOutlet.push_sample([self.pseudoMarkerConfig.baselineMarkerString])
                self.log_message(Logger.INFO," PseudoDevice - sending " + self.pseudoMarkerConfig.baselineMarkerString)
                baseline_iterations += 1
                marker_iterations += 1
                time.sleep(self.pseudoMarkerConfig.seconds_between_baseline_marker)
            if baseline_iterations < self.pseudoMarkerConfig.num_baseline_markers and marker_iterations < self.pseudoMarkerConfig.number_marker_iterations:
                if self.pseudoMarkerConfig.repeat:
                    marker_iterations = 0
                    baseline_iterations = 0
                else:
                    break

'''
class PseudoMarkerConfig:
    markerName = "PyBCIPsuedoMarkers"
    markerType = "Markers"
    repeat = True
    num_baseline_markers = 10
    number_marker_iterations = 10
    seconds_between_markers = 5
    seconds_between_baseline_marker = 10
'''