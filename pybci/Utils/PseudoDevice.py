################ PsuedoDevice.py ########################
# used for creating fake LSL device data and markers    #
# Please note! sample rate is not exact, expect         #
# some drop over time! linked to overall cpu strain     #
# Written by Liam Booth 19/08/2023                      #
#########################################################
from ..Utils.Logger import Logger
from ..Configuration.PseudoDeviceSettings import PseudoDataConfig, PseudoMarkerConfig
import multiprocessing, time, threading, pylsl, queue
import numpy as np
from collections import deque

class PseudoDeviceController:
    log_queue = None
    def __init__(self, execution_mode='process', *args, **kwargs):
        self.execution_mode = execution_mode
        self.args = args
        self.kwargs = kwargs

        # Create a command queue for the worker
        if self.execution_mode == 'process':
            self.command_queue = multiprocessing.Queue()
            self.stop_signal = multiprocessing.Event()
            self.worker = multiprocessing.Process(target=self._run_device)
            self.log_queue = multiprocessing.Queue()
            # Note: Don't initialize self.device here for 'process' mode!
        elif self.execution_mode == 'thread':
            self.command_queue = None  # Not needed for threads, but kept for consistency
            self.stop_signal = False
            self.device = PseudoDevice(*self.args, **self.kwargs, stop_signal=self.stop_signal, is_multiprocessing=False)  # Initialize for 'thread' mode
            self.worker = threading.Thread(target=self._run_device)
        else:
            raise ValueError(f"Unsupported execution mode: {execution_mode}")

        self.worker.start()
        # Initialize the logger
        
        self.logger = Logger(log_queue=self.log_queue)
        self.log_reader_process = None
        if self.execution_mode == 'process':
            self.log_reader_process = multiprocessing.Process(target=self.logger.start_queue_reader)
            self.log_reader_process.start()


    def _run_device(self):
        if self.execution_mode == 'process':
            device = PseudoDevice(*self.args, **self.kwargs, stop_signal=self.stop_signal, log_queue=self.log_queue, is_multiprocessing=True)  # Initialize locally for 'process' mode

            while not self._should_stop():
                if not self.command_queue.empty():
                    command = self.command_queue.get()
                    if command == "BeginStreaming":
                        device.BeginStreaming()
                # Sleep for a brief moment before checking again
                time.sleep(0.01)

        elif self.execution_mode == 'thread':
            while not self._should_stop():
                self.device.update()  # or any other method you want to run continuously
                time.sleep(0.01)

    def _should_stop(self):
        if self.execution_mode == 'process':
            return self.stop_signal.is_set()
        else:  # thread
            return self.stop_signal

    def BeginStreaming(self):
        if self.execution_mode == 'process':
            self.command_queue.put("BeginStreaming")
        else:  # thread
            self.device.BeginStreaming()

    def StopStreaming(self):
        if self.execution_mode == 'process':
            self.stop_signal.set()
        else:  # thread
            self.stop_signal = True
        self.worker.join()  # Wait for the worker to finish

    def close():
        # add close logic
        print("close it")

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
        if pseudoMarkerDataConfigs == None:
            pseudoMarkerDataConfigs = [PseudoDataConfig(), PseudoDataConfig(), PseudoDataConfig()]
            pseudoMarkerDataConfigs[0].amplitude =  5
            pseudoMarkerDataConfigs[1].amplitude =  6
            pseudoMarkerDataConfigs[2].amplitude =  7
            pseudoMarkerDataConfigs[0].frequency =  6
            pseudoMarkerDataConfigs[1].frequency =  9
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
        self.phase_offset = 0

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


    def GeneratePseudoEMG(self,samplingRate, duration, noise_level, amplitude, frequency):
        """
            Generate a pseudo EMG signal for a given gesture.
            Arguments:
            - sampling_rate: Number of samples per second
            - duration: Duration of the signal in seconds
            - noise_level: The amplitude of Gaussian noise to be added (default: 0.1)
            - amplitude: The amplitude of the EMG signal (default: 1.0)
            - frequency: The frequency of the EMG signal in Hz (default: 10.0)
            Returns:
            - emg_signal: The generated pseudo EMG signal as a 2D numpy array with shape (channels, samples)
        """
        num_samples = int(samplingRate * duration)
        # Initialize the EMG signal array
        emg_signal = np.zeros((num_samples, self.channelCount))
        times = np.linspace(0, duration, num_samples)
        # Generate the pseudo EMG signal based on the marker settings
        emg_channel = amplitude * np.sin(2 * np.pi * frequency * times + self.phase_offset)# * times)  # Sinusoidal EMG signal
        # Add Gaussian noise to the EMG signal
        noise = np.random.normal(0, noise_level, num_samples)
        emg_channel += noise
        # Store the generated channel in the EMG signal array
        self.phase_offset = (self.phase_offset + 2 * np.pi * frequency * (num_samples/samplingRate)) % (2*np.pi)
        emg_channel = emg_channel[::-1]
        for channel in range(self.channelCount):
            emg_signal[:, channel] = emg_channel
        return emg_signal


    def update(self):
        with self.lock:  # Acquire the lock
            if not (self.stop_signal.is_set() if self.is_multiprocessing else self.stop_signal):
                current_time = time.time()
                delta_time = current_time - self.last_update_time
                if not self.markerQueue.empty():
                    self.current_marker = self.markerQueue.get()
                if self.current_marker != None:
                    if not self.markerQueue.empty():
                        self.current_marker = self.markerQueue.get()
                    for i, command in enumerate(self.markerConfigStrings):
                        if self.current_marker == command:
                            total_samples_required = int(self.sampleRate * self.pseudoMarkerDataConfigs[i].duration)
                            num = self.GeneratePseudoEMG(self.sampleRate,delta_time, self.pseudoMarkerDataConfigs[i].noise_level, 
                                                            self.pseudoMarkerDataConfigs[i].amplitude, self.pseudoMarkerDataConfigs[i].frequency)
                            # If this is the start of marker data generation, set the end time for this marker
                            self.samples_generated += len(num)
                            if self.samples_generated >= total_samples_required:
                                self.current_marker = None
                                self.samples_generated = 0  
                else:# send baseline
                    num = self.GeneratePseudoEMG(self.sampleRate,delta_time, self.baselineConfig.noise_level, 
                                                    self.baselineConfig.amplitude, self.baselineConfig.frequency)
                self.outlet.push_chunk(num.tolist())
                self.last_update_time = current_time

    def StopStreaming(self):
        if self.is_multiprocessing:
            self.stop_signal.set()
        else:  # For threading
            self.stop_signal = True
        self.thread.join()  # Wait for the thread to finish

        if self.pseudoMarkerConfig.autoplay:
            self.marker_thread.join()  # Wait for the marker thread to finish
        self.log_message(Logger.INFO, " PseudoDevice - Stopped streaming.")
    

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