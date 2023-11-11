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
import multiprocessing
import numpy as np


def maker_timing(markerName,markerType, pseudoMarkerConfig,markerConfigStrings, markerQueue, stop_signal, log_queue):
    markerInfo = pylsl.StreamInfo(markerName, markerType, 1, 0, 'string', 'Dev')
    markerOutlet = pylsl.StreamOutlet(markerInfo)
    marker_iterations = 0
    baseline_iterations = 0
    while not stop_signal.is_set():
        if marker_iterations < pseudoMarkerConfig.number_marker_iterations:
            for marker in markerConfigStrings:
                markerOutlet.push_sample([marker])  
                markerQueue.put(marker)  # Put the marker into the queue
                log_queue.put(" PseudoDevice - sending marker " + marker)
                for _ in range(int(pseudoMarkerConfig.seconds_between_baseline_marker * 10)):  # Assuming 0.1s intervals
                    time.sleep(0.1)
                    if stop_signal.is_set():
                        return 
        if baseline_iterations < pseudoMarkerConfig.num_baseline_markers:
            markerOutlet.push_sample([pseudoMarkerConfig.baselineMarkerString])
            log_queue.put(" PseudoDevice - sending " + pseudoMarkerConfig.baselineMarkerString)
            baseline_iterations += 1
            marker_iterations += 1
            for _ in range(int(pseudoMarkerConfig.seconds_between_baseline_marker * 10)):  # Assuming 0.1s intervals
                time.sleep(0.1)
                if stop_signal.is_set():
                    return 
        if baseline_iterations < pseudoMarkerConfig.num_baseline_markers and marker_iterations < pseudoMarkerConfig.number_marker_iterations:
            if pseudoMarkerConfig.repeat:
                marker_iterations = 0
                baseline_iterations = 0
            else:
                break

def generate_signal(dataStreamName,dataStreamType, channelCount, 
                    sampleRate, stop_signal,markerQueue, 
                    markerConfigStrings, pseudoMarkerDataConfigs, baselineConfig):
        samples_generated = 0
        currentMarker = markerConfigStrings[0]
        last_update_time = time.time()
        info = pylsl.StreamInfo(dataStreamName, dataStreamType, channelCount, sampleRate, 'float32', 'Dev')
        chns = info.desc().append_child("channels")
        for label in range(channelCount):
            ch = chns.append_child("channel")
            ch.append_child_value("label", str(label+1))
            ch.append_child_value("type", dataStreamType)
        outlet = pylsl.StreamOutlet(info)
        while not stop_signal.is_set():
            start_time = time.time()
            current_time = time.time()
            delta_time = current_time - last_update_time
            if currentMarker is None and not markerQueue.empty():
                currentMarker = markerQueue.get()
                samples_generated = 0
            if currentMarker is not None:
                for i, command in enumerate(markerConfigStrings):
                    if currentMarker == command:
                        if samples_generated == 0:
                            precomputed_signal = precompute_marker_signal(pseudoMarkerDataConfigs[i], sampleRate, channelCount)
                        
                        remaining_samples = len(precomputed_signal) - samples_generated
                        chunk_size = min(int(sampleRate * delta_time), remaining_samples)
                        num = precomputed_signal[samples_generated:samples_generated+chunk_size]
                        samples_generated += chunk_size
                        if samples_generated >= len(precomputed_signal):
                            currentMarker = None
                            samples_generated = 0
            else: # For baseline
                if samples_generated == 0:
                    precomputed_signal = precompute_marker_signal(baselineConfig, sampleRate, channelCount)
                remaining_samples = len(precomputed_signal) - samples_generated
                chunk_size = min(int(sampleRate * delta_time), remaining_samples)
                num = precomputed_signal[samples_generated:samples_generated+chunk_size]
                samples_generated += chunk_size
                if samples_generated >= len(precomputed_signal):
                    samples_generated = 0  # Reset so that the baseline signal can loop
            outlet.push_chunk(num.tolist())
            last_update_time = current_time
            sleep_duration = max(0, (1.0 / 10) - (start_time - time.time()))
            time.sleep(sleep_duration)

# Pre-compute the full signal for a given marker index
def precompute_marker_signal(config, sampleRate, channelCount):
    total_samples_required = int(sampleRate * config.duration)
    times = np.linspace(0, config.duration, total_samples_required)
    full_signal = config.amplitude * np.sin(2 * np.pi * config.frequency * times)
    full_signal += np.random.normal(0, config.noise_level, total_samples_required)
    return np.tile(full_signal, (channelCount, 1)).T


class PseudoDeviceController:
    log_queue = None

    def __init__(self, is_multiprocessing=True, markerConfigStrings = ["Marker1", "Marker2", "Marker3"], 
                pseudoMarkerDataConfigs = None, createMarkers = True,
                pseudoMarkerConfig = PseudoMarkerConfig,
                dataStreamName = "PyBCIPseudoDataStream" , dataStreamType="EMG", 
                sampleRate= 250, channelCount= 8, logger = Logger(Logger.INFO),log_queue=None):
        self.is_multiprocessing = is_multiprocessing 
        self.logger = logger
        self.log_queue = multiprocessing.Queue() if self.is_multiprocessing else queue.Queue()
        # worker_process args
        self.dataStreamName = dataStreamName
        self.dataStreamType = dataStreamType
        self.sampleRate = sampleRate
        self.channelCount = channelCount
        self.createMarkers = createMarkers
        # marker process args
        self.markerName = pseudoMarkerConfig.markerName
        self.markerType = pseudoMarkerConfig.markerType
        self.markerConfigStrings = markerConfigStrings
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
        self.baselineConfig = pseudoMarkerConfig.baselineConfig
        if self.is_multiprocessing:
            self.stop_signal = multiprocessing.Event()
        else:
            self.stop_signal = threading.Event() 
        self.stop_signal.clear()
        self.log_thread = threading.Thread(target=self.log_message)
        self.log_thread.start()
        #self.log_message()
        self.log_queue.put("Initialised PseudoDevice...")
        #self.log_message("Initialised PseudoDevice...")
    
    def StopStreaming(self):
        self.stop_signal.set() # Set the Event to signal termination
        time.sleep(2)
    
    def BeginStreaming(self):
        if self.is_multiprocessing:
            self.markerQueue = multiprocessing.Queue() 
            self.worker_process = multiprocessing.Process(target=generate_signal, args=(self.dataStreamName, self.dataStreamType, self.channelCount, 
                                                                        self.sampleRate,  self.stop_signal, self.markerQueue, #self.log_queue,
                                                                        self.markerConfigStrings,self.pseudoMarkerDataConfigs, self.baselineConfig))
            self.worker_process.start()
            if self.createMarkers is True:
                self.marker_process = multiprocessing.Process(target=maker_timing, args=(self.markerName, self.markerType, self.pseudoMarkerConfig, self.markerConfigStrings, 
                                                                        self.markerQueue, self.stop_signal, self.log_queue))
                self.marker_process.start()
        else:
            if self.createMarkers is True:
                self.markerQueue = queue.Queue()
            self.stop_signal.clear()
            self.thread = threading.Thread(
                target=generate_signal,
                args=(self.dataStreamName, self.dataStreamType, self.channelCount, self.sampleRate,
                      self.stop_signal, self.markerQueue, self.markerConfigStrings,
                      self.pseudoMarkerDataConfigs, self.baselineConfig)
            )
            self.thread.start()
            if self.createMarkers is True:
                self.marker_thread = threading.Thread(
                    target=maker_timing,
                    args=(self.markerName, self.markerType, self.pseudoMarkerConfig, self.markerConfigStrings,
                        self.markerQueue, self.stop_signal, self.log_queue)
                )
                self.marker_thread.start()
        self.log_queue.put(" PseudoDevice - Begin streaming.")

    def log_message(self, level='INFO', message = ""):
        while not self.stop_signal.is_set():
            try:
                message = self.log_queue.get_nowait()  # Non-blocking retrieval of messages
                if message is None:  # A sentinel value to indicate the end of logging
                    break
                self.logger.log(level, message)
            except queue.Empty:  # If the queue is empty
                time.sleep(0.1)  # Sleep for a short duration before checking the queue again
