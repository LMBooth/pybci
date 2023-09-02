################ PsuedoDevice.py ########################
# used for creating fake LSL device data and markers    #
# Please note! sample rate is not exact,        #
# expect some drop over time!                   #
# Written by Liam Booth 19/08/2023              #
#################################################
from ..Utils.Logger import Logger
import random, time, threading, pylsl
import numpy as np
from collections import deque

def precise_sleep(duration):
    end_time = time.time() + duration
    while time.time() < end_time:
        pass

class PseudoDataConfig:
    duration = 1.0 
    noise_level = 0.1
    amplitude = 0.2
    frequency = 1.0

class PseudoDevice:

    def __init__(self, markerName = "PyBCIPsuedoMarkers", markerType = "Markers", markerConfigStrings = ["Marker1", "Marker2", "Marker3", "Marker4", "Marker5"], 
                 pseudoDataConfigs = None, 
                 baselineMarkerString = "baseline", baselineConfig = PseudoDataConfig(),
                 dataStreamName = "PyBCIPsuedoDataStream" , dataStreamType="EMG", 
                 sampleRate= 250, channelCount= 8, logger = Logger(Logger.INFO)):
        self.markerConfigStrings = markerConfigStrings
        if pseudoDataConfigs == None:
            pseudoDataConfigs = [PseudoDataConfig(), PseudoDataConfig(), PseudoDataConfig(), PseudoDataConfig(),PseudoDataConfig()]
            pseudoDataConfigs[0].amplitude =  pseudoDataConfigs[0].amplitude*2
            pseudoDataConfigs[1].amplitude =  pseudoDataConfigs[1].amplitude*3
            pseudoDataConfigs[2].amplitude =  pseudoDataConfigs[2].amplitude*4
            pseudoDataConfigs[3].amplitude =  pseudoDataConfigs[3].amplitude*5
            pseudoDataConfigs[4].amplitude =  pseudoDataConfigs[4].amplitude*6
            pseudoDataConfigs[0].amplitude =  pseudoDataConfigs[0].frequency*2
            pseudoDataConfigs[1].amplitude =  pseudoDataConfigs[1].frequency*3
            pseudoDataConfigs[2].amplitude =  pseudoDataConfigs[2].frequency*4
            pseudoDataConfigs[3].amplitude =  pseudoDataConfigs[3].frequency*5
            pseudoDataConfigs[4].amplitude =  pseudoDataConfigs[4].frequency*6
        self.pseudoDataConfigs = pseudoDataConfigs
        self.sampleRate = sampleRate
        self.channelCount = channelCount
        markerInfo = pylsl.StreamInfo(markerName, markerType, 1, 0, 'string', 'Dev')
        self.markerOutlet = pylsl.StreamOutlet(markerInfo)
        info = pylsl.StreamInfo(dataStreamName, dataStreamType, self.channelCount, self.sampleRate, 'float32', 'Dev')
        chns = info.desc().append_child("channels")
        for label in range(self.channelCount):
            ch = chns.append_child("channel")
            ch.append_child_value("label", str(label+1))
            ch.append_child_value("type", dataStreamType)
        self.outlet = pylsl.StreamOutlet(info)


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
        num_samples = int(duration * samplingRate)
        # Initialize the EMG signal array
        emg_signal = np.zeros((self.totchs, num_samples))
        # Generate the pseudo EMG signal for each channel
        for channel in range(self.totchs):
            # Calculate the time values for the channel
            times = np.linspace(0, duration, num_samples)
            # Generate the pseudo EMG signal based on the selected gesture
            emg_channel = amplitude * np.sin(2 * np.pi * frequency * times)# * times)  # Sinusoidal EMG signal
            # Add Gaussian noise to the EMG signal
            noise = np.random.normal(0, noise_level, num_samples)
            emg_channel += noise
            # Store the generated channel in the EMG signal array
            emg_signal[channel, :] = emg_channel
        return emg_signal


    def update(self):
        with self.lock:  # Acquire the lock
            if self.markerOccurred:
                for i, command in enumerate(self.commandStrings):
                    if self.currentMarker == command:
                        num_samples = int(self.commandDataConfigs[i].duration/10 * self.sampleRate)
                        [self.y[0].popleft() for d in range(num_samples)] 
                        num = self.GeneratePseudoEMG(self.sampleRate,self.commandDataConfigs[i].duration/10, self.commandDataConfigs[i].noise_level, 
                                                        self.commandDataConfigs[i].amplitude, self.commandDataConfigs[i].frequency)
                self.chunkCount += 1
                if self.chunkCount >= 10:
                    self.markerOccurred = False
                    self.chunkCount = 0 
            else:# send baseline
                num_samples = int(self.baselineConfig.duration/10 * self.sampleRate)
                [self.y[0].popleft() for d in range(num_samples)] 
                num = self.GeneratePseudoEMG(self.sampleRate,self.baselineConfig.duration/10, self.baselineConfig.noise_level, 
                                                self.baselineConfig.amplitude, self.baselineConfig.frequency)
            [self.y[0].extend([num[0][d]]) for d in range(num_samples)] 
            for n in range(num.shape[1]):
                self.outlet.push_sample(num[:,n])

    def StopTest(self):    
        self.stop_signal = True
        self.thread.join() # wait for the thread to finish

    def BeginTest(self):
        self.stop_signal = False
        self.thread = threading.Thread(target=self._generate_signal)
        self.thread.start()

    def _generate_signal(self):
        while not self.stop_signal:
            start_time = time.time()
            self.update()
            sleep_duration = max(0, (1.0 / 10) - (start_time - time.time()))
            precise_sleep(sleep_duration)

    def SendMarker(self):
        with self.lock:  # Acquire the lock
            self.markerOutlet.push_sample([self.currentMarker])
            self.markerOccurred = True
    
    def SendBaseline(self):
        self.markerOutlet.push_sample(["Baseline"])