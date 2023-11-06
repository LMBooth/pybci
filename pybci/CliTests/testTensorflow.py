import time
import threading
import argparse
from ..pybci import PyBCI
import tensorflow as tf# bring in tf for custom model creation
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


stop_signal = threading.Event()  # Global event to control the main loop

def command_listener():
    while  not stop_signal.is_set():
        command = input("PyBCI: [CLI] - Enter 'stop' to terminate\n")
        if command == 'stop':
            stop_signal.set()
            break



class CLI_testPytorchWrapper:
    def __init__(self, createPseudoDevice, min_epochs_train, min_epochs_test,num_chs, num_feats, num_classes, timeout):
        if createPseudoDevice:
            self.num_chs = 8 # 8 channels are created in the PseudoLSLGenerator
            self.num_feats = 2 # default is mean freq and rms to keep it simple
            self.num_classes = 4 # number of different triggers (can include baseline) sent, defines if we use softmax of binary

        self.createPseudoDevice = createPseudoDevice
        self.timeout = timeout
        self.min_epochs_train = min_epochs_train
        self.min_epochs_test = min_epochs_test
        self.accuracy = 0
        self.currentMarkers = {}
        if self.min_epochs_test <= self.min_epochs_train:
            self.min_epochs_test = self.min_epochs_train+1
            # Define the GRU model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Reshape((num_chs*num_feats, 1), input_shape=(num_chs*num_feats,)))
        model.add(tf.keras.layers.GRU(units=256))#, input_shape=num_chs*num_feats)) # maybe should show this example as 2d with toggleable timesteps disabled
        model.add(tf.keras.layers.Dense(units=512, activation='relu'))
        model.add(tf.keras.layers.Flatten())#   )tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax')) # softmax as more then binary classification (sparse_categorical_crossentropy)
        #model.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # sigmoid as ninary classification (binary_crossentropy)
        model.summary()
        model.compile(loss='sparse_categorical_crossentropy',# using sparse_categorical as we expect multi-class (>2) output, sparse because we encode targetvalues with integers
                optimizer='adam',
                metrics=['accuracy'])
        #current_os = get_os()
        #if current_os == "Windows":
        self.bci = PyBCI(minimumEpochsRequired = 3, createPseudoDevice=True, model = model)
        #else:
        #    pdc = PseudoDeviceController(execution_mode="process")
        #    pdc.BeginStreaming()
        #    time.sleep(10)
        #    self.bci = PyBCI(minimumEpochsRequired = 3, createPseudoDevice=True, pseudoDeviceController=pdc, model = model)

        #self.bci = PyBCI(minimumEpochsRequired = min_epochs_train, createPseudoDevice=createPseudoDevice, model = model)
        main_thread = threading.Thread(target=self.loop)
        main_thread.start()
        if self.timeout:
            print("PyBCI: [CLI] - starting timeout thread")
            self.timeout_thread = threading.Thread(target=self.stop_after_timeout)
            self.timeout_thread.start()
        main_thread.join()
        if timeout is not None:
            self.timeout_thread.join()
        

    def loop(self):
        while not self.bci.connected: # check to see if lsl marker and datastream are available
            self.bci.Connect()
            time.sleep(1)
        self.bci.TrainMode() # now both marker and datastreams available start training on received epochs
        self.accuracy = 0
        test = False
        try:
            while not stop_signal.is_set():  # Add the check here
                if test is False:
                    self.currentMarkers = self.bci.ReceivedMarkerCount() # check to see how many received epochs, if markers sent to close together will be ignored till done processing
                    time.sleep(0.5) # wait for marker updates
                    print("Markers received: " + str(self.currentMarkers) +" Accuracy: " + str(round(self.accuracy,2)))
                    if len(self.currentMarkers) > 1:  # check there is more then one marker type received
                        if min([self.currentMarkers[key][1] for key in self.currentMarkers]) > self.bci.minimumEpochsRequired:
                            classInfo = self.bci.CurrentClassifierInfo() # hangs if called too early
                            self.accuracy = classInfo["accuracy"]
                        if min([self.currentMarkers[key][1] for key in self.currentMarkers]) > self.min_epochs_test:  
                            self.bci.TestMode()
                            test = True
                else:
                    markerGuess = self.bci.CurrentClassifierMarkerGuess() # when in test mode only y_pred returned
                    guess = [key for key, value in self.currentMarkers.items() if value[0] == markerGuess]
                    print("Current marker estimation: " + str(guess))
                    time.sleep(0.2)
            self.bci.StopThreads()
        except KeyboardInterrupt: # allow user to break while loop
            print("\nLoop interrupted by user.")

    def stop_after_timeout(self):
        time.sleep(self.timeout)
        stop_signal.set()
        print("\nTimeout reached. Stopping threads.")

    # Add these methods in CLI_testSimpleWrapper class
    def get_accuracy(self):
        return self.accuracy

    def get_current_markers(self):
        return self.currentMarkers

def main(createPseudoDevice=True, min_epochs_train=4, min_epochs_test=10, num_chs = 8, num_feats = 2, num_classes = 4, timeout=None):
    global num_chs_g, num_feats_g, num_classes_g
    num_chs_g = num_chs
    num_feats_g = num_feats
    num_classes_g = num_classes
    command_thread = threading.Thread(target=command_listener)
    command_thread.daemon = True
    command_thread.start()
    
    my_bci_wrapper = CLI_testPytorchWrapper(createPseudoDevice, min_epochs_train, min_epochs_test,num_chs, num_feats, num_classes,timeout)
    command_thread.join()
    return my_bci_wrapper  # Return this instance

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tensorflow GRU is used for model and pseudodevice generates 8 channels of 3 marker types and baseline. Similar to the testTensorflow.py in the examples folder.")
    parser.add_argument("--createPseudoDevice", default=True, type=bool, help="Set to True or False to enable or disable pseudo device creation. pseudodevice generates 8 channels of 3 marker types and baseline.")
    parser.add_argument("--min_epochs_train", default=4, type=int, help='Minimum epochs to collect before model training commences, must be less than, min_epochs_test. If less than min_epochs_test defaults to min_epochs_test+1.')
    parser.add_argument("--min_epochs_test", default=14, type=int, help='Minimum epochs to collect before model testing commences, if less than min_epochs_test defaults to min_epochs_test+1.')
    parser.add_argument("--num_chs", default=8, type=int, help='Num of channels in data stream to configure tensorflow model, if PseudoDevice==True defaults to 8.')
    parser.add_argument("--num_classes", default=4, type=int, help='Num of classes in marker stream to configure tensorflow model, if PseudoDevice==True defaults to 4.')
    parser.add_argument("--timeout", default=None, type=int, help="Timeout in seconds for the script to automatically stop.")

    args = parser.parse_args()
    main(**vars(args))
