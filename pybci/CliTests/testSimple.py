import argparse
from pybci import PyBCI
import time 
import threading

def main(createPseudoDevice=True, min_epochs_train=4, min_epochs_test=10, timeout=None):
    def loop(bci):
        while not bci.connected: # check to see if lsl marker and datastream are available
            bci.Connect()
            time.sleep(1)
        bci.TrainMode() # now both marker and datastreams available start training on received epochs
        accuracy = 0
        test = False
        try:
            while(True):
                if test is False:
                    currentMarkers = bci.ReceivedMarkerCount() # check to see how many received epochs, if markers sent to close together will be ignored till done processing
                    time.sleep(0.5) # wait for marker updates
                    print("Markers received: " + str(currentMarkers) +" Accuracy: " + str(round(accuracy,2)), end="         \r")
                    if len(currentMarkers) > 1:  # check there is more then one marker type received
                        if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired:
                            classInfo = bci.CurrentClassifierInfo() # hangs if called too early
                            accuracy = classInfo["accuracy"]
                        if min([currentMarkers[key][1] for key in currentMarkers]) > min_epochs_test:  
                            bci.TestMode()
                            test = True
                else:
                    markerGuess = bci.CurrentClassifierMarkerGuess() # when in test mode only y_pred returned
                    guess = [key for key, value in currentMarkers.items() if value[0] == markerGuess]
                    print("Current marker estimation: " + str(guess), end="           \r")
                    time.sleep(0.2)
        except KeyboardInterrupt: # allow user to break while loop
            print("\nLoop interrupted by user.")

    def stop_after_timeout(bci):
        time.sleep(timeout)
        print("\nTimeout reached. Stopping threads.")
        bci.StopThreads()
    
    if min_epochs_test <= min_epochs_train:
        min_epochs_test = min_epochs_train+1
    bci = PyBCI(minimumEpochsRequired = min_epochs_train, createPseudoDevice=createPseudoDevice)
    main_thread = threading.Thread(target=loop, args=(bci,))
    main_thread.start()

    if timeout:
        timeout_thread = threading.Thread(target=stop_after_timeout, args=(bci,))
        timeout_thread.start()
        timeout_thread.join()

    main_thread.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Runs simple setup where sklearn support-vector-machine is used for model and pseudodevice generates 8 channels of 3 marker types and a baseline. Similar to the testSimple.py in the examples folder.")
    parser.add_argument("--createPseudoDevice", default=True, type=bool, help="Set to True or False to enable or disable pseudo device creation. pseudodevice generates 8 channels of 3 marker types and baseline.")
    parser.add_argument("--min_epochs_train", default=4, type=int, help='Minimum epochs to collect before model training commences, must be less than, min_epochs_test. If less than min_epochs_test defaults to min_epochs_test+1.')
    parser.add_argument("--min_epochs_test", default=14, type=int, help='Minimum epochs to collect before model testing commences, if less than min_epochs_test defaults to min_epochs_test+1.')
    parser.add_argument("--timeout", default=None, type=int, help="Timeout in seconds for the script to automatically stop.")


    args = parser.parse_args()
    main(**vars(args))
