Introduction
############

.. _installation:
Installation
===================

To use PyBCI, first install it using pip:

.. code-block:: console

   pip install install-pybci
   
(currently can only install-pybci due to pybci name similarities with another package on pypi)

For unstable dev installations and up-to-date git pushes use:

.. code-block:: console

   pip install --index-url https://test.pypi.org/simple/ install-pybci


.. _simpleimplementation:

Simple Implementation:
===================
For example:

>>> import time
>>> from pybci import PyBCI
>>> bci = PyBCI()
>>> while not bci.connected:
>>>     bci.Connect()
>>>     time.sleep(1)
>>> bci.TrainMode()
>>> try:
>>>     while(True):
>>>         currentMarkers = bci.ReceivedMarkerCount() # check to see how many received epochs, if markers sent to close together will be ignored till done processing
>>>         time.sleep(1) # wait for marker updates
>>>         print("Markers received: " + str(currentMarkers) +" Class accuracy: " + str(accuracy), end="\r")
>>>         if len(currentMarkers) > 1:  # check there is more then one marker type received
>>>             if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired:
>>>                 classInfo = bci.CurrentClassifierInfo() # hangs if called too early
>>>                 accuracy = classInfo["accuracy"]
>>>             if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired+1:  
>>>                 bci.TestMode()
>>>                 break
>>>     while True:
>>>         markerGuess = bci.CurrentClassifierMarkerGuess() # when in test mode only y_pred returned
>>>         guess = [key for key, value in currentMarkers.items() if value[0] == markerGuess]
>>>         print("Current marker estimation: " + str(guess), end="\r")
>>>         time.sleep(0.5)
>>> except KeyboardInterrupt: # allow user to break while loop
>>>     pass

What is PyBCI?
===================
PyBCI is a python based brain computer interface software designed to receive a varying number, be it singular or multiple, Lab Streaming Layer enabled physiological sensor data streams. An understanding of time-series data analysis, the lab streaming layer protocol, and machine learning techniques are a must to integrate innovative ideas with this interface. An LSL marker stream is required to train the model, where a received marker epochs the data received on the accepted datastreams based on a configurable time window around certain markers - where custom marker strings can optionally have its epoch timewindow split and overlapped to count for more then one marker, example: a baseline marker may have one marker sent for a 60 second window, where as target actions may only be ~0.5s long, so to conform when testing the model and giving a standardised window length would be desirable to split the 60s window after the received baseline marker in to ~0.5s windows. By overlapping windows we try to account for potential missed signal patterns/aliasing, as a rule of thumb it would be advised when testing a model to have an overlap >= than 50%, see Shannon nyquist criterion.

Once the data has been epoched it is sent for feature extraction, there is a general feature extraction class which can be configured for general time and/or frequency analysis based features, data streams types like "EEG" and "EMG". (DevNOTE: looking to write class for basic pupil labs example to be passed, or solely samples + channels, or other custom classes passed to selected marker streams >:] )

Finally a passable, customisable sklearn or tensorflow classifier can be giving to the bci class, once a defined number of epochs have been obtained for each received epoch/marker type the classifier can begin to fit the model. It's advised to use bci.ReceivedMarkerCount() to get the number of received training epochs received, once the min num epochs received of each type is >= pybci.minimumEpochsRequired (default 10 of each epoch) the mdoel will begin to fit. Once fit classifier info can be queried with CurrentClassifierInfo, when a desired accuracy is met or number of epochs TestMode() can be called. Once in test mode you can query (sould change function to own function and queue for quering testthread) what pybci estimates the current bci epoch is(typically bseline is used for no state).

The `examples folder <https://github.com/LMBooth/pybci/tree/main/pybci/Examples>`__ found on the github has a pseudo `LSL data generator and marker creator <https://github.com/LMBooth/pybci/tree/main/pybci/Examples/PsuedoLSLStreamGenerator>`__ so the examples can run without the need of LSL capable hardware.

