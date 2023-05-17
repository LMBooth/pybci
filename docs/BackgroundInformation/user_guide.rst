User Guide
############

.. _installation:

Installation
===================

To use PyBCI, first install it using pip:

.. code-block:: console

   pip install --index-url https://test.pypi.org/simple/ pybci


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
>>> while(True):
>>>     currentMarkers = bci.ReceivedMarkerCount()
>>>     time.sleep(1) # wait for marker updates
>>>     print(currentMarkers)
>>>     if len(currentMarkers) > 1:  # check there is more then one marker type received
>>>         if min([currentMarkers[key][1] for key in currentMarkers]) > 10:
>>>             bci.TestMode()
>>>             break
>>> try:
>>>     while True:
>>>         time.sleep(1)
>>> except KeyboardInterrupt: # allow user to break while loop
>>>     pass


Theory of Operation
===================

1. Requirements Prior Initialising with python bci = PyBCI() 
------------------------------------------------------------
The bci must have >=1 LSL datastream with an accepted dataType ("EEG", "EMG", "Gaze") {hopefully configurable in the future t pass custom fature decoding class}
The bci must have ==1 LSL markerstream selected (if more then one LSL marker stream on system set the desired ML training marker stream with PyBCI(markerStream="yourMarkerStream")). Warning: If None set picks first available in list.

2. Thread Creation.
----------------------------------------------------------------------
Once configuration settings are set various threads are created.

2.1 Marker Thread
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The marker stream has its own thread which recieves markers from the target LSL marker stream and when in train mode pushes this marker to the available datastreams.

2.2 Data Threads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each data stream has its own thread created, the thread is responsible for pipelining received data on FIFO's and slicing appropriately based on globalEpochSettings and customEpochSettings.

2.3 Feature Extractor Thread
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The feature extractor thread receives data from the various data stream threads and prepares epoch data for the classification thread based on settings made in each markers respective feature FeatureSettings`FeatureSettings<https://github.com/LMBooth/pybci/blob/main/pybci/Configuration/FeatureSettings.py>`_.

2.4 Classifier Thread
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Machine learning thread is responsible for 

3. Train Mode
----------
3.1 FeaturesExtractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3.2 Classifier
~~~~~~~~~~~~~~


4.Test Mode
-----------
4.1 Estimated Marker and decoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

4.2 Resetting or Adding to Train mode Feature Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~