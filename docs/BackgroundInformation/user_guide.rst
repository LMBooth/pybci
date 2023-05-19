User Guide
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

Theory of Operation
===================

1. Requirements Prior Initialising with ``bci = PyBCI()``
------------------------------------------------------------
The bci must have >=1 LSL datastream with an accepted dataType ("EEG", "EMG", "Gaze") {hopefully configurable in the future t pass custom fature decoding class}
The bci must have ==1 LSL markerstream selected (if more then one LSL marker stream on system set the desired ML training marker stream with ``PyBCI(markerStream="yourMarkerStream"))``. Warning: If None set picks first available in list.

2. Thread Creation.
----------------------------------------------------------------------
Once configuration settings are set various threads are created.

2.1 Marker Thread
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The marker stream has its own thread which recieves markers from the target LSL marker stream and when in train mode pushes this marker to the available datastreams.

2.2 Data Threads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each data stream has its two threads created, one data and one feautre extractor, the thread is responsible for pipelining received data on FIFO's and potentially slicing and overlapping so many seconds before and after the marker appropriately based on the classes `GlobalEpochSettings <https://github.com/LMBooth/pybci/blob/main/pybci/Configuration/EpochSettings.py>`_  and `IndividualEpochSettings <https://github.com/LMBooth/pybci/blob/main/pybci/Configuration/EpochSettings.py>`_, set with ``globalEpochSettings`` and ``customEpochSettings`` when initialising ``PyBCI()``.

2.3 Feature Extractor Threads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The feature extractor threads receive data from their corresponding data stream thread and prepares epoch data for reunification in the classification thread with other devices in the same epoch.

The feature extraction techniques used can vary drastically between devices, to resolve this custom classes can be created to deal with specific stream types and passed to ``streamCustomFeatureExtract`` when initialising ``PyBCI()``, discussed more in :ref:`custom-extractor`.

The default feature extraction used is ``GeneralFeatureChoices`` found in `FeatureSettings.py <https://github.com/LMBooth/pybci/blob/main/pybci/Configuration/FeatureSettings.py>`_, see :ref:`generic-extractor` for more details.

2.4 Classifier Thread
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Machine learning thread is responsible for receiving data from the various feature extraction threads, syncrhonising based on the number of target data streams, then passes thse features for testing and training mahine learning tensorflow and scikit learn models and classifiers. 

3. Train Mode
----------

3.1 FeaturesExtractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


3.2 Classifier
~~~~~~~~~~~~~~
Before the classifier can be run a minimum number of marker strings must be received for each type of target marker, set with the minimumEpochsRequired variable (default: 10).

An sklearn classifier of the users choosing can be passed with the clf variable, or a tensorflow model with pased to model.

The classifier performance or updated modedl/clf types can be queried by calling CurrentClassifierInfo(), example:

.. code-block:: python
   bci = PyBCI()
   classInfo = bci.CurrentClassifierInfo()

Where classInfo is a dict of:
.. code-block:: python
   classinfo = {
      "clf":self.classifier.clf,
      "model":self.classifier.model,
      "accuracy":self.classifier.accuracy
   }


4.Test Mode
-----------
4.1 Estimated Marker and decoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

4.2 Resetting or Adding to Train mode Feature Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(Functionality is yet to be coded or added, pending...)