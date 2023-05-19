Theory of Operation
############

1. Requirements Prior Initialising with ``bci = PyBCI()``
=========================================================
EEG", "EMG", "Gaze") {hopefully configurable in the future t pass custom fature decoding class}
The bci must have ==1 LSL markerstream selected (if more then one LSL marker stream on system set the desired ML training marker stream with ``PyBCI(markerStream="yourMarkerStream"))``. Warning: If None set picks first available in list.

2. Thread Creation.
=========================================================
Once configuration settings are set various threads are created.

2.1 Marker Thread
**********************************************
The marker stream has its own thread which recieves markers from the target LSL marker stream and when in train mode pushes this marker to the available datastreams.

2.2 Data Threads
**********************************************
Each data stream has its two threads created, one data and one feautre extractor, the thread is responsible for pipelining received data on FIFO's and potentially slicing and overlapping so many seconds before and after the marker appropriately based on the classes `GlobalEpochSettings <https://github.com/LMBooth/pybci/blob/main/pybci/Configuration/EpochSettings.py>`_  and `IndividualEpochSettings <https://github.com/LMBooth/pybci/blob/main/pybci/Configuration/EpochSettings.py>`_, set with ``globalEpochSettings`` and ``customEpochSettings`` when initialising ``PyBCI()``.

2.3 Feature Extractor Threads
**********************************************
The feature extractor threads receive data from their corresponding data stream thread and prepares epoch data for reunification in the classification thread with other devices in the same epoch.

The feature extraction techniques used can vary drastically between devices, to resolve this custom classes can be created to deal with specific stream types and passed to ``streamCustomFeatureExtract`` when initialising ``PyBCI()``, discussed more in :ref:`custom-extractor`.

The default feature extraction used is ``GeneralFeatureChoices`` found in `FeatureSettings.py <https://github.com/LMBooth/pybci/blob/main/pybci/Configuration/FeatureSettings.py>`_, see :ref:`generic-extractor` for more details.

2.4 Classifier Thread
**********************************************
The Classifier thread is responsible for receiving data from the various feature extraction threads, syncrhonising based on the number of target data streams, then passes uses these features for testing and training mahine learning tensorflow and scikit-learn models and classifiers. 

To set you own clf and model see the examples found 

3. Testing and Training the Model
=========================================================

3.2 Training
**********************************************
3.2.1 Retrieiving current estimate
-----------------------------------------
Before the classifier can be run a minimum number of marker strings must be received for each type of target marker, set with the minimumEpochsRequired variable (default: 10).

An sklearn classifier of the users choosing can be passed with the clf variable, or a tensorflow model with pased to model.

The classifier performance or updated model/clf types can be queried by calling ``CurrentClassifierInfo()``, example:

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


3.2 Testing
**********************************************
3.2.1 Retrieiving current estimate
-----------------------------------------------


3.2.2 Resetting or Adding to Train mode Feature Data
-----------------------------------------------
(Functionality is yet to be coded or added, pending...)