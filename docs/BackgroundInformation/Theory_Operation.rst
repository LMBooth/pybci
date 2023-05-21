Theory of Operation
############

1. Requirements Prior Initialising with `bci = PyBCI()`
=========================================================
The bci must have ==1 LSL marker stream selected (if more then one LSL marker stream on system set the desired ML training marker stream with :class:`markerStream` to  :py:class:`PyBCI()`). Warning: If None set it picks first available in list, if more then one marker stream available to LSL then it is advised to hard select on intialisation.

2. Thread Creation
=========================================================
Once configuration settings are set various threads are created.

2.1 Marker Thread
**********************************************
The marker stream has its own thread which recieves markers from the target LSL marker stream and when in train mode, the marker thread pushed the marker to all available data threads informing when to slice the data, see :ref:`set_custom_epoch_times`. Set the desired ML training marker stream with :class:`markerStream` to  :py:class:`PyBCI()`.

2.2 Data Threads
**********************************************
Each data stream has its two threads created, one data and one feautre extractor, the thread is responsible for pipelining received data on `deque` FIFO's and optionally slicing and overlapping so many seconds before and after the marker appropriately based on the classes `GlobalEpochSettings <https://github.com/LMBooth/pybci/blob/main/pybci/Configuration/EpochSettings.py>`_  and `IndividualEpochSettings <https://github.com/LMBooth/pybci/blob/main/pybci/Configuration/EpochSettings.py>`_, set with :class:`globalEpochSettings` and :class:`customEpochSettings` when initialising :py:class:`PyBCI()`.

Add desired dataStreams by passing a list of accepted data stream names with `dataStreams`.

Upon data thread creation the effective sample rate is queried for each LSL data stream, if the sample rate is 0 an `Asynchronous thread <https://github.com/LMBooth/pybci/blob/main/pybci/ThreadClasses/AsyncDataReceiverThread.py>`_ is created for FIFO handling, though potentially more accurate, it is far more computationally intensive to slice data than the `synchronous data thread <https://github.com/LMBooth/pybci/blob/main/pybci/ThreadClasses/DataReceiverThread.py>`_. If n effective sample rate  greater than 0 is supplised by the LSL datastream a syncrhnous data thread is used for slicing epochs relative to markers in training mode and continously slices in testing mode.

2.3 Feature Extractor Threads
**********************************************
The feature extractor threads receive data from their corresponding data stream thread and prepares epoch data for reunification in the classification thread with other devices in the same epoch.

The feature extraction techniques used can vary drastically between devices, to resolve this custom classes can be created to deal with specific stream types and passed to :class:`streamCustomFeatureExtract` when initialising  :py:class:`PyBCI()`, discussed more in :ref:`custom-extractor`.

The default feature extraction used is :ref:`GeneralFeatureChoices` found in `FeatureSettings.py <https://github.com/LMBooth/pybci/blob/main/pybci/Configuration/FeatureSettings.py>`_, see :ref:`generic-extractor` for more details.

2.4 Classifier Thread
**********************************************
The Classifier thread is responsible for receiving data from the various feature extraction threads, synchronising based on the number of target data streams, then uses the features and target marker values for testing and training the selected machine learning tensorflow or scikit-learn model or classifier. If a valid marker stream and datastream/s are available we can start the bci machine learning training by calling :func:`PyBCI.TrainMode()`.

Once in test mode a datathreads continuously slice time windows of data and optionally overlap these windows - according to :class:`globalEpochSettings`when initialising :py:class:`PyBCI()` - nd test the extracted features against the currently fit model. 

If the model is not performing well the user can always swap back to training model to gather more data with :func:`PyBCI.TestMode()`.

To set you own clf and model see the examples found `here for sklearn <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testSklearn.py>`_, and `here for tensorflow <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testTensorflow.py>`_.

3. Testing and Training the Model
=========================================================

3.2 Training
**********************************************
3.2.1 Retrieiving current estimate
-----------------------------------------
Before the classifier can be run a minimum number of marker strings must be received for each type of target marker, set with the `minimumEpochsRequired` variable (default: 10) to :py:class:`PyBCI()`.

An sklearn classifier of the users choosing can be passed with the `clf` variable, or a tensorflow model with passed to `model` when instantiating with :py:class:`PyBCI()`.

The classifier performance or updated model/clf types can be queried by calling :func:`PyBCI.CurrentClassifierInfo()` example:

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
It is recommended to periodically query the current estimated marker with 

.. code-block:: python

    classGuess = bci.CurrentClassifierMarkerGuess()

where classGuess is an index value relating to the marker value in the marker dict returned with :func:`PyBCI.ReceivedMarkerCount()`.

3.2.2 Resetting or Adding to Train mode Feature Data
-----------------------------------------------
The user can call :func:`PyBCI.TrainMode()` again to go back to training the model and add to the existing feature data with new LSL markers signifying new epochs to be processed.
