Theory of Operation
############

Requirements Prior Initialising with `bci = PyBCI()`
=========================================================
The bci must have ==1 LSL marker stream selected (if more then one LSL marker stream is on the system it is recommended to set the desired ML training marker stream with :class:`markerStream` to  :py:class:`PyBCI()`). Warning: If None set it picks first available in list.

Thread Creation
=========================================================
Once configuration settings are set various threads are created.

Marker Thread
**********************************************
The marker stream has its own thread which recieves markers from the target LSL marker stream and when in train mode, the marker thread pushed the marker to all available data threads informing when to slice the data, see :ref:`set_custom_epoch_times`. Set the desired ML training marker stream with :class:`markerStream` to  :py:class:`PyBCI()`.

Data Threads
**********************************************
Each data stream has two threads created, one data and one feature extractor. The data thread is responsible for setting pre-allocated numpy arrays for each data stream inlet which pulls chunks of data from the LSL. When in training mode it gathers data so many seconds before and after a marker to prepare for feature extraction, with the option of slicing and overlapping so many seconds before and after the marker appropriately based on the classes `GlobalEpochSettings <https://github.com/LMBooth/pybci/blob/main/pybci/Configuration/EpochSettings.py>`_  and `IndividualEpochSettings <https://github.com/LMBooth/pybci/blob/main/pybci/Configuration/EpochSettings.py>`_, set with :class:`globalEpochSettings` and :class:`customEpochSettings` when initialising :py:class:`PyBCI()`.

Add desired dataStreams by passing a list of accepted data stream names with :class:`dataStreams`. By setting :class:`dataStreams` all other data inlets will be ignored except those in this list.

Note: Data so many seconds before and after the relative marker timestamp is decided by the data relative timestamps. If the LSL data stream pushes chunks infrequently [ > (windowLength - (1-windowOverlap))] and doesn't give each sample its own overwritten timestamp issues could occur. (Kept legacy data threads AsyncDataReceiver and DataReceiver in threads folder in case modifications needed based on so many samples before and after decided by expected sample rate if people find this becomes an issue for certain devices)

Feature Extractor Threads
**********************************************
The feature extractor threads receive data from their corresponding data thread and prepares epoch data for re-unification in the classification thread with other devices in the same epoch.

The feature extraction techniques used can vary drastically between devices, to resolve this custom classes can be created to deal with specific stream types and passed to :class:`streamCustomFeatureExtract` when initialising  :py:class:`PyBCI()`, discussed more in :ref:`custom-extractor`.

The default feature extraction used is :ref:`GenericFeatureExtractor` found in `FeatureSettings.py <https://github.com/LMBooth/pybci/blob/main/pybci/Utils/FeatureExtractor.py>`_, with :ref:`GeneralFeatureChoices` found in `FeatureSettings.py <https://github.com/LMBooth/pybci/blob/main/pybci/Configuration/FeatureSettings.py>`_, see :ref:`generic-extractor` for more details.

Classifier Thread
**********************************************
The Classifier thread is responsible for receiving data from the various feature extraction threads, synchronising based on the number of target data streams, then uses the features and target marker values for testing and training the selected machine learning tensorflow or scikit-learn model or classifier. If a valid marker stream and datastream/s are available we can start the bci machine learning training by calling :func:`PyBCI.TrainMode()`.

Once in test mode a datathreads continuously slice time windows of data and optionally overlap these windows - according to :class:`globalEpochSettings`when initialising :py:class:`PyBCI()` - nd test the extracted features against the currently fit model. 

If the model is not performing well the user can always swap back to training model to gather more data with :func:`PyBCI.TestMode()`.

To set you own clf and model see the examples found `here for sklearn <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testSklearn.py>`_, and `here for tensorflow <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testTensorflow.py>`_.

The figure below illustrates the general flow of data between threads on initialisation:

.. image:: ../Images/flowchart/Flowchart.svg
   :alt: Alternative text describing the image

Testing and Training the Model
=========================================================

Training
**********************************************
Retrieiving current estimate
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

When in test mode data is captured :class:`tmin` seconds before the training marker and :class:`tmax` after the marker, if the :class:`splitCheck` otion is True then the epochs will be sliced up and overlapped set by the :class:`globalEpochSettings` :class:`windowLength` and :class:`overlap` options, see :ref:`set_custom_epoch_times` for more information and illustrations.


Testing
**********************************************
Retrieiving current estimate
-----------------------------------------------
When in test mode the data threads will continously pass time windows to the respective feature extractor threads. 

It is recommended to periodically query the current estimated marker with:

.. code-block:: python

    classGuess = bci.CurrentClassifierMarkerGuess()

where :class:`classGuess` is an integer relating to the marker value in the marker dict returned with :func:`PyBCI.ReceivedMarkerCount()`. See the :ref:`examples` for reference on how to setup sufficient training before switching to test mode and quering live classification esitmation. 

3.2.2 Resetting or Adding to Train mode Feature Data
-----------------------------------------------
The user can call :func:`PyBCI.TrainMode()` again to go back to training the model and add to the existing feature data with new LSL markers signifying new epochs to be processed.
