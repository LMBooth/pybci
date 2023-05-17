User Guide
############

Theory of Operation
===================

  1. Requirements Prior Initialising with python bci = PyBCI() 
  ------------------------------------------------------------
    The bci must have >=1 LSL datastream with an accepted dataType ("EEG", "EMG", "Gaze") {hopefully configurable in the future t pass custom fature decoding class}
    The bci must have ==1 LSL markerstream selected (if more then one LSL marker stream on system set the desired ML training marker stream with PyBCI(markerStream="yourMarkerStream")). Warning: If None set picks first available in list.

  2. Once configuration settings are set various threads are created.
  ----------------------------------------------------------------------
    The marker stream has its own thread which recieves markers from the target LSL marker stream and when in train mode pushes this marker to the available datastreams.
    Each data stream has its own thread created responsible for pipleining received data on FIFO's and slicing approprialey based on globalEpochSettings and customEpochSettings.
  
  3. Train Mode
  ----------
    3.1 FeaturesExtractor
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    3.2 Classifier
    ~~~~~~~~~~~~~~


  4.Test Mode
  ----------
    4.1 Estimated Marker and decoding
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    4.2 Resetting or Adding to Train mode Feature Data
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~