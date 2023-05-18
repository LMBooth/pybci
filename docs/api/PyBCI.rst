=========
PyBCI API
=========

.. module:: pybci

Classes
=======

.. class:: PyBCI
   .. method:: __init__(dataStreams = None, markerStream= None, streamTypes = None, markerTypes = None, printDebug = True,globalEpochSettings = GlobalEpochSettings(), customEpochSettings = {}, streamChsDropDict = {}, freqbands = [[1.0, 4.0], [4.0, 8.0], [8.0, 12.0], [12.0, 20.0]], featureChoices = GeneralFeatureChoices(), minimumEpochsRequired = 10, clf= None, model = None)

      :dataStreams: 
         list(string) - list of target data streams.
      :markerStream: 
         string - target marker training stream.
      :streamTypes: 
         list(string) - list of target data stream types if no specified streams set with dataStreams.
      :markerTypes: 
         list(string) - list of target data marker types if no specified marker set with markerStream.
      :printDebug: 
         bool - (default True) Sets whether PyBCI debug messages are printed.

      :globalEpochSettings: 
         GlobalEpochSettings() - (default ) can be found in pybci.Configurations folder, sets global epoch timing settings
         - splitCheck: bool (default False) - Checks whether or not subdivide epochs.
         - tmin: int (default 0) - Time in seconds to capture samples before marker.
         - tmax: int (default 1) - Time in seconds to capture samples after marker.
         - windowLength: float (default 0.5) - If splitcheck true - time in seconds to split epoch. 
         - windowOverlap: float(default 0.5) - If splitcheck true  percentage value > 0 and < 1, example if epoch has tmin of 0 and tmax of 1 with window.
      :customEpochSettings: dict(str:IndividualEpochSetting()) - Each key in the dict specifies the target marker received on the marker stream and sets if the target epoch should have its time window cut up. 
         IndividualEpochSetting
         - splitCheck: bool (default False) Checks whether or not subdivide epochs. (Note: If True, divides epoch based on window global overlap and length as all have to be uniform to match with testmode window size)
         - tmin: int (default 0) Time in seconds to capture samples before marker.
         - tmax: int (default 1) Time in seconds to capture samples after marker.
      :streamChsDropDict: dict(str:list(int)) - Each key specifies the datastream and the list of indicies specifies which channels to drop in that keys stream.
      :freqbands: list(list()) - (default [[1.0, 4.0], [4.0, 8.0], [8.0, 12.0], [12.0, 20.0]]) 2D list of frequency bands for feature extraction where 1st dimension is m extensible and 2nd must have a length of 2 [lowerFc, higherFc].
      :featureChoices: GeneralFeatureChoices() - (default GeneralFeatureChoices()) - Sets trget features for decoding from time series data (pybci.utils.FeatureExtractor) 
      :minimumEpochsRequired: Int (default 10) minimum number of required epochs before model compiling begins (Warning: too low an suffer from inadequate test train epoch splitting for accuracy validation)
      :clf: sklearn.base.ClassifierMixin() - (default SVM) allows user sklearn clf to be passed, if no model or clf is passed then defaults to sklearn SVM with rbf kernel.
      :model: tf.keras.Model() - allows user tensorflow model to be passed, if no model or clf is passed then defaults to sklearn SVM with rbf kernel.

   .. method:: Connect()
      Checks LSL for avilable Marker and Data streams.

   .. method:: TrainMode()
      Sets to train mode
   
   .. method:: TestMode()
      Sets to Test mode