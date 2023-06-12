Feature Selection
############
.. _feature-debugging:
Recommended Debugging
--------------------------------
When initialisaing the :class:`PyBCI()` class we can set :class:`logger` to "TIMING" to time our feature extraction time, note a warning will be produced if the feature extraction time is longer then the windowLength*(1-windowOverlap), if this is the case a delay will continuously grow as data builds in the queues. To fix this reduce channel count, feature count, feature complexity, or sample rate until the feature extraction time is acceptable, this will help create near-real-time classification.


.. _generic-extractor:
Generic Time-Series Feature Extractor
--------------------------------

The `generic feature extractor class found here <https://github.com/LMBooth/pybci/blob/main/pybci/Utils/FeatureExtractor.py>`_ is the default feature extractor for obtaining generic time-series features to a 1d list for classification, note this is used if nothing is passed to :class:`streamCustomFeatureExtract` for its respective datastream. See :ref:`custom-extractor` and :ref:`raw-extractor` for other feature extraction methods.

The available features can be  for each of the below boolean feature choices. The `FeatureSettings class GeneralFeatureChoices <https://github.com/LMBooth/pybci/blob/main/pybci/Configuration/FeatureSettings.py>`_ gives a quick method for selecting the time and/or frequency based feature extraction techniques - useful for reducing stored data and computational complexity.

The features can be selected by setting the respective attributes in the :class:`GeneralFeatureChoices` class to True. When initialising :class:`PyBCI()` we can pass :class:`configuration.GeneralFeatureChoices()` to :class:`featureChoices` which offers a list of booleans to decide the following features, not all options are set by default to reduce computation time:

.. code-block:: python

  class GeneralFeatureChoices:
    psdBand = True
    appr_entropy = False
    perm_entropy = False
    spec_entropy = False
    svd_entropy = False
    samp_entropy = False
    rms = True
    meanPSD = True
    medianPSD = True
    variance = True
    meanAbs = True
    waveformLength = False
    zeroCross = False
    slopeSignChange = False


If :class:`psdBand == True` we can also pass custom :class:`freqbands` when initialising :class:`PyBCI()`, which can be an extensible list of lists, where each inner list has a length of two floats representing the upper and lower frequency band to get the mean power of. The :class:`freqbands` argument is a list of frequency bands for which the average power is to be calculated. By default, it is set to [[1.0, 4.0], [4.0, 8.0], [8.0, 12.0], [12.0, 20.0]], corresponding to typical EEG frequency bands.

The `FeatureExtractor.py <https://github.com/LMBooth/pybci/blob/main/pybci/Utils/FeatureExtractor.py>`_ file is part of the pybci project and is used to extract various features from time-series data, such as EEG, EMG, EOG or other consistent data with a consistent sample rate. The type of features to be extracted can be specified during initialisation, and the code supports extracting various types of entropy features, average power within specified frequency bands, root mean square, mean and median of power spectral density (PSD), variance, mean absolute value, waveform length, zero-crossings, and slope sign changes.

.. _custom-extractor:
Passing Custom Feature Extractor classes 
--------------------------------
Due to the idiosyncratic nature of each LSL data stream and the potential pre-processing/filtering that may be required before data is passed to the machine learning classifier, it can be desirable to have custom feature extraction classes passed to :class:`streamCustomFeatureExtract` When initialising :class:`PyBCI()`. 

:class:`streamCustomFeatureExtract` is a dict where the key is a string for the LSL datastream name and the value is the custom created class that will be used for data on that LSL type, example:

.. code-block:: python

  class EMGClassifier():
    def ProcessFeatures(self, epochData, sr, epochNum): # Every custom class requires a function with this name and structure to extract the featur data and epochData is always [Samples, Channels]
        rmsCh1 = np.sqrt(np.mean(np.array(epochData[:,0])**2)))
        rmsCh2 = np.sqrt(np.mean(np.array(epochData[:,1])**2))) 
        rmsCh3 = np.sqrt(np.mean(np.array(epochData[:,2])**2))) 
        rmsCh4 = np.sqrt(np.mean(np.array(epochData[:,3])**2))) 
        varCh1 = np.var(epochData[:,0]) 
        varCh2 = np.var(epochData[:,1]) 
        varCh3 = np.var(epochData[:,2]) 
        varCh4 = np.var(epochData[:,3]) 
        return [rmsCh1, rmsCh2,rmsCh3,rmsCh4,varCh1,varCh2,varCh3,varCh4]
        
  streamCustomFeatureExtract = {"EMG":EMGClassifier()}
  bci = PyBCI(streamTypes = ["EMG"], streamCustomFeatureExtract=streamCustomFeatureExtract)

NOTE: Every custom class for processing features requires the features to be processed in a function labelled with corresponding arguements as above, namely  :class:`def ProcessFeatures(self, epochData, sr, epochNum):`, the epochNum may be handy for distinguishing baseline information and holding that baseline information in the class to use with features from other markers (pupil data: baseline diameter change compared to stimulus, ECG: resting heart rate vs stimulus, heart rate variability, etc.). Look at :ref:`examples` for more inspiriation of custom class creation and integration. 

:class:`epochData` is a 2D array in the shape of [samps,chs] where chs is the number of channels on the LSL datastream after any are dropped with the variable :class:`streamChsDropDict` and samps is the number of samples captured in the epoch time window depending on the :class:`globalEpochSettings` and :class:`customEpochSettings` - see :ref:`_epoch_timing` for more information on epoch time windows.

The above example returns a 1d array of features, but the target model may specify greater dimensions. More dimensions may be desirable for some pytorch and tensorflow models, but less applicable for sklearn classifiers, this is specific to the model selected.

A practical example of custom datastream decoding can be found in the `Pupil Labs example <https://github.com/LMBooth/pybci/tree/main/pybci/Examples/PupilLabsRightLeftEyeClose>`_, where in the `bciGazeExample.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/PupilLabsRightLeftEyeClose/bciGazeExample.py>`_ file there is a custom class; :class:`PupilGazeDecode()`, which is a very simply example getting the mean pupil diameter of the left, right and both eyes as feature data, then this is used to classify whether someone has their right or left eye closed or both eyes open.


.. _raw-extractor:
Raw time-series
----------------
If the raw time-series data is wanted to be the input for the classifier we can pass a custom class which will allow us to retain a 2d array of [samples, channels] as the input for our model, example given below:

.. code-block:: python
  class RawDecode():
      desired_length = 0
      def ProcessFeatures(self, epochData, sr, target): 
          d = epochData.T
          if self.desired_length == 0: # needed as windows may be differing sizes due to timestamp varience on LSL
              self.desired_length = d.shape[1]
          if d.shape[1] != self.desired_length:
              d = np.resize(d, (d.shape[0],self.desired_length))
          return d 

It should be noted that the default sklean svm used only accepts a 2D array of [epochs, features] not [epochs, samples, channels], however a pytorch CNN or RNN may be more approriate for multi-channel time-series data. A full example of the above using a PyTorch C-NN can be found in the `testRaw.py file here <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testRaw.py>`_, multiple channels are dropped to reduce training and testing time.
