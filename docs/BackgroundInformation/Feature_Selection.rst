Feature Selection
############
.. _generic-extractor:
1. Generic Time-Series Feature Extractor
--------------------------------

The `generic feature extractor class found here <https://github.com/LMBooth/pybci/blob/main/pybci/Utils/FeatureExtractor.py>`_ shows how :class:`GenericFeatureExtractor()` is computationally executed for each of the below boolean feature choices. The `FeatureSettings class GeneralFeatureChoices <https://github.com/LMBooth/pybci/blob/main/pybci/Configuration/FeatureSettings.py>`_ gives a quick method for selecting the time and/or frequency based feature extraction techniques - useful for reducing overall stored data.

The features can be selected by setting the respective attributes in the GeneralFeatureChoices class to True. When initialising :class:`PyBCI()` we can pass :class:`configuration.GeneralFeatureChoices()` to :class:`featureChoices` which offers a list boolean for the following features:

.. code-block:: python

  class GeneralFeatureChoices:
    psdBand = True
    appr_entropy = True
    perm_entropy = True
    spec_entropy = True
    svd_entropy = True
    samp_entropy = True
    rms = True
    meanPSD = True
    medianPSD = True
    variance = True
    meanAbs = True
    waveformLength = True
    zeroCross = True
    slopeSignChange = True


If :class:`psdBand == True` we can also pass custom :class:`freqbands` when initialising :class:`PyBCI()`, which can be an exensible list of lists, where each inner list has a length of two representing the upper and lower frequency band to get the mean power of. The :class:`freqbands` argument is a list of frequency bands for which the average power is to be calculated. By default, it is set to [[1.0, 4.0], [4.0, 8.0], [8.0, 12.0], [12.0, 20.0]], corresponding to typical EEG frequency bands.

The `FeatureExtractor.py <https://github.com/LMBooth/pybci/blob/main/pybci/Utils/FeatureExtractor.py>`_ file is part of the pybci project and is used to extract various features from time-series data, such as EEG, EMG, EOG or other consistent data with a consistent sample rate. The type of features to be extracted can be specified during initialization, and the code supports extracting various types of entropy features, average power within specified frequency bands, root mean square, mean and median of power spectral density (PSD), variance, mean absolute value, waveform length, zero-crossings, and slope sign changes.

.. _raw-extractor:
2. Raw time-series
----------------
(Give example for getting raw time series by passing custom class, probably better for R-NN/LSTM/GRU tensorflow models)

.. _custom-extractor:
3. Passing Custom Feature Extractor classes 
--------------------------------
Due to the idiosyncratic nature of each LSL data stream and the potential pre-processing/filtering that may be required before data is passed to the machine learning classifier, it can be desirable to have custom feature extraction classes passed to :class:`streamCustomFeatureExtract` When initialising :class:`PyBCI()`. 

:class:`streamCustomFeatureExtract` is a dict where the key is a string for the LSL datastream type and the value is the custom created class that will be used for data on that LSL type, example:

.. code-block:: python

  class EMGClassifier():
    def ProcessFeatures(self, epochData, sr, epochNum):
        rmsCh1 = np.sqrt(np.mean(np.array(epochData[0])**2)))
        rmsCh2 = np.sqrt(np.mean(np.array(epochData[1])**2))) 
        rmsCh3 = np.sqrt(np.mean(np.array(epochData[2])**2))) 
        rmsCh4 = np.sqrt(np.mean(np.array(epochData[3])**2))) 
        varCh1 = np.var(epochData[0]) 
        varCh2 = np.var(epochData[1]) 
        varCh3 = np.var(epochData[2]) 
        varCh4 = np.var(epochData[3]) 
        return [rmsCh1, rmsCh2,rmsCh3,rmsCh4,varCh1,varCh2,varCh3,varCh4]
        
  streamCustomFeatureExtract = {"EMG":EMGClassifier()}
  bci = PyBCI(streamTypes = ["EMG"], streamCustomFeatureExtract=streamCustomFeatureExtract)

NOTE: Every custom class for processing features requires the features to be processed in a function labelled with corresponding arguements as above, namely  :class:`def ProcessFeatures(self, epochData, sr, epochNum):`, the epochNum may be handy for distinguishing baseline information and holding that in the class to act use with features from other classes (pupil data: baseline diameter change compared to stimulus, ECG: resting heart rate vs stimulus, heart rate variability, etc.). Look at :ref:`examples` for more inspiriation of custom class creation and integration.

:class:`epochData` is a 2D array in the shape of [chs,samps] where chs is the number of channels on the LSL datastream after any are dropped with the variable :class:`streamChsDropDict` and samps is the number of samples captured in the epoch time window depending on the :class:`globalEpochSettings` and :class:`customEpochSettings` - see :ref:`_epoch_timing` for more information on epoch time windows.

A practical example of custom datastream decoding can be found in the `Pupil Labs example <https://github.com/LMBooth/pybci/tree/main/pybci/Examples/PupilLabsRightLeftEyeClose>`_, where in the `bciGazeExample.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/PupilLabsRightLeftEyeClose/bciGazeExample.py>`_ file there is a custom class; :class:`PupilGazeDecode()`, which is a very simply getting the mean pupil diameter of the left, right and both eyes as feature data, then this is used to classify whether someone has their right or left eye closed or both eyes open.
