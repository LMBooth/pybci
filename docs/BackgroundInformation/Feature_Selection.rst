Feature Selection
############
.. _generic-extractor:
Generic Time-Series Feature Extractor
--------------------------------
The `FeatureSettings class GeneralFeatureChoices <https://github.com/LMBooth/pybci/blob/main/pybci/Configuration/FeatureSettings.py>`_ gives a diverse selection of time and/or frequency based feature extraction techniques - useful for reducing overall stored data.

The :class:`featureChoices` argument is an instance of the :class:`configuration.GeneralFeatureChoices()` class, which allows the user to specify which features they want to calculate. The features can be selected by setting the respective attributes in the GeneralFeatureChoices class to True. When initialising :class:`PyBCI()` we can pass :class:`configuration.GeneralFeatureChoices()` to :class:`featureChoices` which offers a list boolean for the following features:

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


The `generic feature extractor class <https://github.com/LMBooth/pybci/blob/main/pybci/Utils/FeatureExtractor.py>`_ shows how the :class:`GeneralFeatureChoices()` is computationally executed for each of the above boolean choices. 

If :class:`psdBand == True` we can also pass custom :class:`freqbands` when initialising :class:`PyBCI()`, which can be an exensible list of lists, where each inner list has a length of two representing the upper and lower frequency band to get the mean power of. The :class:`freqbands` argument is a list of frequency bands for which the average power is to be calculated. By default, it is set to [[1.0, 4.0], [4.0, 8.0], [8.0, 12.0], [12.0, 20.0]], corresponding to typical EEG frequency bands.

The FeatureExtractor.py file is part of the pybci project and is used to extract various features from time-series data, such as EEG, EMG, EOG or other consistent data with a consistent sample rate. The type of features to be extracted can be specified during initialization, and the code supports extracting various types of entropy features, average power within specified frequency bands, root mean square, mean and median of power spectral density (PSD), variance, mean absolute value, waveform length, zero-crossings, and slope sign changes.

.. _raw-extractor:
Raw time-series
----------------
(Give example for getting raw time series by passing custom class, probably better for R-NN/LSTM/GRU tensorflow models)

.. _custom-extractor:
Passing Custom Feature Extractor classes 
--------------------------------


A practical example of custom datastream decoding can be found in the `Pupil Labs example <https://github.com/LMBooth/pybci/tree/main/pybci/Examples/PupilLabsRightLeftEyeClose>`_, where in the `bciGazeExample.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/PupilLabsRightLeftEyeClose/bciGazeExample.py>`_ file we create our own :class:`PupilGazeDecode()` class which is a very simply mean taker to gauge whether someone has their right or left eye closed or both eyes open.
