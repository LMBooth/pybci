Feature Selection
############
.. _generic-extractor:
Generic Time-Series Feature Extractor
--------------------------------
The `FeatureSettings class GeneralFeatureChoices <https://github.com/LMBooth/pybci/blob/main/pybci/Configuration/FeatureSettings.py>`_ is perfect for generic time-series feature extractor.
The FeatureExtractor.py file is part of the pybci project and is used to extract various features from time-series data, such as EEG, EMG, EOG or other consistent data with a consistent sample rate. The type of features to be extracted can be specified during initialization, and the code supports extracting various types of entropy features, average power within specified frequency bands, root mean square, mean and median of power spectral density (PSD), variance, mean absolute value, waveform length, zero-crossings, and slope sign changes.

The file provides a class GenericFeatureExtractor, which is initialized with freqbands and featureChoices.

The freqbands argument is a list of frequency bands for which the average power is to be calculated. By default, it is set to [[1.0, 4.0], [4.0, 8.0], [8.0, 12.0], [12.0, 20.0]], corresponding to typical EEG frequency bands.

The featureChoices argument is an instance of the GeneralFeatureChoices class, which allows the user to specify which features they want to calculate. The features can be selected by setting the respective attributes in the GeneralFeatureChoices class to True.

The GenericFeatureExtractor class has a method ProcessFeatures, which takes an epoch (a 2D array of time-series data), a sampling rate, and a target marker type as inputs, and returns a 2D numpy array of the calculated features for each channel and the target marker type.

The ProcessFeatures method calculates the power spectral density (PSD) for each channel using the Welch method, and then computes the specified features based on the calculated PSD and the raw time-series data.

.. _raw-extractor:
Raw time-series
----------------
(Feature still needs adding)

.. _custom-extractor:
Passing Custom Feature Extractor classes 
--------------------------------

(Feature still needs adding)
