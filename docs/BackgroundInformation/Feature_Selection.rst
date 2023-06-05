Feature Selection
############
.. _generic-extractor:
Generic Time-Series Feature Extractor
--------------------------------

The `generic feature extractor class found here <https://github.com/LMBooth/pybci/blob/main/pybci/Utils/FeatureExtractor.py>`_ shows how :class:`GenericFeatureExtractor()` is computationally executed for each of the below boolean feature choices. The `FeatureSettings class GeneralFeatureChoices <https://github.com/LMBooth/pybci/blob/main/pybci/Configuration/FeatureSettings.py>`_ gives a quick method for selecting the time and/or frequency based feature extraction techniques - useful for reducing overall stored data.

The features can be selected by setting the respective attributes in the GeneralFeatureChoices class to True. When initialising :class:`PyBCI()` we can pass :class:`configuration.GeneralFeatureChoices()` to :class:`featureChoices` which offers a list boolean for the following features:

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


If :class:`psdBand == True` we can also pass custom :class:`freqbands` when initialising :class:`PyBCI()`, which can be an exensible list of lists, where each inner list has a length of two representing the upper and lower frequency band to get the mean power of. The :class:`freqbands` argument is a list of frequency bands for which the average power is to be calculated. By default, it is set to [[1.0, 4.0], [4.0, 8.0], [8.0, 12.0], [12.0, 20.0]], corresponding to typical EEG frequency bands.

The `FeatureExtractor.py <https://github.com/LMBooth/pybci/blob/main/pybci/Utils/FeatureExtractor.py>`_ file is part of the pybci project and is used to extract various features from time-series data, such as EEG, EMG, EOG or other consistent data with a consistent sample rate. The type of features to be extracted can be specified during initialization, and the code supports extracting various types of entropy features, average power within specified frequency bands, root mean square, mean and median of power spectral density (PSD), variance, mean absolute value, waveform length, zero-crossings, and slope sign changes.

.. _custom-extractor:
Passing Custom Feature Extractor classes 
--------------------------------
Due to the idiosyncratic nature of each LSL data stream and the potential pre-processing/filtering that may be required before data is passed to the machine learning classifier, it can be desirable to have custom feature extraction classes passed to :class:`streamCustomFeatureExtract` When initialising :class:`PyBCI()`. 

:class:`streamCustomFeatureExtract` is a dict where the key is a string for the LSL datastream name and the value is the custom created class that will be used for data on that LSL type, example:

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

the return of the function should be a 1d array of features, unless the target model specifies gerater dimensions More dimensions may be desirable for some tensorflow models, but less applicable for sklearn classifiers.

A practical example of custom datastream decoding can be found in the `Pupil Labs example <https://github.com/LMBooth/pybci/tree/main/pybci/Examples/PupilLabsRightLeftEyeClose>`_, where in the `bciGazeExample.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/PupilLabsRightLeftEyeClose/bciGazeExample.py>`_ file there is a custom class; :class:`PupilGazeDecode()`, which is a very simply getting the mean pupil diameter of the left, right and both eyes as feature data, then this is used to classify whether someone has their right or left eye closed or both eyes open.


.. _raw-extractor:
Raw time-series
----------------
If the raw time-series data is wanted to be the input for the classifier we can pass a custom class which will allow us to retain a 2d array of channels by samples as the input for our model, though when doing this it is required to pass the correct shape as the input to the model, like the tensorflow example given below:

.. code-block:: python

  num_chs = 8 # 8 channels re created in the PsuedoLSLGwnerator
  sum_samps = 250 # sample rate is 250 in the PsuedoLSLGwnerator
  num_classes = 3 # number of different triggers (can include baseline) sent, defines if we use softmax of binary
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Reshape((num_chs,sum_samps, 1), input_shape=(num_chs,sum_samps)))
  model.add(tf.keras.layers.Permute((2, 1, 3)))
  model.add(tf.keras.layers.Reshape((num_chs*sum_samps, 1)))
  model.add(tf.keras.layers.GRU(units=256))#, input_shape=num_chs*num_feats)) # maybe should show this example as 2d with toggleable timesteps disabled
  model.add(tf.keras.layers.Dense(units=512, activation='relu'))
  model.add(tf.keras.layers.Flatten())#   )tf.keras.layers.Dense(units=128, activation='relu'))
  model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax')) # softmax as more then binary classification (sparse_categorical_crossentropy)
  #model.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # sigmoid as binary classification (binary_crossentropy)
  model.summary()
  model.compile(loss='sparse_categorical_crossentropy',# using sparse_categorical as we expect multi-class (>2) output, sparse because we encode targetvalues with integers
                optimizer='adam',
                metrics=['accuracy'])
  class RawDecode():
      def ProcessFeatures(self, epochData, sr, epochNum): 
          return np.array(epochData) # tensorflow wants [1,chs,samps] for testing model
  streamCustomFeatureExtract = {"sendTest" : RawDecode()} # we select EMG as that is the default type in the psuedolslgenerator example
  bci = PyBCI(minimumEpochsRequired = 4, model = model, streamCustomFeatureExtract=streamCustomFeatureExtract )

