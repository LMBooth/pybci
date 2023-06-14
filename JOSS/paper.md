---
title: 'PyBCI: A Python Package for Brain-Computer Interface (BCI) Design/An Open Source Brain-Computer Interface Framework in Python'
tags:
  - brain-computer-interface
  - python
  - bci
  - lsl
  - labstreaminglayer
  - machinelearning
authors:
 - name: Liam Booth
   orcid: 0000-0002-8749-9726
   affiliation: "1"
 - name: Anthony Bateson
   orcid: ----
   affiliation: "1"
 - name: Aziz Asghar
   orcid: ----
   affiliation: "2"
affiliations:
 - name: Faculty of Science and Engineering, University of Hull.
   index: 1
 - name: Hull York Medical School.
   index: 2
date: 22 May 2023
bibliography: paper.bib
---

# Abstract:

PyBCI is a comprehensive, open-source Python framework developed to facilitate brain-computer interface (BCI) research. It encompasses data acquisition, data labelling, feature extraction, and machine learning. PyBCI provides a streamlined, user-friendly platform for conducting real-time BCI applications. The software uses the Lab Streaming Layer (LSL) protocol for data acquisition on various LSL enabled data streams, where a (LSL) marker stream is used for labelling training data. PyTorch, Scikit-learn and TensorFlow are leveraged for machine learning, and Antropy, NumPy, and SciPy for feature extraction.

# Statement of Need:

PyBCI puts emphasis on quick and easy customisation of applied time-series feature extraction techniques and machine learning models, live data classification, and integration into other systems. Remaining lightweight with solely Python packages and no additional visualisation and recording tools, it's assumed a user will have these tools available with their LSL hardware. 

BCIs have the potential to revolutionise the way we interact with computers. However, BCI research is often limited by the availability of open-source, comprehensive, and user-friendly software. PyBCI addresses this gap, offering a simple, flexible, and robust Python-based platform for conducting BCI research. It integrates seamlessly with the LSL for data acquisition and labelling, utilizing popular machine learning packages such as PyTorch, Scikit-learn and TensorFlow, employing Antropy, NumPy, and SciPy for an example feature extraction or optionally allowing the user to pass a custom feature extractor to process specific device data. This combination of software features allows researchers to focus on their experiments rather than software development, expediting the advancement of BCI technology.

# Software functionality and performance:

PyBCI provides an end-to-end solution for BCI research. It uses the Lab Streaming Layer (LSL) to handle data acquisition and labelling, allowing for real-time, synchronous data collection from multiple devices (Kothe, 2014). Samples are collected in chunks from the LSL data streams and stored in pre-allocated NumPy arrays. When in training mode based on a configurable time window before and after each marker type. When in test mode data is continuously processed and analysed based on the global epoch timing settings.  For feature extraction, PyBCI leverages the power of Antropy, NumPy, and SciPy, robust Python libraries known for their efficiency in handling numerical operations (Oliphant, 2006; Virtanen et al., 2020; Vallat, 2023). Machine learning, a crucial component of BCI research, is facilitated through PyTorch, Scikit-learn and TensorFlow. Scikit-learn offers a wide range of traditional algorithms for classification, including things like regression, and clustering (Pedregosa et al., 2011), while TensorFlow and PyTorch provide comprehensive ecosystems for developing and training bespoke deep learning machine learning models (Abadi et al., 2016).

## Theory of Operation:

### Requirements Prior Initialising with `bci = PyBCI()`
The BCI must have ==1 LSL marker stream selected and >=1 LSL data stream/s selected - if more then one LSL marker stream is on the system it is recommended to set the desired ML training marker stream with `markerStream` to  `PyBCI()`, otherwise the first in list is selected. If no set datastreams are selected with `dataStream` to `PyBCI()` all available datastreams will be used and decoded with the `generic-extractor`.

### Thread Creation
Once configuration settings are set 4 types of threaded operations are created: one classifier, one marker thread, with a feature and a data thread for each accepted LSL datastream.

#### Marker Thread
The marker stream has its own thread which receives markers from the target LSL marker stream and when in train mode, the marker thread pushed the marker to all available data threads informing when to slice the data, see `set_custom_epoch_times`. Set the desired ML training marker stream with `markerStream` to `PyBCI()`.

#### Data Threads
Each data stream has two threads created one data and one feature extractor. The data thread is responsible for setting pre-allocated NumPy arrays for each data stream inlet which pulls chunks of data from the LSL. When in training mode it gathers data so many seconds before and after a marker to prepare for feature extraction, with the option of slicing and overlapping so many seconds before and after the marker appropriately based on the classes [GlobalEpochSettings](https://github.com/LMBooth/pybci/blob/main/pybci/Configuration/EpochSettings.py)  and [IndividualEpochSettings](https://github.com/LMBooth/pybci/blob/main/pybci/Configuration/EpochSettings.py), set with `globalEpochSettings` and `customEpochSettings` when initialising `PyBCI()`.

Add desired dataStreams by passing a list of strings containing accepted data stream names with `dataStreams`. By setting `dataStreams` all other data inlets will be ignored except those in the list.

Note: Data so many seconds before and after the LSL marker timestamp is decided by the corresponding LSL data timestamps. If the LSL data stream pushes chunks infrequently (>[windowLength*(1-windowOverlap)]) and doesn't overwrite each sample with linear equidistant timestamps errors in classification output will occur.

#### Feature Extractor Threads
The feature extractor threads receive data from their corresponding data thread and prepares epoch data for re-unification in the classification thread with other devices in the same epoch.

The feature extraction techniques used can vary drastically between devices, to resolve this custom classes can be created to deal with specific stream types and passed to `streamCustomFeatureExtract` when initialising `PyBCI()`, discussed more in `custom-extractor`.

The default feature extraction used is `GenericFeatureExtractor` found in [FeatureSettings.py](https://github.com/LMBooth/pybci/blob/main/pybci/Utils/FeatureExtractor.py), with `GeneralFeatureChoices` found in [FeatureSettings.py](https://github.com/LMBooth/pybci/blob/main/pybci/Configuration/FeatureSettings.py), see `generic-extractor` for more details.

#### Classifier Thread
The Classifier thread is responsible for receiving data from the various feature extraction threads, synchronising based on the number of data streams, then using the features and target marker values for testing and training the selected machine learning PyTorch, TensorFlow or SKLearn model. 

If a valid marker stream and datastream/s are available `PyBCI()` can start machine learning training by calling `TrainMode()`. In training mode strings are received on the selected LSL marker stream which signify a machine learning target value has occurred. A minimum number of each type of string type are required before classification beings, which can be modified with `minimumEpochsRequired` to `PyBCI()` on initialisation. Only after this number has been received of each and a suitable classification accuracy has been obtained should the BCI start test mode. Call `TestMode()` on the `PyBCI()` object to start testing the machine learning model.

Once in test mode the data threads continuously slice time windows of data based on `globalEpochSettings.windowLength` and optionally overlaps these windows according to `globalEpochSettings.windowOverlap` when initialising `PyBCI()`. These windows have features extracted the same as in test mode, then the extracted features are applied to the model/classifier to predict the current target. 

If the model is not performing well the user can always swap back to training model to gather more data with `PyBCI.TestMode()`. It could also be worth to record view it in post to adjust the epoch classifier timing windows accordingly.

Custom Sci-Kit-Learn classifier and PyTorch models can be used, see the examples found [here for sklearn](https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testSklearn.py), and  [here for PyTorch](https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testPyTorch.py).

TensorFlow can also be used [found here](https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testTensorflow.py), (Should be noted in PyBCI there is currently no suppression for TensorFlow text prompts and the model training and testing time can be substantially longer then PyTorch and SKLearn. Any recommendations are welcome in the pull-requests and issues on the git!)
# Impact:

By offering a comprehensive, open-source platform for BCI research, PyBCI has the potential to drive the field forward. It provides researchers with the tools necessary to conduct advanced BCI experiments without requiring extensive software development skills. The integration of LSL, PyTorch, Scikit-learn, TensorFlow, Antropy, NumPy, and SciPy into one platform simplifies the research process, encouraging innovation and collaboration in the field of brain computer/human machine interfaces.


