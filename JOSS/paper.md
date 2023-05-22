---
title: 'PyBCI: A Python Package for Brain-Computer Interface (BCI) Design/An Open Source Brain-Computer Interface Framework in Python'
tags:
  - braincomputerinterface
  - python
  - bci
  - lsl
  - labstreminglayer
  - machinelearning
authors:
 - name: Liam Booth
   orcid: 0000-0002-8749-9726
   affiliation: "1"
 - name: Anthony Bateson
   orcid: ----
   affiliation: "1"
affiliations:
 - name: Faculty of Science and Engineering, University of Hull.
   index: 1
date: 22 May 2023
bibliography: paper.bib
---

# Abstract:

PyBCI is a comprehensive, open-source Python framework developed to facilitate brain-computer interface (BCI) research. It encompasses data acquisition, data labelling, feature extraction, and machine learning. PyBCI provides a streamlined, user-friendly platform for conducting real-time BCI applications. The software uses the Lab Streaming Layer (LSL) protocol for data acquisition on various data streams, where a (LSL) marker stream for labelling training data, Scikit-learn and TensorFlow are levarged for machine learning, and Antropy, NumPy, and SciPy for feature extraction.

# Statement of Need:

BCIs have the potential to revolutionise the way we interact with computers and understand the brain. However, BCI research is often limited by the availability of open-source, comprehensive, and user-friendly software. PyBCI addresses this gap, offering a flexible and robust Python-based platform for conducting BCI research. It integrates seamlessly with the LSL for data acquisition and labelling, utilizing popular machine learning packages such as Scikit-learn and TensorFlow, and employing Antropy, NumPy, and SciPy for an example feature extraction or optionally allowing the user to pass a custom feature extractor to process specific device data. This combination of software features allows researchers to focus on their experiments rather than software development, expediting the advancement of BCI technology.

# Software functionality and performance:

PyBCI provides an end-to-end solution for BCI research. It uses the Lab Streaming Layer (LSL) to handle data acquisition and labelling, allowing for real-time, synchronous data collection from multiple devices (Kothe, 2014). Samples are streamed through datastream FIFOs and stored when in training mode based on a configurable time window before and after each marker type. When in test mode data is continuously processed and analysed based on the global epoch timing settings.  For feature extraction, PyBCI leverages the power of Antropy, NumPy, and SciPy, robust Python libraries known for their efficiency in handling numerical operations (Oliphant, 2006; Virtanen et al., 2020; Vallat, 2023). Machine learning, a crucial component of BCI research, is facilitated through Scikit-learn and TensorFlow. Scikit-learn offers a wide range of algorithms for classification, regression, and clustering (Pedregosa et al., 2011), while TensorFlow provides a comprehensive ecosystem for developing and training machine learning models (Abadi et al., 2016).

## Theory of operation:
1. Requirements Prior Initialising with bci = PyBCI()
The bci must have ==1 LSL marker stream selected (if more then one LSL marker stream on system set the desired ML training marker stream with markerStream to PyBCI()). Warning: If None set it picks first available in list, if more then one marker stream available to LSL then it is advised to hard select on intialisation.

2. Thread Creation
Once configuration settings are set various threads are created.

2.1 Marker Thread
The marker stream has its own thread which recieves markers from the target LSL marker stream and when in train mode, the marker thread pushed the marker to all available data threads informing when to slice the data, see 2. Setting Custom Epoch Times. Set the desired ML training marker stream with markerStream to PyBCI().

2.2 Data Threads
Each data stream has its two threads created, one data and one feautre extractor, the thread is responsible for pipelining received data on deque FIFO’s and optionally slicing and overlapping so many seconds before and after the marker appropriately based on the classes GlobalEpochSettings and IndividualEpochSettings, set with globalEpochSettings and customEpochSettings when initialising PyBCI().

Add desired dataStreams by passing a list of accepted data stream names with dataStreams.

Upon data thread creation the effective sample rate is queried for each LSL data stream, if the sample rate is 0 an Asynchronous thread is created for FIFO handling, though potentially more accurate, it is far more computationally intensive to slice data than the synchronous data thread. If n effective sample rate greater than 0 is supplised by the LSL datastream a syncrhnous data thread is used for slicing epochs relative to markers in training mode and continously slices in testing mode.

2.3 Feature Extractor Threads
The feature extractor threads receive data from their corresponding data stream thread and prepares epoch data for reunification in the classification thread with other devices in the same epoch.

The feature extraction techniques used can vary drastically between devices, to resolve this custom classes can be created to deal with specific stream types and passed to streamCustomFeatureExtract when initialising PyBCI(), discussed more in 2. Passing Custom Feature Extractor classes.

The default feature extraction used is GeneralFeatureChoices found in FeatureSettings.py, see 1. Generic Time-Series Feature Extractor for more details.

2.4 Classifier Thread
The Classifier thread is responsible for receiving data from the various feature extraction threads, synchronising based on the number of target data streams, then uses the features and target marker values for testing and training the selected machine learning tensorflow or scikit-learn model or classifier. If a valid marker stream and datastream/s are available we can start the bci machine learning training by calling PyBCI.TrainMode().

Once in test mode a datathreads continuously slice time windows of data and optionally overlap these windows - according to globalEpochSettings`when initialising :py:class:`PyBCI() - nd test the extracted features against the currently fit model.

If the model is not performing well the user can always swap back to training model to gather more data with PyBCI.TestMode().

To set you own clf and model see the examples found here for sklearn, and here for tensorflow.

3. Testing and Training the Model
3.2 Training
3.2.1 Retrieiving current estimate
Before the classifier can be run a minimum number of marker strings must be received for each type of target marker, set with the minimumEpochsRequired variable (default: 10) to PyBCI().

An sklearn classifier of the users choosing can be passed with the clf variable, or a tensorflow model with passed to model when instantiating with PyBCI().

The classifier performance or updated model/clf types can be queried by calling PyBCI.CurrentClassifierInfo() example:

bci = PyBCI()
classInfo = bci.CurrentClassifierInfo()
Where classInfo is a dict of:

classinfo = {
   "clf":self.classifier.clf,
   "model":self.classifier.model,
   "accuracy":self.classifier.accuracy
}
3.2 Testing
3.2.1 Retrieiving current estimate
It is recommended to periodically query the current estimated marker with

classGuess = bci.CurrentClassifierMarkerGuess()
where classGuess is an index value relating to the marker value in the marker dict returned with PyBCI.ReceivedMarkerCount().

3.2.2 Resetting or Adding to Train mode Feature Data
The user can call PyBCI.TrainMode() again to go back to training the model and add to the existing feature data with new LSL markers signifying new epochs to be processed.

# Impact:

By offering a comprehensive, open-source platform for BCI research, PyBCI has the potential to drive the field forward. It provides researchers with the tools necessary to conduct advanced BCI experiments without requiring extensive software development skills. The integration of LSL, Scikit-learn, TensorFlow, Antropy, NumPy, and SciPy into one platform simplifies the research process, encouraging innovation and collaboration in the field of BCI.

# References:


## Needs adding to .bib and removing from here
Abadi, M., Agarwal, A., Barham, P., et al. (2016). TensorFlow: Large-scale machine learning on heterogeneous distributed systems.
Vallat, R. (2023). Antropy. GitHub. Retrieved May 22, 2023, from https://github.com/raphaelvallat/antropy
Kothe, C. (2014). Lab Streaming Layer (LSL).
Oliphant, T. E. (2006). A guide to NumPy.
Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12,


(((Please note that a complete JOSS paper would also need to include sections on the software's functionality, method of use, and an example of use in research, as well as any acknowledgements and references. )))
