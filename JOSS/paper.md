---
title: 'PyBCI: A Python Package for Brain-Computer Interface (BCI) Design'
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
 - name: Aziz Asghar
   orcid: 0000-0002-3735-4449
   affiliation: "2"
 - name: Anthony Bateson
   orcid: 0000-0002-4780-4458
   affiliation: "1"
affiliations:
 - name: Faculty of Science and Engineering, University of Hull.
   index: 1
 - name: Centre for Anatomical and Human Sciences, Hull York Medical School, University of Hull.
   index: 2
date: 26 June 2023
bibliography: paper.bib
---

# Summary:
PyBCI is a comprehensive, open-source Python framework developed to facilitate brain-computer interface (BCI) research. It encompasses data acquisition, data labelling, feature extraction, and machine learning. PyBCI provides a streamlined, user-friendly platform for creating real-time BCI applications. The software uses the Lab Streaming Layer (LSL) [@lsl] protocol for the unified collection of time-series measurement that handles both the networking, time-synchronization and (near-) real-time access (supported LSL devices found here: https://labstreaminglayer.readthedocs.io/info/supported_devices.html). At least one LSL data stream is required and a single marker stream is used for labelling training data. PyTorch [@NEURIPS2019_9015], SciKit-learn [@scikit-learn] and TensorFlow [@tensorflow2015-whitepaper] are leveraged for applying machine learning classifiers. NumPy [@oliphant2006guide], SciPy [@2020SciPy-NMeth], and Antropy [@vallat_antropy_2023] are utilised for generic time and/or frequency feature extraction examples.


PyBCI is a comprehensive, open-source Python framework developed to facilitate brain-computer interface (BCI) research. It encompasses data acquisition, data labelling, feature extraction, and machine learning. PyBCI provides a streamlined, user-friendly platform for conducting real-time BCI applications. The software uses the Lab Streaming Layer (LSL) [@lsl] protocol for data acquisition on various LSL enabled data streams. The LSL is a system for the unified collection of measurement time series in research experiments that handles both the networking (supported LSL devices found here: https://labstreaminglayer.readthedocs.io/info/supported_devices.html). At least one LSL data stream is required and a single marker stream is used for labelling training data. PyTorch [@NEURIPS2019_9015], SciKit-learn [@scikit-learn] and TensorFlow [@tensorflow2015-whitepaper] are leveraged for applying machine learning classifiers. NumPy [@oliphant2006guide], SciPy [@2020SciPy-NMeth], and Antropy [@vallat_antropy_2023] are utilised for generic time and/or frequency feature extraction examples.

# Statement of Need:

PyBCI puts emphasis on quick and easy customisation of applied time-series feature extraction techniques and machine learning models, live data classification, and integration into other systems. Remaining lightweight with solely Python packages and no additional visualisation and recording tools, it's assumed a user will have these tools available with their LSL hardware. 

BCIs have the potential to revolutionise the way we interact with computers. However, BCI research is often limited by the lack of availability of open-source, comprehensive, and user-friendly software. PyBCI addresses this gap, offering a simple, flexible, and robust Python-based platform for conducting BCI research. It integrates seamlessly with the LSL for data acquisition and labelling, utilizing popular machine learning packages such as PyTorch, Scikit-learn and TensorFlow, employing Antropy, NumPy, and SciPy for an example feature extraction or optionally allowing the user to pass a custom feature extractor to process specific device data. This combination of software features allows researchers to focus on their experiments rather than software development, expediting the advancement of BCI technology.

# Software functionality and performance:

PyBCI provides an end-to-end solution for BCI research. It uses the Lab Streaming Layer (LSL) to handle data acquisition and labelling, allowing for real-time, synchronous data collection from multiple devices [@lsl]. Samples are collected in chunks from the LSL data streams and stored in pre-allocated NumPy arrays. When in training mode based on a configurable time window before and after each marker type. When in test mode data is continuously processed and analysed based on the global epoch timing settings.  For feature extraction, PyBCI leverages the power of NumPy [@oliphant2006guide], SciPy [@2020SciPy-NMeth], and Antropy [@vallat_antropy_2023] robust Python libraries known for their efficiency in handling numerical operations. Machine learning, a crucial component of BCI research, is facilitated with PyTorch [@NEURIPS2019_9015], SciKit-learn [@scikit-learn] and TensorFlow [@tensorflow2015-whitepaper]. Scikit-learn offers a wide range of traditional algorithms for classification, including things like regression, and clustering, while TensorFlow and PyTorch provide comprehensive ecosystems for developing and training bespoke deep learning machine learning models.

# Impact:

By offering a comprehensive, open-source platform for BCI research, PyBCI has the potential to drive the field forward. It provides researchers with the tools necessary to conduct advanced BCI experiments without requiring extensive software development skills. The integration of LSL, PyTorch, Scikit-learn, TensorFlow, Antropy, NumPy, and SciPy into one platform simplifies the research process, encouraging innovation and collaboration in the field of brain computer/human machine interfaces.

# Acknowledgements

The io:bio mobile EEG device [@2021bateson_asghar] was used to create an initial port for streaming time-series physiological data in to the Lab Streaming Layer, so we could receive, analyse, record, and classify EMG, ECG and EEG data - enabling prior required experimentation to creating PyBCI.

The work carried out by Christian Kothe creating the Lab Streaming Layer and continuous maintenance to the pylsl repository by Chadwick Boulay enables unification across many off shelf devices. Chadwick Boulay also gave helpful recommendations in the GitHub issue: https://github.com/labstreaminglayer/pylsl/issues/70.

# References
