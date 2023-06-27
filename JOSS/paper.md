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
   orcid: 0000-0002-4780-4458
   affiliation: "1"
 - name: Aziz Asghar
   orcid: 0000-0002-3735-4449
   affiliation: "2"
affiliations:
 - name: Faculty of Science and Engineering, University of Hull.
   index: 1
 - name: Centre for Anatomical and Human Sciences, Hull York Medical School, University of Hull.
   index: 2
date: 26 June 2023
bibliography: paper.bib
---

# Summary:

PyBCI is a comprehensive, open-source Python framework developed to facilitate brain-computer interface (BCI) research. It encompasses data acquisition, data labelling, feature extraction, and machine learning. PyBCI provides a streamlined, user-friendly platform for conducting real-time BCI applications. The software uses the Lab Streaming Layer (LSL) (Kothe, 2014) protocol for data acquisition on various LSL enabled data streams, where a (LSL) marker stream is used for labelling training data. PyTorch (Paszke et al., 2019), SciKit-learn (Pedregosa et al., 2011) and TensorFlow (Abadi et al., 2016) are leveraged for machine learning. NumPy (Oliphant, 2006), SciPy (Virtanen et al., 2020), and Antropy (Vallat, 2023) are utilised for generic feature extraction examples.

# Statement of Need:

PyBCI puts emphasis on quick and easy customisation of applied time-series feature extraction techniques and machine learning models, live data classification, and integration into other systems. Remaining lightweight with solely Python packages and no additional visualisation and recording tools, it's assumed a user will have these tools available with their LSL hardware. 

BCIs have the potential to revolutionise the way we interact with computers. However, BCI research is often limited by the lack of availability of open-source, comprehensive, and user-friendly software. PyBCI addresses this gap, offering a simple, flexible, and robust Python-based platform for conducting BCI research. It integrates seamlessly with the LSL for data acquisition and labelling, utilizing popular machine learning packages such as PyTorch, Scikit-learn and TensorFlow, employing Antropy, NumPy, and SciPy for an example feature extraction or optionally allowing the user to pass a custom feature extractor to process specific device data. This combination of software features allows researchers to focus on their experiments rather than software development, expediting the advancement of BCI technology.

# Software functionality and performance:

PyBCI provides an end-to-end solution for BCI research. It uses the Lab Streaming Layer (LSL) to handle data acquisition and labelling, allowing for real-time, synchronous data collection from multiple devices (Kothe, 2014). Samples are collected in chunks from the LSL data streams and stored in pre-allocated NumPy arrays. When in training mode based on a configurable time window before and after each marker type. When in test mode data is continuously processed and analysed based on the global epoch timing settings.  For feature extraction, PyBCI leverages the power of NumPy (Oliphant, 2006), SciPy (Virtanen et al., 2020), and Antropy (Vallat, 2023) robust Python libraries known for their efficiency in handling numerical operations. Machine learning, a crucial component of BCI research, is facilitated with PyTorch (Paszke et al., 2019), SciKit-learn (Pedregosa et al., 2011) and TensorFlow (Abadi et al., 2016). Scikit-learn offers a wide range of traditional algorithms for classification, including things like regression, and clustering , while TensorFlow and PyTorch provide comprehensive ecosystems for developing and training bespoke deep learning machine learning models (Abadi et al., 2016).

# Impact:

By offering a comprehensive, open-source platform for BCI research, PyBCI has the potential to drive the field forward. It provides researchers with the tools necessary to conduct advanced BCI experiments without requiring extensive software development skills. The integration of LSL, PyTorch, Scikit-learn, TensorFlow, Antropy, NumPy, and SciPy into one platform simplifies the research process, encouraging innovation and collaboration in the field of brain computer/human machine interfaces.


