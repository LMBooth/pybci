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
 - name: Faculty of Science and Engineering, University of Hull, United Kingdom.
   index: 1
 - name: Centre for Anatomical and Human Sciences, Hull York Medical School, University of Hull, United Kingdom.
   index: 2
date: 26 June 2023
bibliography: paper.bib
---

# Summary:
PyBCI is an open-source Python framework designed to streamline brain-computer interface (BCI) research. It offers a comprehensive platform for real-time data acquisition, labeling, classification and analysis. PyBCI is compatible with a wide range of time-series hardware and software data sources, thanks to its integration with the Lab Streaming Layer (LSL) protocol [@lsl].

# Statement of Need:

BCI research brings together diverse fields like neuroscience, engineering, and data science, requiring specialized tools for data acquisition, feature extraction, and real-time analysis. Existing solutions may offer partial functionalities or be cumbersome to use, slowing down the pace of innovation. PyBCI addresses these challenges by providing a flexible, Python-based platform aimed at researchers and developers in the BCI domain. Assuming a foundational understanding of Python, the software serves as a comprehensive solution for both academic and industry professionals.

Designed to be lightweight and user-friendly, PyBCI emphasizes quick customization and integrates seamlessly with the Lab Streaming Layer (LSL) for data acquisition and labeling [@lsl]. The platform incorporates reputable machine learning libraries like PyTorch [@NEURIPS2019_9015], TensorFlow [@tensorflow2015-whitepaper], and Scikit-learn [@scikit-learn], as well as feature extraction tools such as Antropy [@vallat_antropy_2023], NumPy [@oliphant2006guide], and SciPy [@2020SciPy-NMeth]. This integration allows users to focus more on their research and less on software development. While a detailed comparison with other software solutions will follow in the 'State of the Field' section, PyBCI sets itself apart through its emphasis on ease of use and technological integration.

# State of the Field:

There are a variety of BCI software packages available, each with its own advantages and limitations. Notable packages include solutions like OpenViBE [@OpenViBE] and BCI2000 [@BCI2000] that offer ease of use for those without programming expertise. BciPy [@BciPy], another Python-based platform, provides some level of customization but does not allow for the easy integration of popular machine learning libraries. In contrast, PyBCI offers seamless integration with a variety of machine learning libraries and feature extraction tools. This flexibility makes PyBCI a robust choice for researchers seeking a tailored, code-based approach to their BCI experiments.

# Software functionality and performance:

PyBCI accelerates the pace of BCI research by streamlining data collection, processing, and model analysis. It uses the Lab Streaming Layer (LSL) to handle data acquisition and labelling, allowing for real-time, synchronous data collection from multiple devices [@lsl]. Samples are collected in chunks from the LSL data streams and stored in pre-allocated NumPy arrays. When in training mode based on a configurable time window before and after each marker type. When in test mode, data is continuously processed and analysed based on the global epoch timing settings.  For feature extraction, PyBCI leverages the power of NumPy [@oliphant2006guide], SciPy [@2020SciPy-NMeth], and Antropy [@vallat_antropy_2023], robust Python libraries known for their efficiency in handling numerical operations. Machine learning, a crucial component of BCI research, is facilitated with PyTorch [@NEURIPS2019_9015], SciKit-learn [@scikit-learn] and TensorFlow [@tensorflow2015-whitepaper]. Scikit-learn offers a wide range of traditional algorithms for classification, including things like regression, and clustering, while TensorFlow and PyTorch provide comprehensive ecosystems for developing and training bespoke deep learning machine learning models.

# Impact:

By providing a comprehensive, open-source platform for BCI research, PyBCI aims to advance the field. When integrated with off-the-shelf devices that are LSL-enabled, as well as with pre-built LSL data viewers and marker delivery systems, PyBCI facilitates the efficient design, testing, and implementation of advanced BCI experiments.The integration of LSL, PyTorch, Scikit-learn, TensorFlow, Antropy, NumPy, and SciPy into one platform simplifies the research process, encouraging innovation and collaboration in the field of brain computer/human machine interfaces.

# Acknowledgements

The io:bio mobile EEG device [@2021bateson_asghar] was used to create an initial port for streaming time-series physiological data in to the Lab Streaming Layer, so we could receive, analyse, record, and classify EMG, ECG and EEG data - enabling prior required experimentation to creating PyBCI.

The work carried out by Christian Kothe creating the Lab Streaming Layer and continuous maintenance to the pylsl repository by Chadwick Boulay enables unification across many off shelf devices. Chadwick Boulay also gave helpful recommendations in the GitHub issue: https://github.com/labstreaminglayer/pylsl/issues/70.

# References
