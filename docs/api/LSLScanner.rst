pybci.LSLScanner
===

.. class:: LSLScanner(parent, dataStreamsNames=None, markerStreamName=None, streamTypes=None, markerTypes=None, printDebug=True)

The LSLScanner class scans and selects desired data and marker streams from available LSL streams.

:param parent: Parent object.
:param list dataStreamsNames: Allows user to set custom acceptable EEG stream definitions, if None defaults to streamTypes scan.
:param string markerStreamName: Allows user to set custom acceptable Marker stream definitions, if None defaults to markerTypes scan.
:param list streamTypes: Allows user to set custom acceptable EEG type definitions, ignored if dataStreamsNames not None.
:param list markerTypes: Allows user to set custom acceptable Marker type definitions, ignored if markerStreamName not None.
:param bool printDebug: If true, prints LSLScanner debug information.

.. py:method:: ScanStreams()
