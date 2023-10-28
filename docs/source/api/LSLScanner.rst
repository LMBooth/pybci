pybci.Utils.LSLScanner import LSLScanner
===============
.. class:: LSLScanner(parent, dataStreamsNames=None, markerStreamName=None, streamTypes=None, markerTypes=None, printDebug=True)

The LSLScanner class scans and selects desired data and marker streams from available LSL streams.

:parent: class: Parent object.
:dataStreamsNames: list(str): Allows user to set custom acceptable EEG stream definitions, if None defaults to streamTypes scan.
:markerStreamName: string: Allows user to set custom acceptable Marker stream definitions, if None defaults to markerTypes scan.
:streamTypes: list(str): Allows user to set custom acceptable EEG type definitions, ignored if dataStreamsNames not None.
:markerTypes: list(str):  markerTypes: Allows user to set custom acceptable Marker type definitions, ignored if markerStreamName not None.
:printDebug: bool: If true, prints LSLScanner debug information.

.. py:method:: ScanStreams()

  Scans LSL for both data and marker channels.

.. py:method:: ScanDataStreams()

  Scans available LSL streams and appends inlet to self.dataStreams

.. py:method:: ScanMarkerStreams()

  Scans available LSL streams and appends inlet to self.markerStreams


.. py:method:: CheckAvailableLSL()

  Checks streaminlets available, prints if printDebug

  :returns: `True` if 1 marker stream present and available datastreams are present. False if no datastreams are present and/or more or less than one marker stream is present.
