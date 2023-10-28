pybci.Configuration.EpochSettings import GlobalEpochSettings
===============
.. class:: GlobalEpochSettings()

GlobalEpochSettings class holds global time related variables for slicing epochs and capturing samples so many seconds before and after markers, this will define rolling window in testmode too.

:splitCheck: bool: Checks whether or not subdivide epochs.
:tmin: float: Time in seconds to capture samples before trigger.
:tmax: float: Time in seconds to capture samples after trigger.
:windowLength: float: If splitcheck true - time in seconds to split epoch.
:windowOverlap: float: If splitcheck true - percentage value > 0 and < 1, example; if epoch has tmin of 0 and tmax of 1 with window length of 0.5 we have 1 epoch between t 0 and t0.5 another at 0.25 to 0.75, 0.5 to 1

pybci.Configuration.EpochSettings import IndividualEpochSetting
===============
.. class:: IndividualEpochSetting()

IndividualEpochSetting class holds time related variables for slicing epoch markers with differing time windows to the global window settings, will be sliced and overlapped to create windows in shape of GlobalEpochSettings.windowLength.

:splitCheck: bool: Checks whether or not subdivide epochs.
:tmin: float: Time in seconds to capture samples before trigger.
:tmax: float: Time in seconds to capture samples after trigger.

pybci.Configuration.FeatureSettings import GeneralFeatureChoices
===============
.. class:: GeneralFeatureChoices()

GeneralFeatureChoices class holds booleans for quickly setting the generic feature class extractor.

:psdBand: bool: default = True, Checks whether or not psdBand features are desired.
:appr_entropy: bool: default = False, Checks whether or not appr_entropy features are desired.
:perm_entropy: bool: default = False, Checks whether or not perm_entropy features are desired.
:spec_entropy: bool: default = False, Checks whether or not spec_entropy features are desired.
:svd_entropy: bool: default = False, Checks whether or not svd_entropy features are desired.
:samp_entropy: bool: default = False, Checks whether or not samp_entropy features are desired.
:rms: bool: default = True, Checks whether or not rms features are desired.
:meanPSD: bool: default = True, Checks whether or not meanPSD features are desired.
:medianPSD: bool: default = True, Checks whether or not medianPSD features are desired.
:variance: bool: default = True, Checks whether or not variance features are desired.
:meanAbs: bool: default = True, Checks whether or not meanAbs features are desired.
:waveformLength: bool: default = False, Checks whether or not waveformLength features are desired.
:zeroCross: bool: default = False, Checks whether or not zeroCross features are desired.
:slopeSignChange: bool: default = False, Checks whether or not slopeSignChange features are desired.
