class GlobalEpochSettings:
    splitCheck = False  # checks whether or not subdivide epochs
    tmin = 0      # time in seconds to capture samples before trigger
    tmax = 1      # time in seconds to capture samples after trigger
    windowLength = 0.5    # if splitcheck true - time in seconds to split epoch
    windowOverlap = 0.5 #if splitcheck true  percentage value > 0 and < 1, example if epoch has tmin of 0 and tmax of 1 with window 
    # length of 0.5 we have 1 epoch between t 0 and t0.5 another at 0.25 to 0.75, 0.5 to 1

# customWindowSettings should be dict with marker name and IndividualEpochSetting
class IndividualEpochSetting:
    splitCheck = False  # checks whether or not subdivide epochs
    tmin = 0      # time in seconds to capture samples before trigger
    tmax=  1      # time in seconds to capture samples after trigger