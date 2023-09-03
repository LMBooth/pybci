
class PseudoDataConfig:
    duration = 1.0 
    noise_level = 1
    amplitude = 2
    frequency = 3

class PseudoMarkerConfig:
    markerName = "PyBCIPseudoMarkers"
    markerType = "Markers"
    baselineMarkerString = "baseline"
    repeat = True
    autoplay = True
    num_baseline_markers = 10
    number_marker_iterations = 10
    seconds_between_markers = 5
    seconds_between_baseline_marker = 10
    baselineConfig = PseudoDataConfig()