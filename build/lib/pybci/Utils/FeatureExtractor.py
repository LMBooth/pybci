import numpy as np
from scipy.signal import welch
from scipy.integrate import simps
import warnings, time
from ..Configuration.FeatureSettings import GeneralFeatureChoices
# Filter out UserWarning messages from the scipy package, could be worth moving to init and applying printdebug print levels? (typically nans, 0 and infs causing errors)
warnings.filterwarnings("ignore", category=UserWarning, module="scipy") # used to reduce print statements from constant signals being applied
warnings.filterwarnings("ignore", category=UserWarning, module="antropy") # used to reduce print statements from constant signals being applied
warnings.filterwarnings("ignore", category=RuntimeWarning, module="antropy") # used to reduce print statements from constant signals being applied
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy") # used to reduce print statements from constant signals being applied
#warnings.filterwarnings("ignore", category=RuntimeWarning, module="pybci") # used to reduce print statements from constant signals being applied

class GenericFeatureExtractor():
    def __init__(self, freqbands = [[1.0, 4.0], [4.0, 8.0], [8.0, 12.0], [12.0, 20.0]], featureChoices = GeneralFeatureChoices()):
        super().__init__()
        self.freqbands = freqbands
        self.featureChoices = featureChoices
        #for key, value in self.featureChoices.__dict__.items():
        #    print(f"{key} = {value}")
        selFeats = sum([self.featureChoices.appr_entropy,
            self.featureChoices.perm_entropy,
            self.featureChoices.spec_entropy,
            self.featureChoices.svd_entropy,
            self.featureChoices.samp_entropy,
            self.featureChoices.rms,
            self.featureChoices.meanPSD,
            self.featureChoices.medianPSD,
            self.featureChoices.variance,
            self.featureChoices.meanAbs,
            self.featureChoices.waveformLength,
            self.featureChoices.zeroCross,
            self.featureChoices.slopeSignChange]
        )
        if self.featureChoices.appr_entropy or self.featureChoices.perm_entropy or self.featureChoices.spec_entropy or self.featureChoices.svd_entropy or self.featureChoices.samp_entropy:
            import antropy as ant
        self.numFeatures = (len(self.freqbands)*self.featureChoices.psdBand)+selFeats

    def ProcessFeatures(self, epoch, sr, target):
        """Allows 2D time series data to be passed with given sample rate to get various time+frequency based features.
        Best for EEG, EMG, EOG, or other consistent data with a consistent sample rate (pupil labs does not)
        Which features are chosen is based on self.featureChoices with initialisation. self.freqbands sets the limits for
        desired frequency bands average power.
        Inputs:
            epoch = 2D list or 2D numpy array [chs, samples]
            target = string of received marker type
            sr = samplerate of current device
        Returns:
            features = 2D numpy array of size (chs, (len(freqbands) + sum(True in self.featureChoices)))
            target = same as input target, can be useful for using a baseline number differently
        NOTE: Any channels with a constant value will generate warnings in any frequency based features (constant level == no frequency components).
        """
        numchs = epoch.shape[1]
        features = np.zeros(numchs * self.numFeatures)

        for ch in range(epoch.shape[1]):
            #ch = np.isnan(ch)
            if self.featureChoices.psdBand: # get custom average power within given frequency band from freqbands
                freqs, psd = welch(epoch[:,ch], sr)
                for l, band in enumerate(self.freqbands):
                    if len(freqs) > 0: # len(freqs) can be 0 if signal is all DC
                        idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
                        #if len(psd[idx_band]) == 1: # if freq band is only in one field just pass single value instead of calculating average
                        #print(ch)
                        bp = np.mean(psd[idx_band])
                        #else:
                        #    bp = simps(psd[idx_band], dx=(freqs[1]-freqs[0])) / (band[1] - band[0])
                            #bp = simpson(psd[idx_band], dx=freq_res)
                        features[(ch* self.numFeatures)+l] = bp
                    else:
                        features[(ch* self.numFeatures)+l] = 0
            else:
                freqs, psd = welch(epoch[:,ch], sr)# calculate for mean and median
                l = -1 # accounts for no freqbands being selected
            if self.featureChoices.meanPSD: # mean power
                l += 1
                if len(freqs) > 0: features[(ch* self.numFeatures)+l] = np.mean(psd) # len(freqs) can be 0 if signal is all DC
                else: features[(ch* self.numFeatures)+l] = 0
            if self.featureChoices.medianPSD: # median Power
                l += 1   
                if len(freqs) > 0: features[(ch* self.numFeatures)+l] = np.median(psd) # len(freqs) can be 0 if signal is all DC
                else: features[(ch* self.numFeatures)+l] = 0
            if self.featureChoices.appr_entropy:  # Approximate entropy(X,M,R) X = data, M is , R is 30% standard deviation of X 
                l += 1
                features[(ch* self.numFeatures)+l] = ant.app_entropy(epoch[:,ch]) 
            if self.featureChoices.perm_entropy: # permutation_entropy
                l += 1
                features[(ch* self.numFeatures)+l] = ant.perm_entropy(epoch[:,ch],normalize=True)
            if self.featureChoices.spec_entropy:  # spectral Entropy
                l += 1
                features[(ch* self.numFeatures)+l] = ant.spectral_entropy(epoch[:,ch], sf=sr, method='welch', nperseg = len(epoch[:,ch]), normalize=True)
            if self.featureChoices.svd_entropy:# svd Entropy
                l += 1
                features[(ch* self.numFeatures)+l] = ant.svd_entropy(epoch[:,ch], normalize=True)
            if self.featureChoices.samp_entropy: # sample Entropy
                l += 1
                features[(ch* self.numFeatures)+l] = ant.sample_entropy(epoch[:,ch])
            if self.featureChoices.rms: # rms
                l += 1
                features[(ch* self.numFeatures)+l] = np.sqrt(np.mean(np.array(epoch[:,ch])**2))
            if self.featureChoices.variance: # variance
                l += 1    
                features[(ch* self.numFeatures)+l] =  np.var(epoch[:,ch])
            if self.featureChoices.meanAbs: # Mean Absolute Value 
                l += 1
                try:
                    features[(ch* self.numFeatures)+l] = sum([np.linalg.norm(c) for c in epoch[:,ch]])/len(epoch[:,ch])
                except:
                    features[(ch* self.numFeatures)+l] = 0
            if self.featureChoices.waveformLength: # waveformLength
                l += 1
                try:    
                    features[(ch* self.numFeatures)+l] = sum([np.linalg.norm(c-epoch[inum,ch]) for inum, c in enumerate(epoch[1:,ch])])
                except:
                    features[(ch* self.numFeatures)+l] = 0
            if self.featureChoices.zeroCross: # zeroCross
                l += 1
                features[(ch* self.numFeatures)+l] = sum([1 if c*epoch[inum+1,ch]<0 else 0 for inum, c in enumerate(epoch[:-1,ch])])
            if self.featureChoices.slopeSignChange: # slopeSignChange
                l += 1    
                ssc = sum([1 if (c-epoch[inum+1,ch])*(c-epoch[inum+1,ch])>=0.1 else 0 for inum, c in enumerate(epoch[:-1,ch])])
                features[(ch* self.numFeatures)+l] = ssc
        features[np.isnan(features)] = 0 # checks for nans
        features[features == np.inf] = 0#np.iinfo(np.int32).max
        #print(features)
        
        return features
    
class GazeFeatureExtractor():
    def __init__(self):
        super().__init__()

'''pupil channels in order
confidence: 1 channel
norm_pos_x/y: 2 channels
gaze_point_3d_x/y/z: 3 channels
eye_center0_3d_x/y/z (right/left, x/y/z): 6 channels (3 channels for each eye)
gaze_normal0/1_x/y/z (right/left, x/y/z): 6 channels (3 channels for each eye)
norm_pos_x/y: 2 channels
diameter0/1_2d (right/left): 2 channels
diameter0/1_3d (right/left): 2 channels
22 total
'''