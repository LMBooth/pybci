import antropy as ant
import numpy as np
from scipy.signal import welch
from scipy.integrate import simpson
import warnings
from Configuration.FeatureSettings import FeatureChoices
# Filter out UserWarning messages from the scipy package, could be worth moving to init and applying printdebug print levels?
warnings.filterwarnings("ignore", category=UserWarning, module="scipy") # used to reduce print statements from constant signals being applied
warnings.filterwarnings("ignore", category=UserWarning, module="antropy") # used to reduce print statements from constant signals being applied
warnings.filterwarnings("ignore", category=RuntimeWarning, module="antropy") # used to reduce print statements from constant signals being applied
#warnings.filterwarnings("ignore", category=RuntimeWarning, module="pybci") # used to reduce print statements from constant signals being applied

class FeatureExtractor():

    def __init__(self, freqbands = [[1.0, 4.0], [4.0, 8.0], [8.0, 12.0], [12.0, 20.0]], featureChoices = FeatureChoices()):
        super().__init__()
        self.freqbands = freqbands
        self.featureChoices = featureChoices
        for key, value in self.featureChoices.__dict__.items():
            print(f"{key} = {value}")
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
        self.featureChoices.slopeSignChange])
        self.numFeatures = (len(self.freqbands)*self.featureChoices.psdBand)+selFeats

    def ProcessPupilFeatures(self, epoch):
        pass

    def ProcessECGFeatures(self, epoch):
        pass

    def ProcessGeneralEpoch(self, epoch, sr):
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
            target = same as input target
        NOTE: Any channels with a constant value will generate warnings in any frequency based features (constant level == no frequency components).
        """
        features = np.zeros((len(epoch),self.numFeatures))
        #print(features.shape)
        for k, ch in enumerate(epoch):
            if self.featureChoices.psdBand: # get custom average power within given frequency band from freqbands
                for l, band in enumerate(self.freqbands):
                    #bp = bandpower(ch, sr, band, method='multitaper', window_sec=None, relative=False)  
                    nperseg = (2 / band[0]) * sr
                    freqs, psd = welch(ch, sr)#, nperseg=len(ch)-1)
                    freq_res = freqs[1] - freqs[0]
                    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
                    if len(psd[idx_band]) == 1: # if freq band is only in one field just pass single value instead of calculating average
                        bp = psd[idx_band]
                    else:
                        bp = simpson(psd[idx_band], dx=freq_res)
                    features[k][l] = bp
            else:
                l = -1 # accounts for no freqbands being selected
            if self.featureChoices.appr_entropy:  # Approximate entorpy(X,M,R) X = data, M is , R is 30% standard deviation of X 
                l += 1
                features[k][l] = ant.app_entropy(ch) # 
            if self.featureChoices.perm_entropy: # permutation_entropy
                l += 1
                features[k][l] = ant.perm_entropy(ch,normalize=True)
            if self.featureChoices.spec_entropy:  # spectral Entropy
                l += 1
                features[k][l] = ant.spectral_entropy(ch, sf=sr, method='welch', nperseg = len(ch), normalize=True)
            if self.featureChoices.svd_entropy:# svd Entropy
                l += 1
                features[k][l] = ant.svd_entropy(ch, normalize=True)
            if self.featureChoices.samp_entropy: # sample Entropy
                l += 1
                features[k][l] = ant.sample_entropy(ch)
            if self.featureChoices.rms: # rms
                l += 1
                features[k][l] = np.sqrt(np.mean(np.array(ch)**2))
            if self.featureChoices.meanPSD or self.featureChoices.medianPSD:
                freqs, psd = welch(ch, sr)
                #with warnings.catch_warnings():
                #    warnings.filterwarnings('error')
                #    try:
                if self.featureChoices.meanPSD: # mean power
                    l += 1
                    features[k][l] = sum(freqs*psd)/sum(psd)
                if self.featureChoices.medianPSD: # median Power
                    l += 1   
                    features[k][l] = sum(psd)/2
                #    except Warning as e:
                #        print(k)
                #        print(freqs)
                #        print(psd)
                #        print('error found:', e)

            if self.featureChoices.variance: # variance
                l += 1    
                features[k][l] =  np.var(ch)
            if self.featureChoices.meanAbs: # Mean Absolute Value 
                l += 1
                features[k][l] = sum([np.linalg.norm(c) for c in ch])/len(ch)
            if self.featureChoices.waveformLength: # waveformLength
                l += 1
                features[k][l] = sum([np.linalg.norm(c-ch[inum]) for inum, c in enumerate(ch[1:])])
            if self.featureChoices.zeroCross: # zeroCross
                l += 1
                features[k][l] = sum([1 if c*ch[inum+1]<0 else 0 for inum, c in enumerate(ch[:-1])])
            if self.featureChoices.slopeSignChange: # slopeSignChange
                l += 1    
                ssc = sum([1 if (c-ch[inum+1])*(c-ch[inum+1])>=0.1 else 0 for inum, c in enumerate(ch[:-1])])
                features[k][l] = ssc
        return features