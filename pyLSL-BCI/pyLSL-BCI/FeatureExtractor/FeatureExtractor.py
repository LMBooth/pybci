import antropy as ant
import numpy as np
from scipy.signal import welch
from scipy.integrate import simpson

class UniversalFeatureProcessor():
    freqbands = [[1.0, 4.0], [4.0, 8.0], [8.0, 12.0], [12.0, 20.0]] # delta (1–3 Hz), theta (4–7 Hz), alpha (8–12 Hz), beta (13–30 Hz)
    featureChoices = [True, True, True,True, True, True, True,True, True, True, True,True, True, True] # first is freqbands mean power

    def __init__(self, freqbands = None, featureChoices = None):
        super().__init__()
        self.featureChoices = featureChoices
        self.freqbands = freqbands

    def ProcessEpoch(self, epoch, target, sr):
        """Allows 2D time series data to be passed with given sample rate to get various time+frequency based features.
        Which features are chosen is based on self.featureChoices with initialisation. self.freqbands sets the limits for
        desired frequency bands average power.
        Inputs:
            epoch = 2D list or 2D numpy array [chs, samples]
            target = string of received marker type
            sr = samplerate of current device
        Returns:
            features = 2D numpy array of size (chs, (len(freqbands) + sum(True in self.featureChoices)))
            target = same as input target
        """
        features = np.zeros((len(epoch),(len(self.freqbands)*self.featureChoices[0])+sum(self.featureChoices[1:])))
        for k, ch in enumerate(epoch):
            if self.featureChoices[0]: # get custom average power within given frequency band from freqbands
                for l, band in enumerate(self.freqbands):
                    #bp = bandpower(ch, sr, band, method='multitaper', window_sec=None, relative=False)  
                    nperseg = (2 / band[0]) * sr
                    freqs, psd = welch(ch, sr, nperseg=nperseg)
                    freq_res = freqs[1] - freqs[0]
                    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
                    if len(psd[idx_band]) == 1: # if freq band is only in one field just pass single value instead of calculating average
                        bp = psd[idx_band]
                    else:
                        bp = simpson(psd[idx_band], dx=freq_res)
                    features[k][l] = bp
                    l += 1
            else:
                l = -1 # accounts for no freqbands being selected
            if self.featureChoices[1]:  # Approximate entorpy(X,M,R) X = data, M is , R is 30% standard deviation of X 
                l += 1
                features[k][l] = ant.app_entropy(ch) # 
            if self.featureChoices[2]: # permutation_entropy
                l += 1
                features[k][l] = ant.perm_entropy(ch,normalize=True)
            if self.featureChoices[3]:  # spectral Entropy
                l += 1
                features[k][l] = ant.spectral_entropy(ch, sf=sr, method='welch', normalize=True)
            if self.featureChoices[4]:# svd Entropy
                l += 1
                features[k][l] = ant.svd_entropy(ch, normalize=True)
            if self.featureChoices[5]: # sample Entropy
                l += 1
                features[k][l] = ant.sample_entropy(ch)
            if self.featureChoices[6]: # rms
                l += 1
                features[k][l] = np.sqrt(np.mean(ch**2))
            if self.featureChoices[7] or self.featureChoices[8]:
                freqs, psd = welch(ch, sr)#, nperseg=nperseg)
                if self.featureChoices[7]: # mean power
                    l += 1
                    features[k][l] = sum(freqs*psd)/sum(psd)
                if self.featureChoices[8]: # median Power
                    l += 1   
                    features[k][l] = sum(psd)/2
            if self.featureChoices[9]: # variance
                l += 1    
                features[k][l] =  np.var(ch)
            if self.featureChoices[10]: # Mean Absolute Value 
                l += 1
                features[k][l] = sum([np.linalg.norm(c) for c in ch])/len(ch)
            if self.featureChoices[11]: # waveformLength
                l += 1
                features[k][l] = sum([np.linalg.norm(c-ch[inum]) for inum, c in enumerate(ch[1:])])
            if self.featureChoices[12]: # zeroCross
                l += 1
                features[k][l] = sum([1 if c*ch[inum+1]<0 else 0 for inum, c in enumerate(ch[:-1])])
            if self.featureChoices[13]: # slopeSignChange
                l += 1    
                ssc = sum([1 if (c-ch[inum+1])*(c-ch[inum+1])>=0.1 else 0 for inum, c in enumerate(ch[:-1])])
                features[k][l] = ssc
        return features,target