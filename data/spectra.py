###########################################################
# Spectra.py
# The idea of this script is to use classes Spectra and
#  SingleSpectrum to perform operations with spectra and
#  convert MRS data from/to FID-A and Gannet.
#
# Right now we have:
#  - Reading ground-truths from csvs
#  - creating transients (with noise) from ground-truths
#  - add different types of noise with parameters (Freq & Phase shifts are saved)
#  - perform spectral registration
#
###########################################################


import numpy as np
import random
import math
import scipy
from scipy import optimize
import matplotlib.pyplot as plt




class Spectra:
    """
            Class Spectra - Holds MRS data for performing usual operations
        Class is constructed to hold data of many scans, subspectra and transients.
        FID/Spec are 4d numpy complex arrays - [scan # x spectrum point x edit on/off x transient #]
        Ex.: 40 scans with 2048 points, edit on and off with 160 transients each is a [40 x 2048 x 2 x 160] array
    """

    ## class variables non-initialized in __init__
    ground_truth_fids=None
    added_noise = {}

    ## object can be created with variables or add them later
    def __init__(self,fids=None,specs=None,t=None,ppm=None,
                 spectralwidth=None,dwelltime=None,
                 n=None,b=None,txfreq=None,te=None):
        self.fids=fids
        self.specs=specs
        self.t=t
        self.ppm=ppm
        self.spectralwidth=spectralwidth
        self.dwelltime=dwelltime
        self.n=n
        self.b=b
        self.txfreq=txfreq,
        self.te=te

        if fids is not None and specs is None:
            self._update_specs()

    
    def _update_specs(self):
        """
            Update specs variable to reflect FFT of FID data
            Parameters: None
            Return: Updates self.specs
        """
        self.specs = np.fft.fftshift(np.fft.ifft(self.fids,axis=1),axes=1)

    # Read individual files of simulated ground-truths
    def load_from_ground_truth_csvs(self,on_fids_file,off_fids_file,ppm_file,t_file):
        """
            Load ground truth data from FID-A outputted csv`s
            transforms data into appropriate format.
            ground truth is saved for later comparison.
            Parameters: File names
            Returns: Updates class variables
        """


        on_fids = np.loadtxt(on_fids_file,dtype=complex,delimiter=",").T
        off_fids = np.loadtxt(off_fids_file,dtype=complex,delimiter=",").T
        ppm = np.loadtxt(ppm_file,dtype=float,delimiter=",").T
        t = np.loadtxt(t_file,dtype=float,delimiter=",").T
        
        self.fids = np.zeros(shape=(t.shape[0],t.shape[1],2),dtype=complex)
        self.fids[:,:,0] = off_fids
        self.fids[:,:,1] = on_fids
        self.ppm = ppm
        self.t = t

        #saving ground truth data
        self.ground_truth_fids = self.fids.copy()
        self.ground_truth_spec = np.fft.fftshift(np.fft.ifft(self.ground_truth_fids,axis=1),axes=1)

        # adding dimension for transient
        self.fids = np.expand_dims(self.fids,axis=3)
        
        self._update_specs()


    # decrease the amount of data available - good for testing
    def select_scans(self,min,max=None):
        if max is None:
            max=min
        self.fids = self.fids[min:max+1]
        self.ppm = self.ppm[min:max+1]
        self.t = self.t[min:max+1]

        self.ground_truth_fids = self.ground_truth_fids[min:max+1]
        self.ground_truth_spec = self.ground_truth_spec[min:max+1]

        self._update_specs()
    
    # add dimension and repeat ground-truth, to enable addition of noise later
    def make_transients(self, transient_count=320):
        """
            expand ground-truth data into transients (noiseless)
            Parameter: Transient Count
            Returns: Updaters Fids and Specs
        """
        if transient_count%2!=0:
            raise Exception("Transient count must be multiple of 2")

        # in case its missing a dimension - shouldn't happen
        if len(self.fids.shape)<4:
            self.fids = np.expand_dims(self.fids,axis=3)

        # repeat ground-truth for all transients
        self.fids = np.repeat(self.fids,int(transient_count/2),axis=3)

        self._update_specs()

    # add amplitude normally generated amplitude noise to fids data
    def add_random_amplitude_noise(self,noise_level_base=10,noise_level_scan_var=3):
        """
            Add normal amplitude noise to time-domain data
            Parameters:
                - Noise level base: base Applied to all transients & scans
                - Noise level scan var: level of variation between different scans
            Returns: Updates Fids and Specs
        """
        base_noise = noise_level_base*np.ones(self.fids.shape[0])+np.random.uniform(low=-noise_level_scan_var,high=noise_level_scan_var,size=self.fids.shape[0])
        
        #adds real and imaginary noise
        noise_real = np.random.normal(0,base_noise.reshape(-1,1,1,1),size=self.fids.shape)
        noise_imag = 1j*np.random.normal(0,base_noise.reshape(-1,1,1,1),size=self.fids.shape)

        self.fids = self.fids + noise_real + noise_imag

        self._update_specs()
    
    def add_random_frequency_noise(self,noise_level_base=7,noise_level_scan_var=3):
        """
            Adds frequency shift to scans according to normal distribution
            Parameters:
                - Noise level base: base applied to all transients and scans
                - Noise level scan var: level of variation between different scans
            Return: Updates Fids and Specs
        """
        base_noise = noise_level_base*np.ones(self.fids.shape[0])+np.random.uniform(low=-noise_level_scan_var,high=noise_level_scan_var,size=self.fids.shape[0])
        
        #noise = np.random.normal(0,base_noise.reshape(-1,1,1,1),size=(self.fids.shape[0],1,self.fids.shape[2],self.fids.shape[3]))
        noise = np.random.uniform(-base_noise.reshape(-1,1,1,1),base_noise.reshape(-1,1,1,1),size=(self.fids.shape[0],1,self.fids.shape[2],self.fids.shape[3]))

        self.fids = self.fids*np.exp(1j*self.t.reshape(self.t.shape[0],self.t.shape[1],1,1)*noise*2*math.pi)

        self._update_specs()

        # save values for Frequency and Phase Correction y-data
        if "frequency_drift" not in self.added_noise:
            self.added_noise["frequency_drift"] = noise
        else:
            self.added_noise["frequency_drift"] += noise

    def add_linear_frequency_noise(self,offset_var=2,slope_var=4):
        """
            Adds frequency shift to scans with linear tendency between transients
            Offset and var are sampled from normal distribution
            Parameters:
                - offset_var: standard deviation of offset sampling
                - slope_var: slope of linear tendency 
            Return: Updates Fids and Specs
        """
        #base_offsets = np.random.normal(0,offset_var,size=(self.fids.shape[0]))
        #base_slopes = np.random.normal(0,slope_var,size=(self.fids.shape[0]))
        base_offsets = np.random.uniform(-offset_var,offset_var,size=(self.fids.shape[0]))
        base_slopes = np.random.uniform(-slope_var,slope_var,size=(self.fids.shape[0]))

        seq_array = np.arange(0,self.fids.shape[2]*self.fids.shape[3]).reshape(-1,self.fids.shape[2]).T.reshape(1,1,self.fids.shape[2],-1)

        noise = base_offsets.reshape(-1,1,1,1) + base_slopes.reshape(-1,1,1,1)*seq_array/(self.fids.shape[2]*self.fids.shape[3])

        self.fids = self.fids*np.exp(1j*self.t.reshape(self.t.shape[0],self.t.shape[1],1,1)*noise*2*math.pi)

        self._update_specs()

        # save values for Frequency and Phase Correction y-data
        if "frequency_drift" not in self.added_noise:
            self.added_noise["frequency_drift"] = noise
        else:
            self.added_noise["frequency_drift"] += noise

    def add_random_phase_noise(self,noise_level_base=5,noise_level_scan_var=3):
        """
            Adds phase shift to scans according to normal distribution
            Parameters:
                - Noise level base: base applied to all transients and scans
                - Noise level scan var: level of variation between different scans
            Return: Updates Fids and Specs
        """
        base_noise = noise_level_base*np.ones(self.fids.shape[0])+np.random.uniform(low=-noise_level_scan_var,high=noise_level_scan_var,size=self.fids.shape[0])
        
        #noise = np.random.normal(0,base_noise.reshape(-1,1,1,1),size=(self.fids.shape[0],1,self.fids.shape[2],self.fids.shape[3]))
        noise = np.random.uniform(-base_noise.reshape(-1,1,1,1),base_noise.reshape(-1,1,1,1),size=(self.fids.shape[0],1,self.fids.shape[2],self.fids.shape[3]))

        self.fids = self.fids*np.exp(1j*noise*math.pi/180)

        self._update_specs()

        # save values for Frequency and Phase Correction y-data
        if "phase_drift" not in self.added_noise:
            self.added_noise["phase_drift"] = noise
        else:
            self.added_noise["phase_drift"] += noise

    def apply_phase_and_frequency_correction(self):
        """
            Apply spectral registration (SR) as a reference FPC method 
            -- Implemented version might not be the best possible, maybe improve later?
            -- SR applied at each scan subspectra, using SingleSpectrum class
            Parameters: None
            Returns: Fids and Specs updated, freq and phase corrections also saved.
        """
        
        # initialize correction matrixes
        phase_correction = np.zeros((self.fids.shape[0],1,self.fids.shape[2],self.fids.shape[3]))
        freq_correction = np.zeros((self.fids.shape[0],1,self.fids.shape[2],self.fids.shape[3]))

        # iterate through scans
        for i in range(self.fids.shape[0]):
            # iterate through on and off
            for j in range(self.fids.shape[2]):
                print(f"i:{i} - j:{j}")
                # create single spectrum object and perform spec registration within it.
                single_spec = SingleSpectrum(fids=self.fids[i,:,j,:].reshape(self.fids.shape[1],self.fids.shape[3]),
                                             t=self.t[i],ppm=self.ppm[i])
                
                single_spec.apply_phasefreq_correction()
                phase_correction[i,0,j,:]=single_spec.fpc_results["phase"]
                freq_correction[i,0,j,:]=single_spec.fpc_results["frequency"]
        
        # apply corrections
        self.fids = self.fids*np.exp(1j*self.t.reshape(self.t.shape[0],self.t.shape[1],1,1)*freq_correction*2*math.pi)
        self.fids = self.fids*np.exp(1j*phase_correction*math.pi/180)

        self.phase_correction = phase_correction
        self.freq_correction = freq_correction


# Adding comments for this class later
class SingleSpectrum:

    def __init__(self,fids=None,specs=None,t=None,ppm=None,
                 spectralwidth=None,dwelltime=None,
                 n=None,b=None,txfreq=None,te=None):
        self.fids=fids
        self.specs=specs
        self.t=t
        self.ppm=ppm
        self.spectralwidth=spectralwidth
        self.dwelltime=dwelltime
        self.n=n
        self.b=b
        self.txfreq=txfreq,
        self.te=te

        if fids is not None and specs is None:
            self._update_specs()

    def add_phasefreq(self,freq=0,phase=0):
        new_fids = self.fids*np.exp(1j*self.t.reshape(-1,1)*freq*2*math.pi)
        new_fids = new_fids*np.ones(new_fids.shape)*np.exp(1j*phase*math.pi/180)
        self.fids=new_fids

    def _update_specs(self):
        self.specs = np.fft.fftshift(np.fft.ifft(self.fids,axis=0),axes=0)

    def apply_phasefreq_correction(self,mode='spectral_registration'):
        if mode=='spectral_registration':
            self._apply_spec_registration()
        else:
            raise Exception("only spectral registration implemented")
    
    
    def _apply_spec_registration(self):

        iter=0
        max_iter=20

        cum_fs=np.zeros(shape=(1,self.fids.shape[1]))
        cum_phs=np.zeros(shape=(1,self.fids.shape[1]))

        while iter<max_iter:

            ppmmin = 1.6 + 0.1*random.random()
            ppmmaxarray = list(np.concatenate([3.5+0.1*np.random.random(size=2),4+0.1*np.random.random(size=3),5.5+0.1*np.random.random(size=1)]))
            ppmmax=ppmmaxarray[random.randint(0,5)]

            base = self.output_avg_spectrum()
            base = base.output_freqrange(ppmmin,ppmmax)

            transients_freqrange = self.output_freqrange(ppmmin,ppmmax)
            y = np.concatenate([np.real(base.fids.flatten()),np.imag(base.fids.flatten())])

            fs=np.zeros(shape=(1,self.fids.shape[1]))
            phs=np.zeros(shape=(1,self.fids.shape[1]))

            for i in range(0,transients_freqrange.fids.shape[1]):
                x = np.concatenate([np.real(transients_freqrange.fids[:,i].flatten()),np.imag(transients_freqrange.fids[:,i].flatten()),transients_freqrange.t])
                try:
                    parFit,_ = scipy.optimize.curve_fit(op_freqPhaseShiftComplexRangeNest,x,y,[0,0],maxfev=5000)
                except:
                    #print(f"failed spec registration - iter: {iter} - transient: {i+1}")
                    parFit=[0,0]
                
                fs[0,i]=parFit[0]
                phs[0,i]=parFit[1]
                #print(fs)

            self.add_phasefreq(fs,phs)
            self._update_specs()

            cum_fs = np.concatenate([cum_fs,fs],axis=0)
            cum_phs = np.concatenate([cum_phs,phs],axis=0)         

            iter+=1
        
        self.fpc_results={
            "frequency":cum_fs.sum(axis=0),
            "phase":cum_phs.sum(axis=0)
        }

    # simple function to make specs match FFT of fids
    def _update_specs(self):
        self.specs = np.fft.fftshift(np.fft.ifft(self.fids,axis=0),axes=0)

    def output_avg_spectrum(self):
        out_spec = SingleSpectrum(
            fids=self.fids.mean(axis=1).reshape(-1,1),
            t=self.t,
            ppm=self.ppm,
            spectralwidth=self.spectralwidth,
            dwelltime=self.dwelltime,
            n=self.n,
            b=self.b,
            txfreq=self.txfreq,
            te=self.te
        )
        out_spec._update_specs()
        return out_spec

    def output_single_spectrum(self,i):
        out_spec = SingleSpectrum(
            fids=self.fids[:,i],
            t=self.t,
            ppm=self.ppm,
            spectralwidth=self.spectralwidth,
            dwelltime=self.dwelltime,
            n=self.n,
            b=self.b,
            txfreq=self.txfreq,
            te=self.te
        )
        out_spec._update_specs()
        return out_spec
    
    def output_freqrange(self,ppmmin,ppmmax):
        
        
        start_ind = np.argwhere(self.ppm<=ppmmax).min()
        end_ind = np.argwhere(self.ppm>=ppmmin).max()
        end_ind = end_ind+(start_ind-end_ind)%2
        
        check=False
        while check==False:
            ppm = self.ppm[start_ind:end_ind]
            specs = self.specs[start_ind:end_ind,:]
            fids = np.fft.fft(np.fft.fftshift(specs,axes=0),axis=0)
            dppm = ppm[1]-ppm[0]
            ppmrange = abs(ppm[-1]-ppm[0])+dppm
            spectralwidth = ppmrange*3*42.577
            dwell_time = 1/spectralwidth
            t = np.arange(0,dwell_time*(ppm.shape[0]),dwell_time)

            if t.shape==ppm.shape:
                check=True
            else:
                end_ind+=2

        out_spec = SingleSpectrum(
            fids=fids,
            specs=specs,
            t=t,
            ppm=ppm,
            spectralwidth=spectralwidth,
            dwelltime=dwell_time,
            n=t.shape[0],
            b=self.b,
            txfreq=self.txfreq,
            te=self.te
        )
        return out_spec

# in this function we must convert the complex array into a [real,complex] concat array-1d
def op_freqPhaseShiftComplexRangeNest(input,f,p):
    fid=input[0:int(input.shape[0]/3)]+input[int(input.shape[0]/3):int(input.shape[0]*2/3)]*1j
    t = input[int(input.shape[0]*2/3):]
    shifted=fid*np.exp(1j*t*f*2*math.pi)
    shifted=shifted*np.ones(shifted.shape)*np.exp(1j*p*math.pi/180)
    y=np.concatenate([np.real(shifted),np.imag(shifted)])
    return y



'''
spec = Spectra()
prefix="fpc_sim_data/"


spec.load_from_ground_truth_csvs(f"{prefix}fidsON_July2022.csv",f"{prefix}fidsOFF_July2022.csv",f"{prefix}ppm_July2022.csv",f"{prefix}t_July2022.csv")

spec.select_scans(0,3)

spec.make_transients(transient_count=320)

#spec.add_random_phase_noise()

#spec.add_linear_frequency_noise()

spec.add_random_frequency_noise()
print(spec.added_noise["frequency_drift"].shape)

spec.add_linear_frequency_noise()
print(spec.added_noise["frequency_drift"].shape)

#spec.add_random_amplitude_noise()

#print(spec.fids.shape)
'''



