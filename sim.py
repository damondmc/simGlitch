import lal
from lalpulsar import simulateCW
import time 
from tqdm import tqdm
import numpy as np 

def waveform(h0, cosi, freq, f1dot, f2dot, f3dot, f4dot, tglitch, df_permanent, df_tau, df1_permanent, df1_tau, tau):
    # Validate that all glitch-related parameters are lists of the same length
    if not all(len(lst) == len(tglitch) for lst in [df_permanent, df_tau, df1_permanent, df1_tau, tau]):
        raise ValueError("All glitch parameter lists must have the same length")
    
    def wf(dt):
        # Base phase calculation
        dphi = freq * dt + f1dot * 0.5 * dt**2 + f2dot * (1./6.) * dt**3 + \
               f3dot * (1./24.) * dt**4 + f4dot * (1./120.) * dt**5
        
        # Add contributions from each glitch
        for i in range(len(tglitch)):
            if dt > tglitch[i]:
                dphi += df_tau[i] * np.exp(-(dt - tglitch[i]) / tau[i]) * (dt - tglitch[i])
                dphi += df_permanent[i] * (dt - tglitch[i])
                dphi += df1_tau[i] * np.exp(-(dt - tglitch[i]) / tau[i]) * 0.5 * (dt - tglitch[i])**2
                dphi += df1_permanent[i] * 0.5 * (dt - tglitch[i])**2
        
        dphi = lal.TWOPI * dphi
        
        # Amplitude calculations remain unchanged
        ap = h0 * (1.0 + cosi**2) / 2.0
        ax = h0 * cosi
        
        return dphi, ap, ax
    return wf
 
    
def genSampleParam(f0min, f0max, f1min, f1max, nSample):
    """
    Generate parameters for a given number of random samples.

    Parameters:
    - f0min, f0max (float): Frequency range for parameter sampling.
    - f1min, f1max (float): Frequency derivative range for parameter sampling.
    - nSample (int): Number of random samples.

    Returns:
    - params (list): List of parameter combinations for signal generation.
    """
    print("Sample generation.")
    f0_arr = np.random.uniform(f0min, f0max, nSample)
    f1_arr = np.random.uniform(f1min, f1max, nSample)
    phi_arr = np.random.uniform(0, 2*np.pi, nSample)
    psi_arr = np.random.uniform(-np.pi/4, np.pi/4, nSample)
    cosi_arr = np.random.uniform(-1, 1, nSample)
    
    alpha_arr = np.random.uniform(0, 2*np.pi, nSample)
    sinDelta = np.random.uniform(-1 , 1, nSample)
    delta_arr = np.arcsin(sinDelta)
    
    params = np.column_stack((f0_arr, f1_arr, phi_arr, psi_arr, cosi_arr, alpha_arr, delta_arr))
    return params


tstart   = 1368970000
Tdata    = 400 * 86400
tref     = tstart + 0.5 * Tdata
h0       = 1e-24
cosi     = 1
psi      = 0
phi0     = 3.210
freq     = 20.0
f1dot    = -1.35e-9
f2dot    = 0
f3dot    = 0
f4dot    = 0

tglitch = [tstart + 0.2 * Tdata, tstart + 0.5 * Tdata]
tglitch = [t - tref for t in tglitch]  # Normalize each element of tglitch by tref
df_permanent = [5e-3, 5e-3]
df_tau = [5e-2, 5e-2]
df1_permanent = [-5e-11, -5e-11]
df1_tau = [-5e-11, -5e-11]
tau = [20*86400, 20*86400]

dt_wf    = 5
alpha    = 6.12
delta    = 1.02
detector = 'H1'
 
t0 = time.time()
wf = waveform(h0, cosi, freq, f1dot, f2dot, f3dot, f4dot, tglitch, df_permanent, df_tau, df1_permanent, df1_tau, tau)
S = simulateCW.CWSimulator(tref, tstart, Tdata, wf, dt_wf, phi0, psi, alpha, delta, detector)
 
# To write SFT files
for file, i, N in tqdm(S.write_sft_files(fmax=21, Tsft=1800, comment="simCW", out_dir='./sfts/'), total=Tdata / 86400 * 48):
    #print('Generated SFT file %s (%i of %i)' % (file, i+1, N))
    continue 
    
print("Done. Time used: {}s".format(time.time()-t0))