import numpy as np
import lal
from lalpulsar import simulateCW
from tqdm import tqdm
import time
import multiprocessing as mp
import os
from utils import gen_amplitude_params, gen_sky_location_params, gen_frequency_params, gen_glitch_params


import numpy as np
import matplotlib.pyplot as plt
import lal

# Assume gen_glitch_params and waveform are defined as in the latest artifact
def gen_glitch_params(n, tstart, Tdata, freq, f1dot, m_glitches=2, 
                      delta_f_p_range=(1e-9, 1e-6), delta_f_t_range=(1e-9, 1e-6), 
                      delta_f1dot_p_range=(-1e-4, -1e-3), delta_f1dot_t_range=(-1e-14, -1e-13), 
                      tau_range=(10*86400, 200*86400), Q=0.5):
    """
    Generate glitch parameters for n pulsars, each with exactly m_glitches, using Crab/Vela as a guide for ranges.
    Based on Yim & Jones (2020, arXiv:2007.05893).
    """
    tglitch = []
    delta_f_p = []
    delta_f_t = []
    delta_f1dot_p = []
    delta_f1dot_t = []
    tau = []
    
    for _ in range(n):
        m = m_glitches
        tglitch.append(np.random.uniform(tstart, tstart + Tdata, m))
        df_p = np.random.uniform(delta_f_p_range[0] * freq, 
                                delta_f_p_range[1] * freq, m)
        df_t = np.random.uniform(delta_f_t_range[0] * freq, 
                                delta_f_t_range[1] * freq, m)
        df_t = Q * (df_p + df_t) / (1 - Q)
        delta_f_t.append(df_t)
        df1dot_p = np.random.uniform(delta_f1dot_p_range[0] * f1dot, 
                                    delta_f1dot_p_range[1] * f1dot, m)
        tau_vals = np.random.uniform(tau_range[0], tau_range[1], m)
        df1dot_t = -df_t / tau_vals
        delta_f1dot_t.append(df1dot_t)
        delta_f1dot_p.append(df1dot_p)
        tau.append(tau_vals)
    
    return tglitch, delta_f_p, delta_f_t, delta_f1dot_p, delta_f1dot_t, tau


def waveform(h0, cosi, freq, f1dot, f2dot, f3dot, f4dot, tglitch, delta_f_p, delta_f_t, 
             delta_f1dot_p, delta_f1dot_t, tau):
    """
    Generate GW waveform for a pulsar with glitches, incorporating transient mountains.
    Based on Yim & Jones (2020, arXiv:2007.05893).
    """
    if not all(len(lst) == len(tglitch) for lst in [delta_f_p, delta_f_t, delta_f1dot_p, delta_f1dot_t, tau]):
        raise ValueError("All glitch parameter lists must have the same length")
    
    def wf(dt):
        dphi = freq * dt + f1dot * 0.5 * dt**2 + f2dot * (1./6.) * dt**3 + \
               f3dot * (1./24.) * dt**4 + f4dot * (1./120.) * dt**5
        h0_t = h0
        f0 = freq
        for i in range(len(tglitch)):
            if dt > tglitch[i]:
                delta_t = dt - tglitch[i]
                dphi += delta_f_p[i] * delta_t
                dphi += delta_f_t[i] * np.exp(-delta_t / tau[i]) * delta_t
                dphi += delta_f1dot_p[i] * 0.5 * delta_t**2
                dphi += delta_f1dot_t[i] * np.exp(-delta_t / tau[i]) * 0.5 * delta_t**2
                
                
                tp_ratio = np.sqrt(  (f0/(f0+delt_f_p+delta_f_t))**5 * ((f1+delta_f1_p+delta_f1dot_t)/f1)) # reccale according to permanent change
                
                p_ratio = np.sqrt(  (f0/(f0+delt_f_p))**5 * ((f1+delta_f1_p)/f1)) # reccale according to permanent change
                t_ratio = tp_ratio - p_ratio
                
                # include the expotential change
                h0_t *= (  p_ratio + t_ratio * np.exp(-delta_t/(2*tau[i])) )
                                
        dphi = lal.TWOPI * dphi
        ap = h0_t * (1.0 + cosi**2) / 2.0
        ax = h0_t * cosi
        return dphi, ap, ax
    return wf


def simulate_signal(params, h0, tstart, Tdata, dt_wf, detector, fmax, Tsft, out_dir, signal_idx):
    """
    Simulate a single continuous wave signal and generate SFT files.

    Parameters:
    - params (tuple): Signal parameters (freq_params, phi0, psi, cosi, alpha, delta, glitch_params).
    - h0 (float): Strain amplitude.
    - tstart (float): Start time of observation (GPS seconds).
    - Tdata (float): Duration of observation (seconds).
    - dt_wf (float): Time step for waveform sampling (seconds).
    - detector (str): Detector name (e.g., 'H1').
    - fmax (float): Maximum frequency for SFTs.
    - Tsft (float): SFT duration (seconds).
    - out_dir (str): Output directory for SFT files.
    - signal_idx (int): Index of the signal for unique output naming.

    Returns:
    - None: Generates SFT files in out_dir/signal_idx/.
    """
    tref = tstart + 0.5 * Tdata
    freq_params, phi0, psi, cosi, alpha, delta, glitch_params = params
    freq, f1dot, f2dot, f3dot, f4dot = freq_params
    tglitch, df_permanent, df_tau, df1_permanent, df1_tau, tau = glitch_params
    
    # Normalize tglitch relative to tref
    tglitch_norm = [t - tref for t in tglitch]
    
    # Create waveform
    wf = waveform(h0, cosi, freq, f1dot, f2dot, f3dot, f4dot, 
                  tglitch_norm, df_permanent, df_tau, df1_permanent, df1_tau, tau)
    
    # Create output directory for this signal
    signal_out_dir = os.path.join(out_dir, f"simCW_{signal_idx}")
    os.makedirs(signal_out_dir, exist_ok=True)
    
    # Simulate signal
    S = simulateCW.CWSimulator(tref, tstart, Tdata, wf, dt_wf, phi0, psi, alpha, delta, detector)
    
    # Write SFT files
    for file, j, N in S.write_sft_files(fmax=fmax, Tsft=Tsft, comment=f"simCW_{signal_idx}", out_dir=signal_out_dir):
        pass

def main(n, m, h0, tstart, Tdata, dt_wf, detector, fmax, Tsft, out_dir='./sfts/', 
         freq_ranges=[(10, 30), (-1e-8, 1e-8)], freq_order=1, 
         glitch_params_ranges=None):
    """
    Generate parameters for n signals with m glitches each, save them, and simulate signals in parallel.

    Parameters:
    - n (int): Number of signals to simulate.
    - m (int): Number of glitches per signal.
    - h0 (float): Strain amplitude.
    - tstart (float): Start time of observation (GPS seconds).
    - Tdata (float): Duration of observation (seconds).
    - dt_wf (float): Time step for waveform sampling (seconds).
    - detector (str): Detector name (e.g., 'H1').
    - fmax (float): Maximum frequency for SFTs.
    - Tsft (float): SFT duration (seconds).
    - out_dir (str): Base output directory for SFT files.
    - freq_ranges (list): List of (min, max) tuples for frequency derivatives.
    - freq_order (int): Order of frequency derivatives (0 to 4).
    - glitch_params_ranges (dict): Ranges for glitch parameters.

    Returns:
    - None: Generates SFT files and saves parameters to params.npz.
    """
    
    t0 = time.time()
    
    # Generate all parameters
    freq_params = gen_frequency_params(n, freq_order, freq_ranges)
    amp_params = gen_amplitude_params(n)
    sky_params = gen_sky_location_params(n)
    
    # Generate glitch parameters with fixed m glitches
    default_glitch_ranges = {
        'df_permanent': (5e-3, 5e-3),
        'df_tau': (5e-2, 5e-2),
        'df1_permanent': (-5e-11, -5e-11),
        'df1_tau': (-5e-11, -5e-11),
        'tau': (20*86400, 20*86400)
    }
    glitch_params_ranges = glitch_params_ranges or default_glitch_ranges
    glitch_params = gen_glitch_params(n, tstart, Tdata, n_glitches_range=(m, m), 
                                     df_permanent_range=glitch_params_ranges['df_permanent'],
                                     df_tau_range=glitch_params_ranges['df_tau'],
                                     df1_permanent_range=glitch_params_ranges['df1_permanent'],
                                     df1_tau_range=glitch_params_ranges['df1_tau'],
                                     tau_range=glitch_params_ranges['tau'])
    
    # Pad frequency parameters with zeros if freq_order < 4
    freq_params_padded = np.zeros((n, 5))
    freq_params_padded[:, :freq_order+1] = freq_params
    
    # Save parameters to .npz file
    np.savez(os.path.join(out_dir, 'params.npz'),
             freq_params=freq_params_padded,
             phi0=amp_params[:, 0], psi=amp_params[:, 1], cosi=amp_params[:, 2],
             alpha=sky_params[:, 0], delta=sky_params[:, 1],
             tglitch=[params[0] for params in glitch_params],
             df_permanent=[params[1] for params in glitch_params],
             df_tau=[params[2] for params in glitch_params],
             df1_permanent=[params[3] for params in glitch_params],
             df1_tau=[params[4] for params in glitch_params],
             tau=[params[5] for params in glitch_params])
    
    # Prepare arguments for multiprocessing
    sim_args = [( (freq_params_padded[i], amp_params[i, 0], amp_params[i, 1], amp_params[i, 2], 
                   sky_params[i, 0], sky_params[i, 1], glitch_params[i]),
                  h0, tstart, Tdata, dt_wf, detector, fmax, Tsft, out_dir, i)
                for i in range(n)]
    
    # Simulate signals in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(lambda args: simulate_signal(*args), sim_args), 
                  total=n, desc="Simulating signals"))
    
    print(f"Done. Time used: {time.time() - t0:.2f}s")

# Example usage matching original parameters
if __name__ == "__main__":
    tstart = 1368970000
    Tdata = 400 * 86400
    h0 = 1e-24
    dt_wf = 5
    detector = 'H1'
    fmax = 21
    Tsft = 1800
    n = 1  # One signal
    m = 2  # Two glitches # crab is ~ 1 per year, the most frequent one PSR J0537-6910, is ~ 3 per year, we can set it to [0,10] to be very conservative 
    
    freq_ranges = [(20.0, 20.0), (-1.35e-9, -1.35e-9)]  # Fixed freq, f1dot
    freq_order = 1  # f0 and f1
    glitch_params_ranges = {
        'df_permanent': (5e-3, 5e-3),
        'df_tau': (5e-2, 5e-2),
        'df1_permanent': (-5e-11, -5e-11),
        'df1_tau': (-5e-11, -5e-11),
        'tau': (20*86400, 20*86400)
    }
    
    main(
        n=n,
        m=m,
        h0=h0,
        tstart=tstart,
        Tdata=Tdata,
        dt_wf=dt_wf,
        detector=detector,
        fmax=fmax,
        Tsft=Tsft,
        freq_ranges=freq_ranges,
        freq_order=freq_order,
        glitch_params_ranges=glitch_params_ranges
    )
    
    
    
###############
