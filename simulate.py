#!/home/hoitim.cheung/.conda/envs/glitch/bin/python
import lal
from lalpulsar import simulateCW
import time
import multiprocessing as mp
from utils import *
import argparse

def waveform(h0, cosi, freq, f1dot, f2dot, f3dot, f4dot, glitch_params_norm):
    """
    Generate GW waveform for a pulsar with glitches, ensuring h0 ∝ sqrt(|f1dot| / f^5).
    
    Parameters:
    - h0 (float): Initial strain amplitude.
    - cosi (float): Cosine of inclination angle.
    - freq (float): Initial frequency (f0).
    - f1dot (float): Initial frequency derivative.
    - f2dot, f3dot, f4dot (float): Higher-order frequency derivatives.
    - tglitch (list): Glitch times.
    - delta_f_p (list): Permanent frequency changes.
    - delta_f_t (list): Transient frequency changes.
    - delta_f1dot_p (list): Permanent frequency derivative changes.
    - tau (list): Glitch decay timescales.
    
    Returns:
    - wf (function): Waveform function returning (dphi, ap, ax).
    """
    
    # Validate glitch parameters
    for gp in glitch_params_norm:
        if len(gp) != 6:
            raise ValueError("Each glitch parameter set must contain 6 elements: "
                             "[tglitch, df_p, df_t, df1_p, tau, Q]")
    if f1dot == 0:
        raise ValueError("F1dot is zero.")

    # Initial amplitude scaling factor
    f0 = freq
    f1dot0 = f1dot
    h0_scale = h0 / np.sqrt(np.abs(f1dot0) / f0)
    
    def wf(dt):
        # Phase evolution
        dphi = freq * dt + f1dot * 0.5 * dt**2 + f2dot * (1./6.) * dt**3 + \
               f3dot * (1./24.) * dt**4 + f4dot * (1./120.) * dt**5
        
        # Initialize effective frequency and f1dot
        f_eff = f0
        f1dot_eff = f1dot0
        
        # Apply glitch contributions
        for gp in glitch_params_norm:
            tglitch, df_p, df_t, df1_p, tau, _ = gp
            if dt > tglitch:
                delta_t = dt - tglitch
                dphi += df_p * delta_t # nomial freq permanent change
                dphi += df_t * np.exp(-delta_t / tau) * delta_t # nomial freq transisent change
                dphi += df1_p * 0.5 * delta_t**2
                
                # Update effective frequency and f1dot
                f_eff += df_p + df_t * np.exp(-delta_t / tau)
                f1dot_eff += df1_p - df_t / tau * np.exp(-delta_t / tau)
        
        # Scale h0 based on effective f and f1dot
        if len(glitch_params_norm):
            if f1dot_eff == 0:
                raise ValueError("Effective f1dot is zero.")
            h0_t = h0_scale * np.sqrt(np.abs(f1dot_eff) / f_eff) 
            #h0_t = h0
        else:
            h0_t = h0
        
        dphi = lal.TWOPI * dphi
        ap = h0_t * (1.0 + cosi**2) / 2.0
        ax = h0_t * cosi
        return dphi, ap, ax
    
    return wf

def simulate_signal(signal_params):
    """
    Simulate a single continuous wave signal and generate SFT files.
    
    Parameters:
    - signal_params (dict): Dictionary containing signal parameters:
        - freq_params (array): [freq, f1dot, f2dot, f3dot, f4dot]
        - phi0 (float): Initial phase
        - psi (float): Polarization angle
        - cosi (float): Cosine of inclination
        - alpha (float): Right ascension
        - delta (float): Declination
        - glitch_params (tuple): Glitch parameters
        - h0 (float): Strain amplitude
        - tstart (float): Start time (GPS seconds)
        - Tdata (float): Duration (seconds)
        - dt_wf (float): Waveform time step (seconds)
        - detector (str): Detector name
        - Tsft (float): SFT duration (seconds)
        - label (str): Output directory label
        - signal_idx (int): Signal index
    """
    fmin = signal_params['fmin']
    fmax = signal_params['fmax']
    age = signal_params['age']
    tref = signal_params['tstart'] + 0.5 * signal_params['Tdata']
    freq_params = signal_params['freq_params']
    phi0 = signal_params['phi0']
    psi = signal_params['psi']
    cosi = signal_params['cosi']
    alpha = signal_params['alpha']
    delta = signal_params['delta']
    glitch_params = signal_params['glitch_params']
    h0 = signal_params['h0']
    tstart = signal_params['tstart']
    Tdata = signal_params['Tdata']
    dt_wf = signal_params['dt_wf']
    detector = signal_params['detector']
    Tsft = signal_params['Tsft']
    sqrtSX = signal_params['sqrtSX']
    label = signal_params['label']
    signal_idx = signal_params['signal_idx']
    
    freq, f1dot, f2dot, f3dot, f4dot = freq_params
    
    # Normalize tglitch relative to tref
    glitch_params_norm = [
        [gp[0] - tref, gp[1], gp[2], gp[3], gp[4], gp[5]]
        for gp in glitch_params
    ]
    
    wf = waveform(h0, cosi, freq, f1dot, f2dot, f3dot, f4dot, glitch_params_norm)
    
    signal_out_dir = os.path.join('/home/hoitim.cheung/glitch/data', label, f'{fmin}-{fmax}Hz', f"simCW{signal_idx}")
    temp_dir = os.path.join('/scratch/hoitim.cheung/data', label, f'{fmin}-{fmax}Hz', f"simCW{signal_idx}")
    
    os.makedirs(signal_out_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    S = simulateCW.CWSimulator(tref, tstart, Tdata, wf, dt_wf, phi0, psi, alpha, delta, detector)

    lim = f_lim(freq, age, Tdata/86400)
    fmin = float(freq-lim) # Minimum frequency of narrow-band
    fmax= float(freq+lim) # Maximum frequency of narrow-band   
    fband = int(fmax-fmin) # Total bandwidth covered by narrow-band

    for file, j, N in S.write_sft_files(noise_sqrt_Sh=sqrtSX, fmax=fmax, Tsft=Tsft, comment=f"simCW{signal_idx}", out_dir=temp_dir):
        pass
    
    combine_sfts(fmin=fmin, fmax=fmax, fband=fband, ts=tstart, te=tstart+Tdata, output=signal_out_dir, sft_dir=temp_dir)
    
# Updated main function with strict parameter validation
def main(params):
    """
    Generate parameters for n signals with m glitches each, save them, and simulate signals in parallel.

    Parameters:
    - params (dict): Dictionary containing:
        - n (int): Number of signals.
        - m (int): Number of glitches per signal.
        - h0 (float): Strain amplitude.
        - tstart (float): Start time (GPS seconds).
        - Tdata (float): Duration (seconds).
        - dt_wf (float): Waveform time step (seconds).
        - detector (str): Detector name (e.g., 'H1').
        - sqrtSX' (float): Noise amplitude.
        - Tsft (float): SFT duration (seconds).
        - label (str): Output directory label.
        - freq_ranges (list): List of (min, max) tuples for frequency derivatives.
        - freq_order (int): Order of frequency derivatives (0 to 4).
        - glitch_params_ranges (dict): Ranges for glitch parameters.
        - alpha (float, optional): Fixed right ascension (radians).
        - delta (float, optional): Fixed declination (radians).
        - seed (int, optional): Random seed for reproducibility.

    Returns:
    - None: Generates SFT files and saves parameters to params.npz.

    Raises:
    - KeyError: If any required parameter is missing from params.
    """
    # Define required parameters
    required_params = [
        'n', 'm', 'h0', 'tstart', 'Tdata', 'dt_wf', 'detector',
        'Tsft', 'label', 'freq_ranges', 'freq_order',
        'glitch_params_ranges'
    ]
    
    # Check for missing parameters
    missing_params = [key for key in required_params if key not in params]
    if missing_params:
        raise KeyError(f"Missing required parameters: {', '.join(missing_params)}")
    
    # Extract parameters
    n = params['n']
    m = params['m']
    h0 = params['h0']
    tstart = params['tstart']
    Tdata = params['Tdata']
    dt_wf = params['dt_wf']
    detector = params['detector']
    Tsft = params['Tsft']
    sqrtSX = params['sqrtSX']
    label = params['label']
    age = params['age']
    freq_ranges = params['freq_ranges']
    freq_order = params['freq_order']
    glitch_params_ranges = params['glitch_params_ranges']
    alpha = params.get('alpha')
    delta = params.get('delta')
    seed = params.get('seed')
    n_cpu = params.get('n_cpu')
    
    # Validate parameters
    if freq_order > 4:
        raise ValueError("freq_order must be <= 4")
    if len(freq_ranges) != freq_order + 1:
        raise ValueError(f"freq_ranges must contain exactly {freq_order + 1} (min, max) pairs")
    if alpha is not None and (alpha < 0 or alpha > 2 * np.pi):
        raise ValueError("alpha must be in [0, 2π]")
    if delta is not None and (delta < -np.pi/2 or delta > np.pi/2):
        raise ValueError("delta must be in [-π/2, π/2]")
    required_glitch_keys = ['tglitch', 'delta_f_over_f', 'delta_f1dot_over_f1dot', 'Q', 'tau']
    missing_glitch_keys = [key for key in required_glitch_keys if key not in glitch_params_ranges]
    if missing_glitch_keys:
        raise KeyError(f"Missing required glitch_params_ranges keys: {', '.join(missing_glitch_keys)}")
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        print(f"Using random seed: {seed}")
    
    t0 = time.time()
    
    # Generate NS parameters
    freq_params = gen_frequency_params(n, freq_order, freq_ranges)
    amp_params = gen_amplitude_params(n)
    if alpha is not None and delta is not None:
        alpha_arr = np.full(n, alpha)
        delta_arr = np.full(n, delta)
        sky_params = np.column_stack((alpha_arr, delta_arr))
        print(f"Using fixed sky location: alpha={alpha}, delta={delta}")
    else:
        sky_params = gen_sky_location_params(n)
        
    # Generate glitch parameters
    glitch_params = gen_glitch_params(
        n, m, tstart, Tdata, freq_params[:, 0], freq_params[:, 1],
        tglitch_range=glitch_params_ranges['tglitch'],
        delta_f_over_f_range=glitch_params_ranges['delta_f_over_f'],
        delta_f1dot_over_f1dot_range=glitch_params_ranges['delta_f1dot_over_f1dot'],
        Q_range=glitch_params_ranges['Q'],
        tau_range=glitch_params_ranges['tau']
    )
    
    
    tglitch = np.linspace(glitch_params_ranges['tglitch'][0], glitch_params_ranges['tglitch'][1], n)
        
    for i in range(n):
        glitch_params[i][0][0] = tglitch[i]
    
    # Pad frequency parameters with zeros if freq_order < 4
    freq_params_padded = np.zeros((n, 5))
    freq_params_padded[:, :freq_order+1] = freq_params
    
    fmin, fmax = freq_ranges[0]
    # Save parameters to .cvs file
    save_params(h0, sqrtSX, fmin, fmax, n, m, tstart, freq_params_padded, amp_params, sky_params, glitch_params, label, filename='signal_glitch_params.csv')
  
    
    # Create list of dictionaries for each signal
    sim_args = [
        {
            'fmin': fmin,
            'fmax': fmax,
            'age': age,
            'freq_params': freq_params_padded[i],
            'phi0': amp_params[i, 0],
            'psi': amp_params[i, 1],
            'cosi': amp_params[i, 2],
            'alpha': sky_params[i, 0],
            'delta': sky_params[i, 1],
            'glitch_params': glitch_params[i],
            'h0': h0,
            'tstart': tstart,
            'Tdata': Tdata,
            'dt_wf': dt_wf,
            'detector': detector,
            'Tsft': Tsft,
            'sqrtSX': sqrtSX,
            'label': label,
            'signal_idx': i
        }
        for i in range(n)
    ]
    
    # Simulate signals in parallel
    with mp.Pool(processes=n_cpu) as pool:
        list(tqdm(pool.imap_unordered(simulate_signal, sim_args), 
                  total=n, desc="Simulating signals"))
    
    print(f"\nDone. Time used: {time.time() - t0:.2f}s")

# Example usage
if __name__ == "__main__":
    
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run lalpulsar_Weave commands with multiprocessing.")
    parser.add_argument('--t_ref', type=int, default=10,
                        help="Number of CPU cores to use for multiprocessing (default: all available)")
    parser.add_argument('--label', default='no_glitch',
                        help="Label for data directory (no_glitch or with_glitch)")
    
    args = parser.parse_args()

    
    from cw_manager.target import CassA as target
    
    freq = 100
    f1min, f1max = -freq/target.tau, 0
    f2min, f2max = 0, 7*f1min**2/freq

    f1min, f1max = -freq/target.tau, -freq/target.tau
    f2min, f2max = 0, 0 

    
    depth = 40
    sqrtSX = 1e-23 
    h0 = sqrtSX/depth 
    t_ref = args.t_ref
    label = args.label
    
    print(f"depth:{depth}, sqrtSX:{sqrtSX}, h0:{h0}")
    
    sim_params = {
        'n': 32,
        'm': 1,
        'h0': h0,
        'tstart': 1368970000,
        'Tdata': 100 * 86400,
        'dt_wf': 5,
        'detector': 'H1',
        'sqrtSX': sqrtSX,
        'Tsft': 1800,
        'label': label,
        'age': target.tau,
        'freq_ranges': [(freq, freq), (f1min, f1max), (f2min, f2max)],
        'freq_order': 2,
        'glitch_params_ranges': {
            'tglitch': (1368970000 + 0*86400, 1368970000 + 100*86400), 
            'delta_f_over_f': (1e-6, 1e-6),
            'delta_f1dot_over_f1dot': (1e-4, 1e-4),
            'Q': (0.2, 0.2),
            'tau': (20*86400, 20*86400)
        },
        'alpha': target.alpha,
        'delta': target.delta,
        'seed': 0, 
        'n_cpu':32
    }

#         'glitch_params_ranges': {
#             'delta_f_over_f': (1e-9, 3e-6),
#             'delta_f1dot_over_f1dot': (1e-4, 1e-1),
#             'Q': (0.1, 0.9),
#             'tau': (20*86400, 300*86400)
#         },
    
    main(sim_params)
