import lal
from lalpulsar import simulateCW
import time
import multiprocessing as mp
from utils import *

# Corrected waveform function
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
    h0_scale = h0 / np.sqrt(np.abs(f1dot0) / f0**5)
    
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
        if f1dot_eff == 0:
            raise ValueError("Effective f1dot is zero.")
        h0_t = h0_scale * np.sqrt(np.abs(f1dot_eff) / f_eff) 
        
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
        - out_dir (str): Output directory
        - signal_idx (int): Signal index
    """
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
    out_dir = signal_params['out_dir']
    signal_idx = signal_params['signal_idx']
    
    freq, f1dot, f2dot, f3dot, f4dot = freq_params
    
    # Normalize tglitch relative to tref
    glitch_params_norm = [
        [gp[0] - tref, gp[1], gp[2], gp[3], gp[4], gp[5]]
        for gp in glitch_params
    ]
    
    wf = waveform(h0, cosi, freq, f1dot, f2dot, f3dot, f4dot, glitch_params_norm)
    
    signal_out_dir = os.path.join(out_dir, f"simCW{signal_idx}")
    os.makedirs(signal_out_dir, exist_ok=True)
    
    S = simulateCW.CWSimulator(tref, tstart, Tdata, wf, dt_wf, phi0, psi, alpha, delta, detector)

    lim = f_lim(freq, age, Tdata/86400)
    fmin = float(freq-lim) # Minimum frequency of narrow-band
    fmax= float(freq+lim) # Maximum frequency of narrow-band   
    fband = int(fmax-fmin) # Total bandwidth covered by narrow-band

    for file, j, N in S.write_sft_files(noise_sqrt_Sh=sqrtSX, fmax=fmax, Tsft=Tsft, comment=f"simCW{signal_idx}", out_dir=signal_out_dir):
        pass
    
    combine_sfts(fmin=fmin, fmax=fmax, fband=fband, ts=tstart, te=tstart+Tdata, output=signal_out_dir, sft_dir=signal_out_dir)

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
        - out_dir (str): Output directory.
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
        'Tsft', 'out_dir', 'freq_ranges', 'freq_order',
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
    out_dir = params['out_dir']
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
    required_glitch_keys = ['delta_f_over_f', 'delta_f1dot_over_f1dot', 'Q', 'tau']
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
        delta_f_over_f_range=glitch_params_ranges['delta_f_over_f'],
        delta_f1dot_over_f1dot_range=glitch_params_ranges['delta_f1dot_over_f1dot'],
        Q_range=glitch_params_ranges['Q'],
        tau_range=glitch_params_ranges['tau']
    )
    
    # Pad frequency parameters with zeros if freq_order < 4
    freq_params_padded = np.zeros((n, 5))
    freq_params_padded[:, :freq_order+1] = freq_params
    
    # Save parameters to .cvs file
    save_params(n, m, tstart, freq_params_padded, amp_params, sky_params, glitch_params, out_dir, filename='signal_glitch_params.csv')
  
    
    # Create list of dictionaries for each signal
    sim_args = [
        {
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
            'out_dir': out_dir,
            'signal_idx': i
        }
        for i in range(n)
    ]
    
    # Simulate signals in parallel
    with mp.Pool(processes=n_cpu) as pool:
        list(tqdm(pool.imap_unordered(simulate_signal, sim_args), 
                  total=n, desc="Simulating signals"))
    
    print(f"Done. Time used: {time.time() - t0:.2f}s")

# Example usage
if __name__ == "__main__":
    
    from cw_manager.target import CassA as target
    
    freq = 450
    f1min, f1max = -freq/target.tau, 0
    f2min, f2max = 0, 7*f1min**2/freq

    sim_params = {
        'n': 500,
        'm': 0,
        'h0': 1e-25,
        'tstart': 1368970000,
        'Tdata': 156 * 86400,
        'dt_wf': 5,
        'detector': 'H1',
        'sqrtSX': 1e-23,
        'Tsft': 1800,
        'out_dir': './no_glitch',
        'age': target.age,
        'freq_ranges': [(freq, freq), (f1min, f1max), (f2min, f2max)],
        'freq_order': 2,
        'glitch_params_ranges': {
            'delta_f_over_f': (1e-6, 3e-6),
            'delta_f1dot_over_f1dot': (1e-3, 1e-2),
            'Q': (0.8, 1),
            'tau': (50*86400, 50*86400)
        },
        'alpha': target.alpha,
        'delta': target.delta,
        'seed': 0, 
        'n_cpu':16
    }

    main(sim_params)
