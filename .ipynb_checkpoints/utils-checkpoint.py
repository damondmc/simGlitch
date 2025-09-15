import numpy as np
import os
import glob
import subprocess
from pathlib import Path
from tqdm import tqdm

def f_lim(freq, tau=300., obsDay=240.):
    # limit for narrowband [f-lim/2,f+0.9+lim/2] needed for search using 1 year data, ref time in the middle of the search 
    # lim ~ fdot *(t-tref) and |fdot| <= f/tau, tau = 300,700, lim<= f*(t-tref)/tau = 0.5 * f/300 * Tobs/365
    # use 0.6 instead of 0.5 for antenna pattern buffer and multiple by the ration of total obs. day to a year
    lim = np.ceil(0.8 * (freq/tau) * (obsDay/365.) ) 
    return lim

def gen_amplitude_params(nSample, phi_range=(0, 2*np.pi), psi_range=(-np.pi/4, np.pi/4), cosi_range=(-1, 1)):
    """
    Generate amplitude parameters (phi, psi, cosi) for a given number of random samples.

    Parameters:
    - nSample (int): Number of random samples.
    - phi_range (tuple): Min and max for phase (default: 0 to 2π).
    - psi_range (tuple): Min and max for polarization angle (default: -π/4 to π/4).
    - cosi_range (tuple): Min and max for cosine of inclination (default: -1 to 1).

    Returns:
    - params (np.ndarray): Array of shape (nSample, 3) with columns [phi, psi, cosi].
    """
    #print("Generating amplitude parameters.")
    phi_arr = np.random.uniform(0, 2*np.pi, nSample)
    psi_arr = np.random.uniform(-np.pi/4, np.pi/4, nSample)
    cosi_arr = np.random.uniform(-1, 1, nSample)
    return np.column_stack((phi_arr, psi_arr, cosi_arr))

def gen_sky_location_params(nSample, alpha_range=(0, 2*np.pi), sinDelta_range=(-1, 1)):
    """
    Generate sky location parameters (alpha, delta) for a given number of random samples.

    Parameters:
    - nSample (int): Number of random samples.
    - alpha_range (tuple): Min and max for right ascension (default: 0 to 2π).
    - sinDelta_range (tuple): Min and max for sin(delta) (default: -1 to 1).

    Returns:
    - params (np.ndarray): Array of shape (nSample, 2) with columns [alpha, delta].
    """
    #print("Generating sky location parameters.")
    alpha_arr = np.random.uniform(alpha_range[0], alpha_range[1], nSample)
    sinDelta = np.random.uniform(sinDelta_range[0], sinDelta_range[1], nSample)
    delta_arr = np.arcsin(sinDelta)
    return np.column_stack((alpha_arr, delta_arr))

def gen_frequency_params(nSample, n, freq_ranges):
    """
    Generate frequency parameters (f0, f1, ..., fn) for a given number of random samples.

    Parameters:
    - nSample (int): Number of random samples.
    - n (int): Order of frequency derivatives (0 to 4).
    - freq_ranges (list): List of (min, max) tuples for each frequency derivative [f0, f1, ..., fn].

    Returns:
    - params (np.ndarray): Array of shape (nSample, n+1) with columns [f0, f1, ..., fn].

    Raises:
    - ValueError: If n > 4 or freq_ranges length does not match n+1.
    """
    #print("Generating frequency parameters.")
    if n > 4:
        raise ValueError("Order n must be <= 4.")
    if len(freq_ranges) != n + 1:
        raise ValueError(f"freq_ranges must contain exactly {n + 1} (min, max) pairs for f0 to f{n}.")
    
    freq_arrays = []
    for i, (xmin, xmax) in enumerate(freq_ranges):
        freq_arrays.append(np.random.uniform(xmin, xmax, nSample))
    
    return np.column_stack(freq_arrays)



def gen_glitch_params(n, m, tstart, Tdata, freq, f1dot, 
                     delta_f_over_f_range=(1e-9, 1e-6), delta_f1dot_over_f1dot_range=(-1e-4, -1e-3), 
                     Q_range=(0, 1), tau_range=(10*86400, 200*86400)):
    """
    Generate glitch parameters for n pulsars, each with a specified number of glitches.
    Parameters are drawn based on observables delta_f/f, delta_f1/f1, and Q.
    """
    print("Generating glitch parameters.")
    glitch_params = []
    
    freq = np.atleast_1d(freq)
    f1dot = np.atleast_1d(f1dot)
    
    
    for i in range(n):
        if m == 0:
            glitch_params.append([])
            continue
        
        tglitch = np.random.uniform(tstart, tstart + Tdata, m)
        delta_f_over_f = np.random.uniform(delta_f_over_f_range[0], delta_f_over_f_range[1], m)
        delta_f = delta_f_over_f * freq[i]
        Q = np.random.uniform(Q_range[0], Q_range[1], m)
        delta_f_t = Q * delta_f 
        delta_f_p = (1-Q) * delta_f 
        delta_f1dot_over_f1dot = np.random.uniform(delta_f1dot_over_f1dot_range[0], delta_f1dot_over_f1dot_range[1], m)
        delta_f1_p = delta_f1dot_over_f1dot * f1dot[i]
        tau = np.random.uniform(tau_range[0], tau_range[1], m)
        
        # Create list of m glitch parameter sets for this pulsar
        glitch_sets = [
            [tglitch[j], delta_f_p[j], delta_f_t[j], delta_f1_p[j], tau[j], Q[j]]
            for j in range(m)
        ]
        glitch_params.append(glitch_sets)
    
    return glitch_params
    
def save_params(fmin, fmax, n, m, tstart, freq_params, amp_params, sky_params, glitch_params, label, filename='params.csv'):
    """
    Save parameters to a CSV file with n*m rows, combining source parameters with glitch parameters.
    Handles cases where glitch parameters are empty.

    Parameters:
    n : int
        Number of signals/sources.
    m : int
        Number of glitches per signal.
    tstart : float
        Start time (GPS seconds)
    freq_params : ndarray
        Frequency parameters array of shape (n, 5).
    amp_params : ndarray
        Amplitude parameters array of shape (n, 3) containing [phi0, psi, cosi].
    sky_params : ndarray
        Sky parameters array of shape (n, 2) containing [alpha, delta].
    glitch_params : list
        List of length n, where each element is a list of m glitch parameter sets.
        Each glitch parameter set contains [tglitch, df_permanent, df_transisent, df1_permanent, tau, Q].
        Can be empty for some or all signals.
    label : str
        Output directory path label for the CSV file.
    filename : str
        Name of the output CSV file (default: 'params.csv').
    """

    # Check if glitch_params is empty or m == 0
    if m > 0:
        # Standard case: glitches exist
        num_columns = 2 + 5 + 3 + 2 + 6 + 1  # n_th_signal, m_th_glitch, freq_params (5), amp_params (3), sky_params (2), glitch_params (6), tglitch_day
        data = np.zeros((n * m, num_columns))
        headers = ['n_th_signal', 'm_th_glitch', 'f0', 'f1', 'f2', 'f3', 'f4', 'phi0', 'psi', 'cosi', 'alpha', 'delta',
                   'tglitch', 'df_p', 'df_t', 'df1_p', 'tau', 'Q', 'tglitch_day']
        fmt = ['%d', '%d', '%.8f', '%.8e', '%.8e', '%.8e', '%.8e', '%.8f', '%.8f', '%.8f', '%.8f', '%.8f',
               '%d', '%.8e', '%.8e', '%.8e', '%.8f', '%.8f', '%.2f']

        # Fill in the data
        for i in range(n):
            for j in range(m):
                row_idx = i * m + j
                data[row_idx, 0] = i  # n-th signal
                data[row_idx, 1] = j  # m-th glitch
                data[row_idx, 2:7] = freq_params[i]  # f0 to f4
                data[row_idx, 7:10] = amp_params[i]  # phi0, psi, cosi
                data[row_idx, 10:12] = sky_params[i]  # alpha, delta
                data[row_idx, 12:18] = glitch_params[i][j]  # tglitch, df_permanent, df_transisent, df1_permanent, tau, Q
                data[row_idx, -1] = (glitch_params[i][j][0] - tstart) / 86400  # tglitch in days
    else:
        # No glitches: only include signal parameters
        num_columns = 2 + 5 + 3 + 2  # n_th_signal, m_th_glitch, freq_params (5), amp_params (3), sky_params (2)
        data = np.zeros((n, num_columns))
        headers = ['n_th_signal', 'm_th_glitch', 'f0', 'f1', 'f2', 'f3', 'f4', 'phi0', 'psi', 'cosi', 'alpha', 'delta']
        fmt = ['%d', '%d', '%.8f', '%.8e', '%.8e', '%.8e', '%.8e', '%.8f', '%.8f', '%.8f', '%.8f', '%.8f']

        # Fill in the data
        for i in range(n):
            data[i, 0] = i  # n-th signal
            data[i, 1] = 0  # m-th glitch (set to 0 as there are no glitches)
            data[i, 2:7] = freq_params[i]  # f0 to f4
            data[i, 7:10] = amp_params[i]  # phi0, psi, cosi
            data[i, 10:12] = sky_params[i]  # alpha, delta

    # Save to CSV
    savepath = os.path.join('/home/hoitim.cheung/glitch/data', label, f'{fmin}-{fmax}Hz')
    os.makedirs(savepath, exist_ok=True)
    
    filepath = os.path.join('/home/hoitim.cheung/glitch/data', label, f'{fmin}-{fmax}Hz', filename)
    np.savetxt(filepath, data, delimiter=',', header=','.join(headers), comments='', fmt=fmt)


def combine_sfts(fmin, fmax, fband, ts, te, output, sft_dir, fx=0.0):
    """
    Run lalpulsar_splitSFTs command for SFT files in a directory, sorted by timestamp.
    
    Parameters:
    - fmin (float): Minimum frequency
    - fmax (float): Maximum frequency
    - fband (float): Frequency band
    - fx (float): Frequency step
    - ts (int): Start time
    - te (int): End time
    - output (str): Output directory or filename prefix
    - sft_dir (str): Directory containing SFT files
    
    Returns:
    - None: Executes the command for each SFT file
    """
    # Get all .sft files in the directory
    sft_files = glob.glob(os.path.join(sft_dir, "*.sft"))
    
    # Sort files by timestamp (extracted from filename)
    # Assumes filename format like H-1_H1_1800SFT_simCW0-TIMESTAMP-DURATION.sft
    def get_timestamp(filename):
        # Extract the timestamp part (e.g., 1368970000 from H-1_H1_1800SFT_simCW0-1368970000-1800.sft)
        parts = os.path.basename(filename).split('-')
        if len(parts) >= 3:
            try:
                return int(parts[-2])  # Timestamp is second-to-last part
            except ValueError:
                return 0  # Fallback if timestamp is not an integer
        return 0
    
    sft_files = sorted(sft_files, key=get_timestamp)
    
    # Construct the command template
    cmd_template = (
        "lalpulsar_splitSFTs "
        "-fs {} -fe {} -fb {} -fx {} -ts {} -te {} -n {} -- {}"
    )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    
    # Run the command for each SFT file
    for sft in tqdm(sft_files, total=len(sft_files), desc="Combining SFTs..."):
        cmd = cmd_template.format(fmin, fmax, fband, fx, ts, te, output, sft)
        print(f"Executing: {cmd}")
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command for {sft}: {e}")
   
    # Run the command to remove sfts
    for sft in sft_files:
        try:
            os.remove(sft)  # Delete the SFT file after successful processing
        except OSError as e:
            print(f"Error removing file {sft}: {e}")
