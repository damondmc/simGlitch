import numpy as np
import os

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
    print("Generating amplitude parameters.")
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
    print("Generating sky location parameters.")
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
    print("Generating frequency parameters.")
    if n > 4:
        raise ValueError("Order n must be <= 4.")
    if len(freq_ranges) != n + 1:
        raise ValueError(f"freq_ranges must contain exactly {n + 1} (min, max) pairs for f0 to f{n}.")
    
    freq_arrays = []
    for i, (fmin, fmax) in enumerate(freq_ranges):
        freq_arrays.append(np.random.uniform(fmin, fmax, nSample))
    
    return np.column_stack(freq_arrays)



def gen_glitch_params(n, m, tstart, Tdata, freq, f1dot, 
                     delta_f_over_f_range=(1e-9, 1e-6), delta_f1dot_over_f1dot_range=(-1e-4, -1e-3), 
                     Q_range=(0, 1), tau_range=(10*86400, 200*86400)):
    """
    Generate glitch parameters for n pulsars, each with a specified number of glitches.
    Parameters are drawn based on observables delta_f/f, delta_f1dot/f1dot, and Q.
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
        delta_f1dot_p = delta_f1dot_over_f1dot * f1dot[i]
        tau = np.random.uniform(tau_range[0], tau_range[1], m)
        
        # Create list of m glitch parameter sets for this pulsar
        glitch_sets = [
            [tglitch[j], delta_f_p[j], delta_f_t[j], delta_f1dot_p[j], tau[j], Q[i]]
            for j in range(m)
        ]
        glitch_params.append(glitch_sets)
    
    return glitch_params

def save_params(n, m, freq_params, amp_params, sky_params, glitch_params, out_dir, filename='params.csv'):
    """
    Save parameters to a CSV file with n*m rows, combining source parameters with glitch parameters.

    Parameters:
    n : int
        Number of signals/sources.
    m : int
        Number of glitches per signal.
    freq_params : ndarray
        Frequency parameters array of shape (n, 5).
    amp_params : ndarray
        Amplitude parameters array of shape (n, 3) containing [phi0, psi, cosi].
    sky_params : ndarray
        Sky parameters array of shape (n, 2) containing [alpha, delta].
    glitch_params : list
        List of length n, where each element is a list of m glitch parameter sets.
        Each glitch parameter set contains [tglitch, df_permanent, df_tau, df1_permanent, df1_tau, tau].
    out_dir : str
        Output directory path for the CSV file.
    filename : str
        Name of the output CSV file (default: 'params.csv').
    """

    # Prepare data for CSV
    data = np.zeros((n*m, 2 + 5 + 3 + 2 + 6))  # n-th signal, m-th glitch, freq_params (5), amp_params (3), sky_params (2), glitch_params (6)

    # Fill in the data
    for i in range(n):
        for j in range(m):
            row_idx = i * m + j
            data[row_idx, 0] = i  # 1-based indexing for n-th signal
            data[row_idx, 1] = j  # 1-based indexing for m-th glitch
            data[row_idx, 2:7] = freq_params[i]  # f0 to f4
            data[row_idx, 7:10] = amp_params[i]         # phi0, psi, cosi
            data[row_idx, 10:12] = sky_params[i]       # alpha, delta
            data[row_idx, 12:18] = glitch_params[i][j]  # tglitch, df_permanent, df_tau, df1_permanent, df1_tau, tau

    # Define column headers
    headers = ['n_th_signal', 'm_th_glitch', 'f0', 'f1', 'f2', 'f3', 'f4', 'phi0', 'psi', 'cosi', 'alpha', 'delta',
               'tglitch', 'df_permanent', 'df_tau', 'df1_permanent', 'df1_tau', 'tau']

    # Save to CSV
    np.savetxt(os.path.join(out_dir, filename), data, delimiter=',', header=','.join(headers), comments='')