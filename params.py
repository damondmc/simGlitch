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
    #print("Generating amplitude parameters.")
    #phi_arr = np.random.uniform(0, 2*np.pi, nSample)
    #psi_arr = np.random.uniform(-np.pi/4, np.pi/4, nSample)
    #cosi_arr = np.random.uniform(-1, 1, nSample)
    
    phi_arr = np.zeros(nSample)
    psi_arr = np.ones(nSample) * np.pi / 4
    cosi_arr = np.ones(nSample)
    
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


#####################################################


def gen_glitch_params(n, m, 
                      tglitch_range=(1368970000, 1368970000 + 100*86400),
                      dnu_nu_range=(1e-9, 1e-6), 
                      dnu1_nu1_range=(-1e-4, -1e-3), 
                      Q_range=(0, 1), tau_range=(10*86400, 200*86400)):
    """
    Generate relative glitch parameters for n pulsars, each with m glitches.
    Fully vectorized for speed, returning chronological glitch events.
    """
    if m == 0:
        return [[] for _ in range(n)]
        
    print("Generating glitch parameters.")
        
    # 1. Glitch times: 
    tglitch = np.random.uniform(tglitch_range[0], tglitch_range[1], size=(n, m))
        
    # 2. Relative Frequency changes
    dnu_nu = np.random.uniform(dnu_nu_range[0], dnu_nu_range[1], size=(n, m))
    
    # 3. Spindown changes
    dnu1_nu1 = np.random.uniform(dnu1_nu1_range[0], dnu1_nu1_range[1], size=(n, m))
    
    # 4. Healing factors (Q)
    Q = np.random.uniform(Q_range[0], Q_range[1], size=(n, m))
    
    # 5. Decay timescales
    tau = np.random.uniform(tau_range[0], tau_range[1], size=(n, m))
    
    # 6. Stack arrays. Output shape: (n, m, 5)
    glitch_array = np.stack([tglitch, dnu_nu, dnu1_nu1, Q, tau], axis=-1)
    
    return glitch_array.tolist()
    
def calc_absolute_glitch_params(nu, nu1dot, dnu_nu, dnu1_nu1, Q):
    """
    Converts relative physical glitch parameters into absolute GW injection parameters.
    Can accept single floats or NumPy arrays (like Pandas DataFrame columns).
    
    Returns:
    - dnu_p: Permanent frequency jump
    - dnu_t: Transient frequency jump
    - dnu1_p: Permanent spindown jump
    """
    dnu = dnu_nu * nu
    
    dnu_p = (1 - Q) * dnu
    dnu_t = Q * dnu
    dnu1_p = dnu1_nu1 * nu1dot
    
    return dnu_p, dnu_t, dnu1_p

def save_params(n, m, freq_params, amp_params, sky_params, glitch_params, filepath):
    """
    Save parameters to a CSV file, combining source parameters with relative glitch parameters.
    """
    # Base headers and formats (14 columns)
    headers = ['n_th_signal', 'm_th_glitch', 'f0', 'f1', 'f2', 'f3', 'f4', 'phi0', 'psi', 'cosi', 'alpha', 'delta']
    fmt = ['%d', '%d', '%.8f', '%.8e', '%.8e', '%.8e', '%.8e', '%.8f', '%.8f', '%.8f', '%.8f', '%.8f']
    
    # Append new relative glitch-specific headers and formats (6 columns)
    if m > 0:
        headers.extend(['tglitch', 'dnu_nu', 'dnu1_nu1', 'Q', 'tau'])
        fmt.extend(['%.8f', '%.8e', '%.8e', '%.8f', '%.8f'])

    data = []
    
    for i in range(n):
        base_params = [
            i,                               
            *freq_params[i],        
            *amp_params[i],         
            *sky_params[i]          
        ]
        
        if m > 0:
            for j in range(m):
                row = base_params.copy()
                row.insert(1, j) 
                
                g_params = glitch_params[i][j]
                
                row.extend(g_params)
                data.append(row)
        else:
            row = base_params.copy()
            row.insert(1, 0)
            data.append(row)
            
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savetxt(filepath, np.array(data), delimiter=',', header=','.join(headers), comments='', fmt=fmt)
    
