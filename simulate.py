#!/home/hoitim.cheung/.conda/envs/lalsuite-dev/bin/python
import os
import shutil
import lal
import lalpulsar as lp
from lalpulsar import simulateCW
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from params import calc_absolute_glitch_params


def waveform(h0, cosi, freq, f1dot, f2dot, f3dot, f4dot, glitch_params_norm):
    """Generate GW waveform for a pulsar with glitches."""
    for gp in glitch_params_norm:
        if len(gp) != 5:
            raise ValueError("Each absolute glitch parameter set must contain 5 elements: "
                             "[tglitch, df_p, df_t, df1_p, tau]")
    if f1dot == 0:
        raise ValueError("F1dot is zero.")

    f0 = freq
    f1dot0 = f1dot
    h0_scale = h0 / np.sqrt(np.abs(f1dot0) / f0)
    
    def wf(dt):
        dphi = freq * dt + f1dot * (1./2.) * dt**2 + f2dot * (1./6.) * dt**3 + \
               f3dot * (1./24.) * dt**4 + f4dot * (1./120.) * dt**5
        
        f_eff = f0
        f1dot_eff = f1dot0
        
        for gp in glitch_params_norm:
            tglitch, df_p, df_t, df1_p, tau = gp
            if dt > tglitch:
                delta_t = dt - tglitch
                dphi += df_p * delta_t 
                dphi += df_t * tau * (1 - np.exp(-delta_t / tau)) 
                dphi += df1_p * 0.5 * delta_t**2
                
                f_eff += df_p + df_t * np.exp(-delta_t / tau)
                f1dot_eff += df1_p - df_t / tau * np.exp(-delta_t / tau)
        
        if len(glitch_params_norm):
            if f1dot_eff == 0:
                raise ValueError("Effective f1dot is zero.")
            h0_t = h0_scale * np.sqrt(np.abs(f1dot_eff) / f_eff) 
        else:
            h0_t = h0
        
        dphi = lal.TWOPI * dphi
        ap = h0_t * (1.0 + cosi**2) / 2.0
        ax = h0_t * cosi
        return dphi, ap, ax
    
    return wf

def simulate_signal(signal_params):
    """Simulate a single continuous wave signal and generate SFT files."""
    tref = signal_params['tref']
    freq_params = signal_params['freq_params']
    phi0 = signal_params['phi0']
    psi = signal_params['psi']
    cosi = signal_params['cosi']
    alpha = signal_params['alpha']
    delta = signal_params['delta']
    glitch_params = signal_params['glitch_params']
    h0 = signal_params['h0']
    
    # Observational Parameters
    timestamps = signal_params['timestamps']
    dt_wf = signal_params['dt_wf']
    window_type = signal_params['window_type']
    window_param = signal_params['window_param']
    IFOS = signal_params['IFOS']
    Tsft = signal_params['Tsft']
    sqrtSX = signal_params['sqrtSX']
    sft_dir = signal_params['sft_dir']
    save_path = signal_params['save_path']
    signal_idx = signal_params['signal_idx']
    
    freq, f1dot, f2dot, f3dot, f4dot = freq_params
    
    df_glitch = 0.0
    tstart = min([timestamps[ifo][0] for ifo in IFOS])
    Tobs = max([timestamps[ifo][-1] for ifo in IFOS]) - min([timestamps[ifo][0] for ifo in IFOS])

    # Convert relative parameters to absolute, and normalize tglitch relative to tref
    glitch_params_norm = []
    for gp in glitch_params:
        tglitch, dnu_nu, dnu1_nu1, Q, tau = gp
        tglitch_rel = tstart + tglitch * Tobs - tref
        dnu_p, dnu_t, dnu1_p = calc_absolute_glitch_params(freq, f1dot, dnu_nu, dnu1_nu1, Q)
        glitch_params_norm.append([tglitch_rel - tref, dnu_p, dnu_t, dnu1_p, tau])
    
        df_glitch += abs(dnu_p) + abs(dnu_t) + (abs(dnu1_p) * Tobs)
        
    # waveform 
    wf = waveform(h0, cosi, freq, f1dot, f2dot, f3dot, f4dot, glitch_params_norm)
    
    # Frequency band for SFT generation (centered around f0 with some margin for spindown and glitches)
    df_doppler = 1.05e-4 * freq
    safety_buffer = 0.01  # Small 0.01 Hz buffer for higher-order derivatives and binning
    # Base spindown deviation over the observation time
    df_spindown = abs(f1dot) * Tobs
    df = df_spindown + df_glitch + df_doppler + safety_buffer

    fmin = float(np.floor(freq - df))
    fmax = float(np.ceil(freq + df))
    fband = int(fmax-fmin) 

    # Create output directories for this signal
    signal_out_dir = os.path.join(save_path, f"simCW{signal_idx}")
    single_sft_dir = os.path.join(save_path, 'tmp', f"simCW{signal_idx}_large")    
    cropped_sft_dir = os.path.join(save_path, 'tmp', f"simCW{signal_idx}_cropped")
    
    os.makedirs(signal_out_dir, exist_ok=True)
    os.makedirs(single_sft_dir, exist_ok=True)
    os.makedirs(cropped_sft_dir, exist_ok=True)

    print(f"[Signal {signal_idx}] Starting. Band: [{fmin}, {fmax}] Hz. df: {df:.2e}", flush=True)

    for ifo in IFOS:
        print(f"[Signal {signal_idx}] {ifo}: Simulating {len(timestamps[ifo])} SFTs...", flush=True)
        t0_gen = time.time()
        
        for i, t in enumerate(timestamps[ifo]):
            simulator = simulateCW.CWSimulator(tref, t, Tsft, wf, dt_wf, phi0, psi, alpha, delta, ifo)
            for file, _, _ in simulator.write_sft_files(noise_sqrt_Sh=sqrtSX, fmax=fmax, Tsft=Tsft, comment=f"simCW{signal_idx}", out_dir=single_sft_dir):
                single_cat = lp.SFTdataFind(file, constraints=None)
                tiny_sft = lp.LoadSFTs(single_cat, fmin, fmax)
                
                spec = lp.SFTFilenameSpec()
                spec.path = cropped_sft_dir
                spec.window_type = window_type
                spec.window_param = window_param
                spec.privMisc = f'simCW{signal_idx}'
                lp.WriteSFTVector2StandardFile(tiny_sft, spec, SFTcomment='simCW', merged=False)

                os.remove(file)

            if (i + 1) % 3000 == 0:
                print(f"[Signal {signal_idx}] {ifo}: Generated {i + 1}/{len(timestamps[ifo])} SFTs...", flush=True)

        print(f"[Signal {signal_idx}] {ifo}: Generation finished in {time.time() - t0_gen:.1f}s. Loading into memory...", flush=True)

        # 1. Load the generated signal/noise
        t0_load = time.time()
        sim_catalog = lp.SFTdataFind(f"{cropped_sft_dir}/*.sft", constraints=None)
        sim_sfts = lp.LoadSFTs(sim_catalog, -1, -1)
        print(f"[Signal {signal_idx}] {ifo}: Loaded {sim_sfts.length} sim SFTs in {time.time() - t0_load:.1f}s.", flush=True)

        # 2. Determine final SFT vector
        if sft_dir is not None:
            # Handle Real Data
            constraints = lp.SFTConstraints()
            constraints.detector = ifo
            sft_files = os.path.join(sft_dir, f'{ifo[0]}-*.sft')
            data_catalog = lp.SFTdataFind(sft_files, constraints=constraints)
            data_sfts = lp.LoadSFTs(data_catalog, fmin, fmax)

            real_timestamps = np.array([float(data_sfts.data[i].epoch) for i in range(data_sfts.length)])
            if len(real_timestamps) != len(timestamps[ifo]) or not np.allclose(real_timestamps, timestamps[ifo], atol=1.0):
                raise ValueError(
                    f"Timestamp mismatch for {ifo}!\n"
                    f"Real SFTs loaded: {len(real_timestamps)}\n"
                    f"Timestamps file provided: {len(timestamps[ifo])}\n"
                    f"Ensure the --timestamps file perfectly matches the SFTs located at {sft_files}."
                )
            print(f"[Signal {signal_idx}] {ifo}: Resizing bands and adding SFTs...", flush=True)
            lp.SFTVectorResizeBand(data_sfts, fmin, fband)
            lp.SFTVectorResizeBand(sim_sfts, fmin, fband)
            lp.SFTVectorAdd(data_sfts, sim_sfts)
        else:
            print(f"[Signal {signal_idx}] {ifo}: Resizing sim band...", flush=True)
            lp.SFTVectorResizeBand(sim_sfts, fmin, fband)
            data_sfts = sim_sfts   # We will save the generated noise
            
        # Write merged file to standard format
        spec = lp.SFTFilenameSpec()
        spec.path = signal_out_dir
        spec.window_type = window_type
        spec.window_param = window_param
        spec.privMisc = f'simCW{signal_idx}'
        lp.WriteSFTVector2StandardFile(data_sfts, spec, SFTcomment='simCW', merged=True)
        
        # Clean up the tiny cropped files for this IFO
        shutil.rmtree(cropped_sft_dir)
        if os.path.exists(single_sft_dir):
            shutil.rmtree(single_sft_dir)

    print(f"[Signal {signal_idx}] Fully complete.", flush=True)

def main(timestamps, df, obs_params, save_path, n_cpu):
    # Group by signal index to handle multiple glitches per pulsar
    signals = df.groupby('n_th_signal')
    sim_args = []
    
    for signal_idx, group in signals:
        # Extract Astrophysical Parameters
        row = group.iloc[0]
        freq_params = [row['f0'], row['f1'], row['f2'], row['f3'], row['f4']]
        phi0, psi, cosi = row['phi0'], row['psi'], row['cosi']
        alpha, delta = row['alpha'], row['delta']
        
        # Glitch parameteres
        glitch_params = []
        for _, g_row in group.iterrows():
            # Check if tglitch exists and is not NaN (handles m=0 cases gracefully)
            if 'tglitch' in g_row and pd.notna(g_row['tglitch']):
                glitch_params.append([
                    g_row['tglitch'], g_row['dnu_nu'], g_row['dnu1_nu1'], g_row['Q'], g_row['tau']
                ])
        
        # 3. Combine with Observational Parameters
        sim_args.append({
            # Astrophysical
            'signal_idx': int(signal_idx),
            'freq_params': freq_params,
            'phi0': phi0, 'psi': psi, 'cosi': cosi,
            'alpha': alpha, 'delta': delta,
            'glitch_params': glitch_params,
            'h0': obs_params['h0'],

            'timestamps': timestamps,
            'sft_dir': obs_params['sft_dir'],
            'sqrtSX': obs_params['sqrtSX'],
            'tref': obs_params['tref'],
            'dt_wf': obs_params['dt_wf'],
            'window_type': obs_params['window_type'],
            'window_param': obs_params['window_param'],
            'IFOS': obs_params['IFOS'],
            'Tsft': obs_params['Tsft'],
            'save_path': save_path
        })
        
    n_signals = len(sim_args)
    print(f"Loaded {n_signals} signals. Starting simulation pool with {n_cpu} CPUs...")
    
    t0 = time.time()
    with mp.Pool(processes=n_cpu) as pool:
        list(tqdm(pool.imap_unordered(simulate_signal, sim_args), total=n_signals, desc="Simulating signals"))
    
    temp_dir = os.path.join(save_path, 'tmp')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    print(f"\nDone. Time used: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run continuous wave simulations from CSV parameters.")
    parser.add_argument('--n_cpu', type=int, default=8, help="Number of CPU cores to use")
    parser.add_argument('--label', default='wglitch_f1_3e-9', help="Label for data directory")
    parser.add_argument('--freq', type=float, default=100.0, help="Target frequency")
    parser.add_argument('--timestamps_file', default='/scratch/kriles_root/kriles0/damoncht/cwglitch/data/real_data/o4a_timestamps.csv', help="Path to text file containing GPS timestamps")
    parser.add_argument('--sft_dir', type=str, default=None, help="Path or wildcard pattern to real SFTs (e.g. '/path/to/*.sft')")
    parser.add_argument('--ref_time', type=float, default=1372426000, help="Reference time for simulation (GPS)")
    parser.add_argument('--Tsft', type=float, default=1800, help="Duration of each SFT segment in seconds")
    parser.add_argument('--IFOS', nargs='+', default=['H1', 'L1'], help="List of interferometers to simulate (e.g. --IFOS H1 L1)")
    parser.add_argument('--h0', type=float, default=1e-25, help="Strain amplitude for the simulated signals")
    parser.add_argument('--sqrtSX', type=float, default=None, help="Noise PSD for Gaussian noise generation")
    parser.add_argument('--window_type', type=str, default='tukey', help="Window type for SFT generation (e.g. 'tukey')")
    parser.add_argument('--window_param', type=float, default=0.001, help="Window parameter for SFT generation (e.g. 0.001 for a very mild Tukey window)")
    parser.add_argument('--dt_wf', type=float, default=5.0, help="Time step for waveform generation in seconds")
    args = parser.parse_args() 

    # --- 1. Real Data vs Gaussian Noise ---
    if args.sft_dir is not None and args.sqrtSX is not None:
        raise ValueError("Conflict: Cannot provide both --sft_dir (real data) and --sqrtSX (Gaussian noise). Please choose one.")

    if args.sft_dir is None and args.sqrtSX is None:
        raise ValueError("Conflict: Cannot missing both --sft_dir (real data) and --sqrtSX (Gaussian noise). Please choose one.")
        
    # --- 2. Calculate Depth safely AFTER setting defaults ---
    if args.sqrtSX is not None:
        sqrtSX = args.sqrtSX
        depth = sqrtSX / args.h0
    else:
        # If using real SFTs, sqrtSX is None, so depth isn't strictly defined here
        sqrtSX = 0 
        print(f"Perform injections into real data.")
        depth = "N/A (Real Data)"
    
    obs_params = {
        'freq': args.freq,
        'sqrtSX': sqrtSX,
        'sft_dir': args.sft_dir,
        'h0': args.h0,
        'tref': args.ref_time,
        'dt_wf': args.dt_wf,
        'window_type': args.window_type,
        'window_param': args.window_param,
        'IFOS': args.IFOS,
        'Tsft': args.Tsft,

    }
    print(f"Observational Setup -> depth:{depth}, sqrtSX:{sqrtSX}, h0:{args.h0}")

    # load timestamps
    print(f"Loading timestamps from: {args.timestamps_file}")
    
    # Read the file (assuming comma-separated; if space-separated, add sep='\s+')
    ts_df = pd.read_csv(args.timestamps_file) 
    
    timestamps = {}
    for ifo in ts_df.columns:
        # Drop the NaNs (padding from unequal lengths) and convert to a numpy array
        timestamps[ifo] = ts_df[ifo].dropna().values
        
    for ifo, ts in timestamps.items():
        print(f"Loaded {len(ts)} timestamps for {ifo}")
    
    csv_path = f'/scratch/kriles_root/kriles0/damoncht/cwglitch/data/{args.label}/100-100Hz/signal_glitch_params.csv'
    save_path = f'/scratch/kriles_root/kriles0/damoncht/cwglitch/data/{args.label}/100-100Hz'
    
    print(f"Loading astrophysical parameters from: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}. Did you run the generation script first?")
        
    df = pd.read_csv(csv_path)
    
    main(timestamps, df, obs_params, save_path, n_cpu=args.n_cpu)