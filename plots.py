import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_glitch_analysis(labels, plot_labels, durations=['5d', '10d', '20d'], n_glitch=1):
    """
    Create a 4x4 subplot grid comparing glitch analysis results.
    
    Parameters:
    -----------
    title_name : str
        Main title for the plot
    label : str
        Labels for the 'With Glitch' data in scatter plots
    glitch_path : str
        Path component for the glitch data files
    durations : list
        List of duration strings (default: ['5d', '10d', '20d', '50d'])
    n_glitch : int
        Number of glitches per signal (default: 1)
    """
    
    # Load parameter data
    df = pd.read_csv(f'./data/{labels[1]}_v2/100-100Hz/signal_glitch_params.csv')

    # Extract parameters
    f0 = df['f0'][::n_glitch].values
    f1 = df['f1'][::n_glitch].values
    f2 = df['f2'][::n_glitch].values
    df_p = df['df_p'].values
    df_t = df['df_t'].values
    df1_p = df['df1_p'].values
    q = df['Q'].values
    t_glitch = df['tglitch_day'].values

    # Initialize figure with 4x4 subplots
    fig, ax = plt.subplots(4, 4, figsize=(24, 18))

    # Loop over durations
    for idx, duration in enumerate(durations):
        # Initialize arrays
        a = np.zeros(32)  # No Glitch
        b = np.zeros(32)  # With Glitch
        c = np.zeros(32)  # No Glitch Diff Noise
        f0_list = np.zeros((32, 2))
        f1_list = np.zeros((32, 2))
        f2_list = np.zeros((32, 2))

        # Load data
        for i in range(32):
            d1 = fits.open(f'./results/{duration}/{labels[0]}_v2/100-100Hz/{labels[0]}_v2_CW{i}.fts')
            d2 = fits.open(f'./results/{duration}/{labels[1]}_v2/100-100Hz/{labels[1]}_v2_CW{i}.fts')
            d3 = fits.open(f'./results/{duration}/{labels[2]}_v2/100-100Hz/{labels[2]}_v2_CW{i}.fts')

            a[i] = d1[1].data['mean2F'][0]
            b[i] = d2[1].data['mean2F'][0]
            c[i] = d3[1].data['mean2F'][0]

            f0_list[i, 0] = d1[1].data['freq'][0]
            f1_list[i, 0] = d1[1].data['f1dot'][0]
            f2_list[i, 0] = d1[1].data['f2dot'][0]

            f0_list[i, 1] = d2[1].data['freq'][0]
            f1_list[i, 1] = d2[1].data['f1dot'][0]
            f2_list[i, 1] = d2[1].data['f2dot'][0]

        # Scatter plot: No Glitch vs With Glitch
        ax[idx, 0].scatter(t_glitch, a, label=plot_labels[0], alpha=0.8)
        ax[idx, 0].scatter(t_glitch, b, label=plot_labels[1], alpha=0.8)
        ax[idx, 0].set_xlabel('t_glitch (days)')
        ax[idx, 0].set_ylabel(r'$\mathcal{F}$')
        ax[idx, 0].grid(True)
        ax[idx, 0].set_title(f'{plot_labels[0]} vs {plot_labels[1]} ({duration})')
        
        for i in range(len(t_glitch)):
            ax[idx, 0].annotate('', xy=(t_glitch[i], b[i]), xytext=(t_glitch[i], a[i]), 
                               arrowprops=dict(arrowstyle='->', color='k', alpha=0.8))
        
        # Scatter plot: No Glitch vs Diff Noise
        ax[idx, 1].scatter(t_glitch, a, label=plot_labels[0], alpha=0.8)
        ax[idx, 1].scatter(t_glitch, c, label=plot_labels[2], alpha=0.8)
        ax[idx, 1].set_xlabel('t_glitch (days)')
        ax[idx, 1].set_ylabel(r'$\mathcal{F}$')
        ax[idx, 1].grid(True)
        ax[idx, 1].set_title(f'{plot_labels[0]} vs {plot_labels[2]} ({duration})')
        
        for i in range(len(t_glitch)):
            ax[idx, 1].annotate('', xy=(t_glitch[i], c[i]), xytext=(t_glitch[i], a[i]), 
                               arrowprops=dict(arrowstyle='->', color='k', alpha=0.8))

        # Histogram: Ratio (With Glitch / No Glitch)
        ratios = b / a
        ax[idx, 2].hist(ratios, bins=10, color='teal', edgecolor='black', density=True)
        ax[idx, 2].set_xlabel(f'Ratio ({plot_labels[1]} / {plot_labels[0]})')
        ax[idx, 2].set_ylabel('Density')
        ax[idx, 2].grid(True)
        ax[idx, 2].set_title(f'Histogram of Ratio ({duration})')
        
        # Histogram: Ratio (Diff Noise)
        ratios = c / a
        ax[idx, 3].hist(ratios, bins=10, color='purple', edgecolor='black', density=True)
        ax[idx, 3].set_xlabel(f'Ratio ({plot_labels[2]} / {plot_labels[0]})')
        ax[idx, 3].set_ylabel('Density')
        ax[idx, 3].grid(True)
        ax[idx, 3].set_title(f'Histogram of Ratio ({duration})')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.show()
    
def plot_glitch_ratio_analysis(labels, plot_labels, durations=['5d', '10d', '20d', '50d'], n_glitch=1):
    """
    Create a 4x4 subplot grid comparing glitch analysis results with ratios.
    
    Parameters:
    -----------
    labels : list
        List of path components for data files ['no_glitch', 'with_glitch', ...]
    plot_labels : list
        List of labels for plots ['No Glitch', 'With Glitch', ...]
    durations : list
        List of duration strings (default: ['5d', '10d', '20d', '50d'])
    n_glitch : int
        Number of glitches per signal (default: 1)
    """
    
    # Load parameter data
    df = pd.read_csv(f'./data/{labels[1]}/100-100Hz/signal_glitch_params.csv')

    # Extract parameters
    f0 = df['f0'][::n_glitch].values
    f1 = df['f1'][::n_glitch].values
    f2 = df['f2'][::n_glitch].values
    df_p = df['df_p'].values
    df_t = df['df_t'].values
    df1_p = df['df1_p'].values
    q = df['Q'].values
    t_glitch = df['tglitch_day'].values

    # Initialize previous arrays
    prev_a = np.ones(32)  # No Glitch
    prev_b = np.ones(32)  # With Glitch

    # Initialize figure with 4x4 subplots
    fig, ax = plt.subplots(4, 4, figsize=(24, 18))

    # Loop over durations
    for idx, duration in enumerate(durations):
        # Initialize arrays
        a = np.zeros(32)  # No Glitch
        b = np.zeros(32)  # With Glitch
        f0_list = np.zeros((32, 2))
        f1_list = np.zeros((32, 2))
        f2_list = np.zeros((32, 2))

        # Load data
        for i in range(32):
            d1 = fits.open(f'./results/{duration}/{labels[0]}_v2/100-100Hz/{labels[0]}_v2_CW{i}.fts')
            d2 = fits.open(f'./results/{duration}/{labels[1]}_v2/100-100Hz/{labels[1]}_v2_CW{i}.fts')
            
            a[i] = d1[1].data['mean2F'][0]
            b[i] = d2[1].data['mean2F'][0]

            f0_list[i, 0] = d1[1].data['freq'][0]
            f1_list[i, 0] = d1[1].data['f1dot'][0]
            f2_list[i, 0] = d1[1].data['f2dot'][0]

            f0_list[i, 1] = d2[1].data['freq'][0]
            f1_list[i, 1] = d2[1].data['f1dot'][0]
            f2_list[i, 1] = d2[1].data['f2dot'][0]

        # Scatter plot: No Glitch vs With Glitch
        ax[idx, 0].scatter(t_glitch, a, label=plot_labels[0], alpha=0.8)
        ax[idx, 0].scatter(t_glitch, b, label=plot_labels[1], alpha=0.8)
        ax[idx, 0].set_xlabel('t_glitch (days)')
        ax[idx, 0].set_ylabel(r'$\mathcal{F}$')
        ax[idx, 0].grid(True)
        ax[idx, 0].set_title(f'{plot_labels[0]} vs {plot_labels[1]} ({duration})')
        
        for i in range(len(t_glitch)):
            ax[idx, 0].annotate('', xy=(t_glitch[i], b[i]), xytext=(t_glitch[i], a[i]), 
                               arrowprops=dict(arrowstyle='->', color='k', alpha=0.8))
        
        # Scatter plot: Ratio (No Glitch vs With Glitch)
        ax[idx, 1].scatter(t_glitch, a/a, label=plot_labels[0], alpha=0.8)
        ax[idx, 1].scatter(t_glitch, b/a, label=plot_labels[1], alpha=0.8)
        ax[idx, 1].set_xlabel('t_glitch (days)')
        ax[idx, 1].set_ylabel(r'$\mathcal{F}$')
        ax[idx, 1].grid(True)
        ax[idx, 1].set_title(f'Ratio ({duration})')
        
        for i in range(len(t_glitch)):
            ax[idx, 1].annotate('', xy=(t_glitch[i], b[i]/a[i]), xytext=(t_glitch[i], a[i]/a[i]), 
                               arrowprops=dict(arrowstyle='->', color='k', alpha=0.8))

        if idx != 0:
            # Histogram: Ratio Increase (No Glitch)
            ratios = a / prev_a
            ax[idx, 2].hist(ratios, bins=10, color='teal', edgecolor='black', density=True)
            ax[idx, 2].set_xlabel(f'Ratio Increase ({plot_labels[0]})')
            ax[idx, 2].set_ylabel('Density')
            ax[idx, 2].grid(True)
            ax[idx, 2].set_title(f'Histogram of Ratio ({duration})')
            
            # Histogram: Ratio Increase (With Glitch)
            ratios = b / prev_b
            ax[idx, 3].hist(ratios, bins=10, color='purple', edgecolor='black', density=True)
            ax[idx, 3].set_xlabel(f'Ratio Increase ({plot_labels[1]})')
            ax[idx, 3].set_ylabel('Density')
            ax[idx, 3].grid(True)
            ax[idx, 3].set_title(f'Histogram of Ratio ({duration})')

        prev_a, prev_b = a, b

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_bias(labels, plot_labels, durations=['5d', '10d', '20d', '50d'], n_glitch=1, markers=['<', 's', 'o']):
    """
    Create a 4x3 subplot grid comparing frequency parameters for glitch analysis.
    
    Parameters:
    -----------
    labels : list
        List of path components for data files ['no_glitch', 'with_glitch', 'no_glitch_diffnoise']
    plot_labels : list
        List of labels for plots ['No Glitch', 'With Glitch', 'No Glitch Noise']
    durations : list
        List of duration strings (default: ['5d', '10d', '20d', '50d'])
    n_glitch : int
        Number of glitches per signal (default: 1)
    markers : list
        List of markers for scatter plots (default: ['o', '<', '>'])
    colors : list
        List of colors for scatter plots (default: ['blue', 'red', 'green'])
    """
    
    # Load parameter data
    df = pd.read_csv(f'./data/{labels[1]}_v2/100-100Hz/signal_glitch_params.csv')

    # Extract parameters
    f0 = df['f0'][::n_glitch].values
    f1 = df['f1'][::n_glitch].values
    f2 = df['f2'][::n_glitch].values
    df_p = df['df_p'].values
    df_t = df['df_t'].values
    df1_p = df['df1_p'].values
    q = df['Q'].values
    t_glitch = df['tglitch_day'].values

    # Initialize figure with 4x3 subplots
    fig, ax = plt.subplots(4, 3, figsize=(24, 18))
    # Loop over durations
    for idx, duration in enumerate(durations):
        # Initialize arrays
        f0_list = np.zeros((32, 3))
        f1_list = np.zeros((32, 3))
        f2_list = np.zeros((32, 3))

        # Load data
        for i in range(32):
            d1 = fits.open(f'./results/{duration}/{labels[0]}_v2/100-100Hz/{labels[0]}_v2_CW{i}.fts')
            d2 = fits.open(f'./results/{duration}/{labels[1]}_v2/100-100Hz/{labels[1]}_v2_CW{i}.fts')
            d3 = fits.open(f'./results/{duration}/{labels[2]}_v2/100-100Hz/{labels[2]}_v2_CW{i}.fts')

            f0_list[i, 0] = d1[1].data['freq'][0]
            f1_list[i, 0] = d1[1].data['f1dot'][0]
            f2_list[i, 0] = d1[1].data['f2dot'][0]

            f0_list[i, 1] = d2[1].data['freq'][0]
            f1_list[i, 1] = d2[1].data['f1dot'][0]
            f2_list[i, 1] = d2[1].data['f2dot'][0]
            
            f0_list[i, 2] = d3[1].data['freq'][0]
            f1_list[i, 2] = d3[1].data['f1dot'][0]
            f2_list[i, 2] = d3[1].data['f2dot'][0]

        # Data and truth arrays for plotting
        data = [f0_list, f1_list, f2_list]
        truth = [f0, f1, f2]

        # Plot for each frequency parameter (f0, f1, f2)
        for plot_i in range(3):
            for j in range(3):
                # Scatter plot: Difference from truth
                ax[idx, plot_i].scatter(t_glitch, (data[plot_i][:, j] - truth[plot_i]), 
                                       color=colors[j], label=plot_labels[j], 
                                       marker=markers[j], facecolors='none', s=30*2*((j+1)))
            ax[idx, plot_i].hlines([0], 0, 100, color="k")
            ax[idx, plot_i].set_xlabel('t_glitch (days)')
            ax[idx, plot_i].set_ylabel(f'f{plot_i}')
            ax[idx, plot_i].legend()
            ax[idx, plot_i].grid(True)
            ax[idx, plot_i].set_title(f'({duration})')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_bias_ab(labels, plot_labels, durations=['5d', '10d', '20d', '50d'], n_glitch=1, markers=['<', 's', 'o']):
    """
    Create a 4x3 subplot grid comparing frequency parameters for glitch analysis.
    
    Parameters:
    -----------
    labels : list
        List of path components for data files ['no_glitch', 'with_glitch', 'no_glitch_diffnoise']
    plot_labels : list
        List of labels for plots ['No Glitch', 'With Glitch', 'No Glitch Noise']
    durations : list
        List of duration strings (default: ['5d', '10d', '20d', '50d'])
    n_glitch : int
        Number of glitches per signal (default: 1)
    markers : list
        List of markers for scatter plots (default: ['o', '<', '>'])
    colors : list
        List of colors for scatter plots (default: ['blue', 'red', 'green'])
    """
    
    # Load parameter data
    df = pd.read_csv(f'./data/{labels[1]}_v2/100-100Hz/signal_glitch_params.csv')

    # Extract parameters
    f0 = df['f0'][::n_glitch].values
    f1 = df['f1'][::n_glitch].values
    f2 = df['f2'][::n_glitch].values
    df_p = df['df_p'].values
    df_t = df['df_t'].values
    df1_p = df['df1_p'].values
    q = df['Q'].values
    t_glitch = df['tglitch_day'].values

    # Initialize figure with 4x3 subplots
    fig, ax = plt.subplots(4, 3, figsize=(24, 18))
    # Loop over durations
    for idx, duration in enumerate(durations):
        # Initialize arrays
        f0_list = np.zeros((32, 2))
        f1_list = np.zeros((32, 2))
        f2_list = np.zeros((32, 2))

        # Load data
        for i in range(32):
            d1 = fits.open(f'./results/{duration}/{labels[0]}_v2/100-100Hz/{labels[0]}_v2_CW{i}.fts')
            d2 = fits.open(f'./results/{duration}/{labels[1]}_v2/100-100Hz/{labels[1]}_v2_CW{i}.fts')

            f0_list[i, 0] = d1[1].data['freq'][0]
            f1_list[i, 0] = d1[1].data['f1dot'][0]
            f2_list[i, 0] = d1[1].data['f2dot'][0]

            f0_list[i, 1] = d2[1].data['freq'][0]
            f1_list[i, 1] = d2[1].data['f1dot'][0]
            f2_list[i, 1] = d2[1].data['f2dot'][0]

        # Data and truth arrays for plotting
        data = [f0_list, f1_list, f2_list]
        truth = [f0, f1, f2]

        # Plot for each frequency parameter (f0, f1, f2)
        for plot_i in range(3):
            for j in range(2):
                # Scatter plot: Difference from truth
                ax[idx, plot_i].scatter(t_glitch, (data[plot_i][:, j] - truth[plot_i]), 
                                       color=colors[j], label=plot_labels[j], 
                                       marker=markers[j], facecolors='none', s=30*2*((j+1)))
            ax[idx, plot_i].hlines([0], 0, 100, color="k")
            ax[idx, plot_i].set_xlabel('t_glitch (days)')
            ax[idx, plot_i].set_ylabel(f'f{plot_i}')
            ax[idx, plot_i].legend()
            ax[idx, plot_i].grid(True)
            ax[idx, plot_i].set_title(f'({duration})')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


