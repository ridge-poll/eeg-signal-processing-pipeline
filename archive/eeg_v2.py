import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
import argparse


# ================ Helper functions ================

def load_raw_files(edf_files):
    """Load EDF files into MNE Raw objects"""
    print("Step 1: Loading EEG data...")
    raw_files = []
    try:
        for file in edf_files:
            out = mne.io.read_raw_edf(file, preload=True, verbose=False)
            raw_files.append(out)
    except FileNotFoundError as e:
        print(f"\nError: Could not find file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError loading files: {e}")
        sys.exit(1)

    for file in raw_files:
        print(f"    Loaded! {len(file.ch_names)} channels, {file.info['sfreq']} Hz sampling rate")
    return raw_files


def filter_data(raw_files, lowcut, highcut):
    """Band-pass and notch filter the EEG signals."""
    print("\nStep 2: Filtering to remove noise...")
    for signal in raw_files:
        signal.filter(l_freq=1.0, h_freq=50.0, verbose=False)
        signal.notch_filter(freqs=60, verbose=False)
    print("  ✓ Filtered")


def compute_psd_all(raw_files):
    """Compute power spectral density for all raw files."""
    print("\nStep 4: Computing frequency spectrum (Welch's method)...")
    psd_list = []
    for raw in raw_files:
        psd = raw.compute_psd(method='welch', fmin=1, fmax=50, verbose=False)
        psd_list.append(psd)

    power_list = [psd.get_data() for psd in psd_list]
    freqs = psd_list[0].freqs
    avg_power_list = [p.mean(axis=0) for p in power_list]

    print(f"  ✓ Computed power spectrum from {freqs[0]:.1f} to {freqs[-1]:.1f} Hz")
    return avg_power_list, freqs


def interpret_band(user_input_band):
    """Interpret a user-specified EEG frequency band"""
    EEG_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
    "mu": ()
    }
    if len(user_input_band) == 1 and isinstance(user_input_band[0], str):
    # Single value, could be a named band
        band_name = user_input_band[0].lower()
        if band_name in EEG_BANDS:
            low, high = EEG_BANDS[band_name]
            band_label = band_name[0].upper() + band_name[1:]
        else:
            print(f"Error: unknown band name '{band_name}'")
            sys.exit(1)
    elif len(user_input_band) == 2:
        # Two numeric values: convert to floats
        try:
            band_label = "Custom"
            low, high = map(float, user_input_band)
        except ValueError:
            print("Error: --band must be either a known name or two numbers (low high)")
            sys.exit(1)
    else:
        print("Error: --band must be either a known name or two numbers (low high)")
        sys.exit(1)

    print(f"Using frequency band: {low:.1f}–{high:.1f} Hz")

    return low, high, band_label


def extract_band_power(avg_power_list, freqs, band):
    """Compute mean power over a band of interest"""
    print("\nStep 5: Measuring band power (8-13 Hz)...")
    low, high, label = band
    band_mask = (freqs >= low) & (freqs <= high)
    band_power_list = [p[band_mask].mean() for p in avg_power_list]
    return band_power_list


def plot_power_spectrum(avg_power_list, freqs, band, band_power_list, show_bar_plot):
    """Plot full spectrum with highlighted alpha band."""
    print("\nStep 6: Creating plot...")
    # db_power_list = [20 * np.log10(a) for a in avg_power_list]

    low, high, label = band
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Determine layout: 1 plot or 2 plots side-by-side
    if show_bar_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

    for i, power in enumerate(avg_power_list):
        ax1.plot(freqs,
                20 * np.log10(power),
                label=f'Signal {i+1}',
                linewidth=1.5,
                color = colors[i % len(colors)])
        

    ax1.axvspan(low, high, alpha=0.3, color='yellow', label=f"{label} Band ({low}-{high} Hz)")
    ax1.set_xlabel('Frequency (Hz)', fontsize=12)
    ax1.set_ylabel('Power (μV²/Hz)', fontsize=12)
    ax1.set_title('Power Spectrum - Occipital Channels', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if show_bar_plot:
        # Create bar graph
        labels = [f"Signal {i+1}" for i in range(len(band_power_list))]
        ax2.bar(labels, band_power_list, color='skyblue')

        # Axis labels and title
        ax2.set_ylabel("Average Power (µV²/Hz)")
        ax2.set_title("Average Power Across Band")
        ax2.grid(axis='y', alpha=0.3)


    plt.show()


# ================ Main ================

def main(*args, lowcut=1, highcut=40,channels=['O1..', 'Oz..', 'O2..'],
         band=["alpha"], plot_psd=True, show_bar=False):

    edf_files = args
    raw_files = load_raw_files(edf_files)
    filter_data(raw_files, lowcut, highcut)
    avg_power_list, freqs = compute_psd_all(raw_files)
    band_of_interest = interpret_band(band)
    band_power_list = extract_band_power(avg_power_list, freqs, band_of_interest)
    if (plot_psd):
        plot_power_spectrum(avg_power_list, freqs, band_of_interest, band_power_list, show_bar)


if __name__ == "__main__":
    # main("physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf",
    #      "physionet.org/files/eegmmidb/1.0.0/S001/S001R02.edf",
    #      "physionet.org/files/eegmmidb/1.0.0/S001/S001R03.edf",
    #      show_bar=True)
    raw = mne.io.read_raw_edf("S001/S001R03.edf", preload=True)
    raw.plot()
    plt.show()
    pass
