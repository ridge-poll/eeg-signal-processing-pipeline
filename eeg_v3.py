import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
import argparse


# ================ Helper functions ================

def load_raw_files(edf_files):
    """Load EDF files into MNE Raw objects."""
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
        signal.filter(l_freq=lowcut, h_freq=highcut, verbose=False)
        signal.notch_filter(freqs=60, verbose=False)
    print("  ✓ Filtered")


def compute_psd_all(raw_files, channels, event_id_list):
    """Compute power spectral density for all raw files."""
    print("\nStep 4: Computing frequency spectrum (Welch's method)...")
    electrodes = ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.',
                  'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..',
                  'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.',
                  'Fp1.', 'Fpz.', 'Fp2.',
                  'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.',
                  'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..',
                  'Ft7.', 'Ft8.',
                  'T7..', 'T8..', 'T9..', 'T10.',
                  'Tp7.', 'Tp8.',
                  'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..',
                  'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.',
                  'O1..', 'Oz..', 'O2..',
                  'Iz..']
    
    if not channels:
        channels = electrodes # no user input, use all channels
    for c in channels:
        if c not in electrodes:
            raise ValueError(f"Invalid channel name: '{c}'. Must be one of {electrodes}")
    

    psd_list = []
    for raw in raw_files:
        raw_region = raw.copy().pick(channels)
        events, event_id = mne.events_from_annotations(raw)

        # Handle flexible user input (names or numeric codes)
        if event_id_list:
            selected_ids = []
            for e in event_id_list:
                if isinstance(e, str) and e in event_id:
                    selected_ids.append(event_id[e])
                elif isinstance(e, int) and e in event_id.values():
                    selected_ids.append(e)
                else:
                    print(f"Invalid event '{e}'. Available events:")
                    for k, v in event_id.items():
                        print(f"  {k} ({v})")
                    raise ValueError(f"Event {e} not found.")
        else:
            selected_ids = list(event_id.values())  # default = all events
        print(f"  Using event IDs {selected_ids} ({[k for k,v in event_id.items() if v in selected_ids]})")

        epochs = mne.Epochs(raw_region, events, selected_ids,
                            tmin=0, tmax=4, baseline=None,
                            preload=True)
        psd = epochs.compute_psd(method='welch', fmin=1, fmax=50, verbose=False)
        psd_list.append(psd)

    power_list = [psd.get_data() for psd in psd_list]
    freqs = psd_list[0].freqs
    avg_power_list = [p.mean(axis=0) for p in power_list]


    print(f"  ✓ Computed power spectrum from {freqs[0]:.1f} to {freqs[-1]:.1f} Hz")
    return avg_power_list, freqs


def interpret_band(user_input_band):
    EEG_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45)
    }
    if len(user_input_band) == 1 and isinstance(user_input_band[0], str):
    # Single value: could be a named band
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
    """Compute mean power over a band of interest."""
    low, high, label = band
    print(f"\nStep 5: Measuring band power ({low}-{high} Hz)...")
    band_mask = (freqs >= low) & (freqs <= high)
    band_power_list = [p[:, band_mask].mean(axis=1) for p in avg_power_list]
    band_power_avg = [bp.mean() for bp in band_power_list]
    print(band_power_avg)
    return band_power_avg


def plot_power_spectrum(avg_power_list, freqs, band, band_power_list, show_bar_plot, save_fig):
    """Plot full spectrum with highlighted alpha band."""
    print("\nStep 6: Creating plot...")

    low, high, label = band
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Determine layout: 1 plot or 2 plots side-by-side
    if show_bar_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))


    for i, power in enumerate(avg_power_list):

        result_numpy = power.sum(axis=0)
        power = result_numpy

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

    if save_fig:
        plt.savefig(save_fig)
    plt.show()



# ================ Main ================

def main():
    parser = argparse.ArgumentParser(description="EEG PSD plotter")
    parser.add_argument("edf_files", nargs="+", help="EDF files to process")# positional argument
    parser.add_argument("--lowcut", type=float, default=1.0)
    parser.add_argument("--highcut", type=float, default=50.0)
    parser.add_argument("--channels", nargs="+", default=[])
    parser.add_argument("--epochs", nargs="+", default=[],
        help="Event IDs to include (e.g. --epochs T0 T2 T3); default includes all")
    parser.add_argument("--band", nargs="+", default=["alpha"],
        help="Frequency band: name (alpha, beta, etc.) or two numbers (e.g. --band 12.5 16)")
    parser.add_argument("--show_bar", action="store_true", default=False,
        help="Show bar plot comparing band power across signals")
    parser.add_argument("--save_fig", type=str, default="",
                        help="Enter name for saved file (e.g. --save_fig image.png)")
    args = parser.parse_args()

    edf_files = args.edf_files
    raw_files = load_raw_files(edf_files)
    filter_data(raw_files, args.lowcut, args.highcut)
    avg_power_list, freqs = compute_psd_all(raw_files, args.channels, args.epochs)
    band_of_interest = interpret_band(args.band)
    band_power_list = extract_band_power(avg_power_list, freqs, band_of_interest)
    plot_power_spectrum(avg_power_list, freqs, band_of_interest, band_power_list, args.show_bar, args.save_fig)


if __name__ == "__main__":
    main()
