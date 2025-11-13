import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
import argparse
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


# ================ Constants ================

STANDARD_ELECTRODES = [
    'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.',
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
    'Iz..'
]

EEG_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}


# ================ Data Classes ================

@dataclass
class FrequencyBand:
    """Represents a frequency band of interest."""
    low: float
    high: float
    label: str
    
    @classmethod
    def from_input(cls, user_input: List) -> 'FrequencyBand':
        """Create a FrequencyBand from user input (name or two numbers)."""
        if len(user_input) == 1 and isinstance(user_input[0], str):
            band_name = user_input[0].lower()
            if band_name in EEG_BANDS:
                low, high = EEG_BANDS[band_name]
                label = band_name.capitalize()
            else:
                raise ValueError(f"Unknown band name '{band_name}'. "
                               f"Available bands: {list(EEG_BANDS.keys())}")
        elif len(user_input) == 2:
            try:
                low, high = map(float, user_input)
                label = "Custom"
            except ValueError:
                raise ValueError("Band must be either a known name or two numbers (low high)")
        else:
            raise ValueError("Band must be either a known name or two numbers (low high)")
        
        print(f"Using frequency band: {low:.1f}-{high:.1f} Hz")
        return cls(low, high, label)


@dataclass
class PSDResults:
    """Stores power spectral density analysis results."""
    avg_power_list: List[np.ndarray]
    freqs: np.ndarray
    band_power_list: List[float]
    band: FrequencyBand
    channels: List[str]


# ================ Core Classes ================

class EEGDataLoader:
    """Handles loading and initial validation of EEG data files."""
    
    def __init__(self, edf_files: List[str]):
        self.edf_files = edf_files
        self.raw_files: List[mne.io.Raw] = []
    
    def load(self) -> List[mne.io.Raw]:
        """Load EDF files into MNE Raw objects."""
        print("Step 1: Loading EEG data...")
        try:
            for file in self.edf_files:
                raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
                self.raw_files.append(raw)
                print(f"    Loaded! {len(raw.ch_names)} channels, "
                      f"{raw.info['sfreq']} Hz sampling rate")
        except FileNotFoundError as e:
            print(f"\nError: Could not find file - {e}")
            sys.exit(1)
        except Exception as e:
            print(f"\nError loading files: {e}")
            sys.exit(1)
        
        return self.raw_files


class EEGPreprocessor:
    """Handles preprocessing steps for EEG data."""
    
    def __init__(self, raw_files: List[mne.io.Raw]):
        self.raw_files = raw_files
    
    def filter(self, lowcut: float = 1.0, highcut: float = 50.0, 
               notch_freq: float = 60.0) -> None:
        """Apply band-pass and notch filtering to remove noise."""
        print("\nStep 2: Filtering to remove noise...")
        for signal in self.raw_files:
            signal.filter(l_freq=lowcut, h_freq=highcut, verbose=False)
            signal.notch_filter(freqs=notch_freq, verbose=False)
        print("  ✓ Filtered")


class PSDAnalyzer:
    """Computes power spectral density analysis."""
    
    def __init__(self, raw_files: List[mne.io.Raw], 
                 channels: Optional[List[str]] = None,
                 event_ids: Optional[List] = None):
        self.raw_files = raw_files
        self.channels = channels if channels else STANDARD_ELECTRODES
        self.event_ids = event_ids
        self._validate_channels()
    
    def _validate_channels(self) -> None:
        """Ensure all requested channels are valid."""
        for channel in self.channels:
            if channel not in STANDARD_ELECTRODES:
                raise ValueError(f"Invalid channel name: '{channel}'. "
                               f"Must be one of {STANDARD_ELECTRODES}")
    
    def _resolve_event_ids(self, events: np.ndarray, 
                          event_id: Dict[str, int], file_index: int) -> List[int]:
        """Convert user event specification to numeric event IDs.
        Raises an error if any requested event is not found in this file."""
        if not self.event_ids:
            return list(event_id.values())
        
        selected_ids = []
        missing_events = []
        
        for e in self.event_ids:
            if isinstance(e, str) and e in event_id:
                selected_ids.append(event_id[e])
            elif isinstance(e, int) and e in event_id.values():
                selected_ids.append(e)
            else:
                missing_events.append(e)
        
        if missing_events:
            print(f"\nError in file {file_index + 1}: Event(s) {missing_events} not found.")
            print(f"   Available events in this file: {list(event_id.keys())}")
            raise ValueError(f"Event(s) {missing_events} not found in file {file_index + 1}. "
                           f"All specified events must be present in all files.")
        
        return selected_ids
    
    def compute_psd(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """Compute power spectral density for all raw files."""
        print("\nStep 4: Computing frequency spectrum (Welch's method)...")
        
        # First pass: check all files have the requested events
        if self.event_ids:
            print("  Validating events across all files...")
            for i, raw in enumerate(self.raw_files):
                events, event_id = mne.events_from_annotations(raw)
                try:
                    self._resolve_event_ids(events, event_id, i)
                except ValueError:
                    # Show events available in ALL files to help user
                    print("\n  Events available across all files:")
                    self._show_common_events()
                    raise
        
        # Second pass: actually compute PSD
        psd_list = []
        for i, raw in enumerate(self.raw_files):
            raw_region = raw.copy().pick(self.channels)
            events, event_id = mne.events_from_annotations(raw)
            
            selected_ids = self._resolve_event_ids(events, event_id, i)
                
            print(f"  File {i + 1}: Using event IDs {selected_ids} "
                  f"({[k for k, v in event_id.items() if v in selected_ids]})")
            
            epochs = mne.Epochs(raw_region, events, selected_ids,
                               tmin=0, tmax=4, baseline=None, preload=True)
            psd = epochs.compute_psd(method='welch', fmin=1, fmax=50, verbose=False)
            psd_list.append(psd)
        
        power_list = [psd.get_data() for psd in psd_list]
        freqs = psd_list[0].freqs
        avg_power_list = [p.mean(axis=0) for p in power_list]
        
        print(f"  ✓ Computed power spectrum from {freqs[0]:.1f} to {freqs[-1]:.1f} Hz")
        return avg_power_list, freqs
    
    def _show_common_events(self) -> None:
        """Display which events are available in each file."""
        all_events = []
        for i, raw in enumerate(self.raw_files):
            events, event_id = mne.events_from_annotations(raw)
            all_events.append(set(event_id.keys()))
            print(f"     File {i + 1}: {sorted(event_id.keys())}")
        
        # Show intersection
        common = set.intersection(*all_events) if all_events else set()
        if common:
            print(f"  Common to all files: {sorted(common)}")
        else:
            print("  No events are common to all files!")
    
    def extract_band_power(self, avg_power_list: List[np.ndarray], 
                          freqs: np.ndarray, band: FrequencyBand) -> List[float]:
        """Compute mean power over a specific frequency band."""
        print(f"\nStep 5: Measuring band power ({band.low}-{band.high} Hz)...")
        band_mask = (freqs >= band.low) & (freqs <= band.high)
        band_power_list = [p[:, band_mask].mean(axis=1) for p in avg_power_list]
        band_power_avg = [bp.mean() for bp in band_power_list]
        print(band_power_avg)
        return band_power_avg


class EEGVisualizer:
    """Creates visualizations for EEG analysis results."""
    
    COLOR_PALETTE = ['blue', 'red', 'green', 'orange', 'purple', 
                     'brown', 'pink', 'gray', 'olive', 'cyan']
    
    @staticmethod
    def create_spectrum_plot(psd_results: PSDResults, 
                            show_bar_plot: bool = False) -> plt.Figure:
        """Create power spectrum visualization with optional bar chart."""
        print("\nStep 6: Creating plot...")
        
        if show_bar_plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot power spectrum
        for i, power in enumerate(psd_results.avg_power_list):
            result_numpy = power.sum(axis=0)
            ax1.plot(psd_results.freqs,
                    20 * np.log10(result_numpy),
                    label=f'Signal {i+1}',
                    linewidth=1.5,
                    color=EEGVisualizer.COLOR_PALETTE[i % len(EEGVisualizer.COLOR_PALETTE)])
        
        # Highlight frequency band
        band = psd_results.band
        ax1.axvspan(band.low, band.high, alpha=0.3, color='yellow', 
                   label=f"{band.label} Band ({band.low}-{band.high} Hz)")
        ax1.set_xlabel('Frequency (Hz)', fontsize=12)
        ax1.set_ylabel('Power (μV²/Hz)', fontsize=12)
        channels_str = ', '.join(psd_results.channels)
        ax1.set_title(f'Power Spectrum - Channels: {channels_str}', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Optional bar chart
        if show_bar_plot:
            labels = [f"Signal {i+1}" for i in range(len(psd_results.band_power_list))]
            ax2.bar(labels, psd_results.band_power_list, color='skyblue')
            ax2.set_ylabel("Average Power (µV²/Hz)")
            ax2.set_title("Average Power Across Band")
            ax2.grid(axis='y', alpha=0.3)
        
        return fig
    
    @staticmethod
    def display_figure(fig: plt.Figure, save_path: Optional[str] = None) -> None:
        """Save and/or display the figure."""
        if save_path:
            fig.savefig(save_path)
            print(f"  ✓ Saved to {save_path}")
        plt.show()


class EEGAnalysisPipeline:
    """Orchestrates the complete EEG analysis workflow."""
    
    def __init__(self, edf_files: List[str], channels: Optional[List[str]] = None,
                 event_ids: Optional[List] = None):
        self.edf_files = edf_files
        self.channels = channels
        self.event_ids = event_ids
        self.raw_files: Optional[List[mne.io.Raw]] = None
        self.psd_results: Optional[PSDResults] = None
    
    def run(self, lowcut: float = 1.0, highcut: float = 50.0,
            band_input: List = None, show_bar_plot: bool = False,
            save_path: Optional[str] = None) -> PSDResults:
        """Execute the complete analysis pipeline."""
        # Load data
        loader = EEGDataLoader(self.edf_files)
        self.raw_files = loader.load()
        
        # Preprocess
        preprocessor = EEGPreprocessor(self.raw_files)
        preprocessor.filter(lowcut, highcut)
        
        # Analyze
        analyzer = PSDAnalyzer(self.raw_files, self.channels, self.event_ids)
        avg_power_list, freqs = analyzer.compute_psd()
        
        # Extract band power
        band = FrequencyBand.from_input(band_input if band_input else ["alpha"])
        band_power_list = analyzer.extract_band_power(avg_power_list, freqs, band)
        
        # Store results
        # self.psd_results = PSDResults(avg_power_list, freqs, band_power_list, band)
        self.psd_results = PSDResults(avg_power_list, freqs, band_power_list, band, self.channels or STANDARD_ELECTRODES)
        
        # Visualize
        visualizer = EEGVisualizer()
        fig = visualizer.create_spectrum_plot(self.psd_results, show_bar_plot)
        visualizer.display_figure(fig, save_path)
        
        return self.psd_results


# ================ CLI Interface ================

def main():
    parser = argparse.ArgumentParser(description="EEG PSD analyzer with OOP design")
    parser.add_argument("edf_files", nargs="+", help="EDF files to process")
    parser.add_argument("--lowcut", type=float, default=1.0,
                       help="High-pass filter cutoff (Hz)")
    parser.add_argument("--highcut", type=float, default=50.0,
                       help="Low-pass filter cutoff (Hz)")
    parser.add_argument("--channels", nargs="+", default=None,
                       help="Channels to analyze (default: all)")
    parser.add_argument("--epochs", nargs="+", default=None,
                       help="Event IDs to include (e.g., T0 T2 T3); default: all")
    parser.add_argument("--band", nargs="+", default=["alpha"],
                       help="Frequency band: name (alpha, beta, etc.) or two numbers")
    parser.add_argument("--show_bar", action="store_true",
                       help="Show bar plot comparing band power across signals")
    parser.add_argument("--save_fig", type=str, default=None,
                       help="""Save figure to file (e.g., image.png) Naming convention:
                       <subject>_<channels>_<band>_<epochs>_<runs>_<task-description>.png
                       (e.g. S001_C4Fc4_mu_T2_R0304_rightfist-real-imag.png)""")
    
    args = parser.parse_args()
    
    try:
        pipeline = EEGAnalysisPipeline(
            edf_files=args.edf_files,
            channels=args.channels,
            event_ids=args.epochs
        )
        
        pipeline.run(
            lowcut=args.lowcut,
            highcut=args.highcut,
            band_input=args.band,
            show_bar_plot=args.show_bar,
            save_path=args.save_fig
        )
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()