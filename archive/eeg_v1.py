import sys
import numpy as np
import matplotlib.pyplot as plt
import mne

# ============================================================================
# STEP 1: LOAD DATA FROM COMMAND LINE ARGUMENTS
# ============================================================================
print("Step 1: Loading EEG data...")

# Check command line arguments
if len(sys.argv) != 3:
    print("Error: Please provide two EDF file paths")
    print("Usage: python eeg_alpha_simple.py <eyes_open.edf> <eyes_closed.edf>")
    print("\nExample:")
    print("  python eeg_alpha_simple.py S001R01.edf S001R02.edf")
    sys.exit(1)

file_open = sys.argv[1]
file_closed = sys.argv[2]

print(f"  Eyes Open file:   {file_open}")
print(f"  Eyes Closed file: {file_closed}")

# Load into MNE (a library for EEG processing)
try:
    raw_open = mne.io.read_raw_edf(file_open, preload=True, verbose=False)
    raw_closed = mne.io.read_raw_edf(file_closed, preload=True, verbose=False)
except FileNotFoundError as e:
    print(f"\nError: Could not find file - {e}")
    sys.exit(1)
except Exception as e:
    print(f"\nError loading files: {e}")
    sys.exit(1)

print(f"  Loaded! {len(raw_open.ch_names)} channels, {raw_open.info['sfreq']} Hz sampling rate")

# ============================================================================
# STEP 2: FILTER THE DATA
# ============================================================================
print("\nStep 2: Filtering to remove noise...")

# Why filter?
# - Raw EEG has slow drifts, electrical noise, muscle artifacts
# - We only care about brain waves (1-50 Hz)

# Band-pass filter: keep frequencies between 1-50 Hz
raw_open.filter(l_freq=1.0, h_freq=50.0, verbose=False)
raw_closed.filter(l_freq=1.0, h_freq=50.0, verbose=False)

# Notch filter: remove 60 Hz power line noise
raw_open.notch_filter(freqs=60, verbose=False)
raw_closed.notch_filter(freqs=60, verbose=False)

print("  ✓ Filtered")

# ============================================================================
# STEP 3: SELECT OCCIPITAL CHANNELS
# ============================================================================
print("\nStep 3: Selecting channels over visual cortex...")

# Alpha waves are strongest at the back of the head (occipital cortex)
# This is where visual processing happens
occipital_channels = ['O1..', 'Oz..', 'O2..']

raw_open = raw_open.pick(occipital_channels)
raw_closed = raw_closed.pick(occipital_channels)

print(f"  Selected: {occipital_channels}")

# ============================================================================
# STEP 4: COMPUTE FREQUENCY SPECTRUM
# ============================================================================
print("\nStep 4: Computing frequency spectrum (Welch's method)...")

# Welch's method: Break signal into chunks, compute frequency content,
# then average. This gives us "power" at each frequency.

psd_open = raw_open.compute_psd(method='welch', fmin=1, fmax=50, verbose=False)
psd_closed = raw_closed.compute_psd(method='welch', fmin=1, fmax=50, verbose=False)

# Extract the data
power_open = psd_open.get_data()  # Shape: (channels, frequencies)
power_closed = psd_closed.get_data()
freqs = psd_open.freqs  # Frequency values (1, 2, 3, ... 50 Hz)

# Average across the 3 occipital channels
power_open_avg = power_open.mean(axis=0)
power_closed_avg = power_closed.mean(axis=0)

print(f"  ✓ Computed power spectrum from {freqs[0]:.1f} to {freqs[-1]:.1f} Hz")

# ============================================================================
# STEP 5: EXTRACT ALPHA POWER
# ============================================================================
print("\nStep 5: Measuring alpha power (8-13 Hz)...")

# Alpha band is 8-13 Hz
# Find which frequency indices correspond to this range
alpha_mask = (freqs >= 8) & (freqs <= 13)

# Average power in the alpha band
alpha_power_open = power_open_avg[alpha_mask].mean()
alpha_power_closed = power_closed_avg[alpha_mask].mean()

# Calculate the difference
increase_pct = ((alpha_power_closed - alpha_power_open) / alpha_power_open) * 100

print(f"\n  Eyes Open:   {alpha_power_open:.2e} μV²/Hz")
print(f"  Eyes Closed: {alpha_power_closed:.2e} μV²/Hz")
print(f"  Increase:    {increase_pct:.1f}%")

# ============================================================================
# STEP 6: VISUALIZE
# ============================================================================
print("\nStep 6: Creating plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Full spectrum
ax1.plot(freqs, power_open_avg, label='Eyes Open', linewidth=2, color='blue')
ax1.plot(freqs, power_closed_avg, label='Eyes Closed', linewidth=2, color='red')
ax1.axvspan(8, 13, alpha=0.3, color='yellow', label='Alpha Band (8-13 Hz)')
ax1.set_xlabel('Frequency (Hz)', fontsize=12)
ax1.set_ylabel('Power (μV²/Hz)', fontsize=12)
ax1.set_title('Power Spectrum - Occipital Channels', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Bar chart comparing alpha power
ax2.bar(['Eyes Open', 'Eyes Closed'], 
        [alpha_power_open, alpha_power_closed],
        color=['blue', 'red'], alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_ylabel('Alpha Power (μV²/Hz)', fontsize=12)
ax2.set_title('Alpha Band Power Comparison', fontsize=13, fontweight='bold')
ax2.text(0.5, max(alpha_power_open, alpha_power_closed) * 0.5,
         f'+{increase_pct:.1f}%', ha='center', fontsize=16, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
ax2.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
# plt.savefig('alpha_analysis.png', dpi=150, bbox_inches='tight')
# print("  ✓ Saved plot as 'alpha_analysis.png'")
plt.show()

# ============================================================================
# INTERPRETATION
# ============================================================================
print("\n" + "=" * 70)
print("WHAT DOES THIS MEAN?")
print("=" * 70)
print(f"""
Result: Alpha power increased by {increase_pct:.1f}% when eyes were closed.

Why does this happen?

1. EYES OPEN:
   - Your visual cortex is actively processing what you see
   - Neurons are busy and not synchronized
   - Less alpha rhythm → LOWER power

2. EYES CLOSED:
   - Visual cortex has nothing to process
   - Neurons "idle" together in rhythm
   - Strong alpha rhythm → HIGHER power

This is called "ALPHA BLOCKING" - alpha waves are blocked (suppressed) 
when you're using your visual cortex.

This is one of the most reliable findings in neuroscience!
""")

print("=" * 70)
print("SUMMARY OF WHAT WE DID:")
print("=" * 70)
print("""
1. Downloaded two 1-minute EEG recordings (eyes open vs closed)
2. Filtered out noise (kept 1-50 Hz, removed 60 Hz power line)
3. Focused on occipital (visual) brain region channels
4. Used Fourier transform (Welch method) to measure power at each frequency
5. Compared alpha power (8-13 Hz) between the two conditions
6. Found that alpha is stronger with eyes closed (as expected!)

KEY CONCEPTS:
- Frequency: How fast the brain waves oscillate (Hz = cycles per second)
- Power: How strong the oscillation is at each frequency
- Alpha waves: 8-13 Hz rhythm that's strong when visual cortex is idle
- Welch's method: A way to reliably estimate power at each frequency
""")