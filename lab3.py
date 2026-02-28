import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk 

fs = 1000          
duration = 10      

ecg = nk.ecg_simulate(duration=duration, sampling_rate=fs, heart_rate=70)

# 2.2.3 Construct time axis
t = np.arange(len(ecg)) / fs

plt.figure(figsize=(10, 4))
plt.plot(t, ecg)
plt.title("Clean Synthetic ECG Signal (Time Domain)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# section 3.1.1 Downsample the ECG signal
rates = [1000, 500, 250, 125]

fig, axs = plt.subplots(3, 2, figsize=(12, 10))
axs = axs.ravel()

for r in rates:
    s = ecg if r == fs else ecg[::(fs // r)]
    t_r = np.arange(len(s)) / r
    m = t_r <= 3
    axs[0].plot(t_r[m], s[m], label=f"{r} Hz")

axs[0].set_title("ECG at Different Sampling Rates (First 3 seconds)")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Amplitude")
axs[0].legend()
axs[0].grid(True)

for i, r in enumerate(rates, start=1):
    s = ecg if r == fs else ecg[::(fs // r)]
    t_r = np.arange(len(s)) / r
    m = t_r <= 3
    axs[i].plot(t_r[m], s[m])
    axs[i].set_title(f"{r} Hz")
    axs[i].set_xlabel("Time (s)")
    axs[i].set_ylabel("Amplitude")
    axs[i].grid(True)

axs[5].axis("off")
plt.tight_layout()
plt.show()
#section 4
X = np.fft.rfft(ecg)
f = np.fft.rfftfreq(len(ecg), 1/fs)
mag = np.abs(X) / len(ecg)

plt.figure(figsize=(10, 4))
plt.plot(f, mag)
plt.title("Clean ECG Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 60)
plt.grid(True)
plt.tight_layout()
plt.show()

#section 5
noise = np.random.randn(len(ecg))
a = 0.25 * np.std(ecg)
ecg_noisy = ecg + a * noise

plt.figure(figsize=(12, 4))
plt.plot(t, ecg, label="Clean")
plt.plot(t, ecg_noisy, label="Noisy", alpha=0.8)
plt.title("Clean vs Noisy ECG (Time Domain)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(t, ecg_noisy)
plt.title("Noisy ECG Signal (Time Domain)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

#section 6
Xc = np.fft.rfft(ecg)
fn = np.fft.rfftfreq(len(ecg), 1/fs)
Mc = np.abs(Xc) / len(ecg)

Xn = np.fft.rfft(ecg_noisy)
Mn = np.abs(Xn) / len(ecg_noisy)

plt.figure(figsize=(10, 4))
plt.plot(fn, Mc, label="Clean")
plt.plot(fn, Mn, label="Noisy", alpha=0.8)
plt.title("Frequency Spectrum: Clean vs Noisy ECG")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 60)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#section 7
from scipy.signal import butter, filtfilt

def butter_filter(x, cutoff, fs, btype, order=4):
    b, a = butter(order, cutoff/(0.5*fs), btype=btype)
    return filtfilt(b, a, x)

# 7.2 
ecg_lpf = butter_filter(ecg_noisy, 40, fs, "low")

m = t <= 3
plt.figure(figsize=(12, 4))
plt.plot(t[m], ecg_noisy[m], label="Noisy", alpha=0.8)
plt.plot(t[m], ecg_lpf[m], label="LPF 40 Hz")
plt.title("7.2 LPF Time-Domain Comparison")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

f = np.fft.rfftfreq(len(ecg_lpf), 1/fs)
M_lpf = np.abs(np.fft.rfft(ecg_lpf)) / len(ecg_lpf)

plt.figure(figsize=(10, 4))
plt.plot(f, M_lpf)
plt.title("7.2 LPF Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 60)
plt.grid(True)
plt.tight_layout()
plt.show()

# 7.3 
ecg_hpf = butter_filter(ecg_noisy, 0.5, fs, "high")

plt.figure(figsize=(12, 4))
plt.plot(t[m], ecg_noisy[m], label="Noisy", alpha=0.8)
plt.plot(t[m], ecg_lpf[m], label="LPF 40 Hz", alpha=0.9)
plt.plot(t[m], ecg_hpf[m], label="HPF 0.5 Hz")
plt.title("7.3 HPF Time-Domain Comparison")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

M_hpf = np.abs(np.fft.rfft(ecg_hpf)) / len(ecg_hpf)

plt.figure(figsize=(10, 4))
plt.plot(f, M_hpf)
plt.title("7.3 HPF Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 60)
plt.grid(True)
plt.tight_layout()
plt.show()