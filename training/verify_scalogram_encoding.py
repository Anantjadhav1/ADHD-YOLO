"""
Is TBR recoverable from the scalogram image? Read-only.

The check the original encoding never had. Correlates TBR computed from the
signal against TBR read back from the R channel across many epochs. If the
encoding preserves band amplitude they track; if it does not, correlation
collapses -- which is what row z-scoring did (r = -0.09 on synthetic data).

    py -m training.verify_scalogram_encoding --subject C09090107
"""
import argparse, warnings
import mne, numpy as np, pywt
mne.set_log_level("ERROR"); warnings.filterwarnings("ignore")

from data_pipeline import subject_split
from data_pipeline.preprocessing import (CHANNELS_19, epoch_signal, filter_raw,
                                         load_raw, split_eoec_by_alpha)
from data_pipeline.image_conversion import (CWT_FREQ_RANGE_HZ, CWT_N_FREQS,
                                            CWT_WAVELET, SCALOGRAM_LOG_RANGE_UV,
                                            _encode_scalogram_rgb)

THETA, BETA = (4, 8), (12, 30)

def band_ratio(power, freqs):
    t = (freqs >= THETA[0]) & (freqs < THETA[1])
    b = (freqs >= BETA[0]) & (freqs < BETA[1])
    return power[t].mean() / power[b].mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--manifest", default="data_pipeline/splits/subject_splits.csv")
    ap.add_argument("--n-epochs", type=int, default=40)
    a = ap.parse_args()

    m = subject_split.load_manifest(a.manifest)
    row = m.loc[m.subject_id == a.subject].iloc[0]
    ep = epoch_signal(split_eoec_by_alpha(filter_raw(load_raw(row.eoec_path)))["ec"])
    data = ep.get_data(picks=[c for c in CHANNELS_19 if c in ep.ch_names])
    ch = [c for c in CHANNELS_19 if c in ep.ch_names]
    sfreq = float(ep.info["sfreq"])

    freqs = np.linspace(*CWT_FREQ_RANGE_HZ, CWT_N_FREQS)
    scales = pywt.frequency2scale(CWT_WAVELET, freqs / sfreq)
    lo, hi = SCALOGRAM_LOG_RANGE_UV
    idx = ch.index("Fz") if "Fz" in ch else 0

    true, from_r, from_b, clip_lo, clip_hi = [], [], [], [], []
    for i in range(min(a.n_epochs, len(data))):
        coeffs, _ = pywt.cwt(data[i][idx], scales, CWT_WAVELET, sampling_period=1/sfreq)
        p = np.abs(coeffs) * 1e6
        rgb = np.round(_encode_scalogram_rgb(p, freqs) * 255) / 255   # as a PNG would
        true.append(band_ratio(p, freqs))
        from_r.append(band_ratio(10 ** (lo + rgb[..., 0] * (hi - lo)), freqs))
        from_b.append(band_ratio(rgb[..., 2] * 6 - 3, freqs))
        clip_lo.append((rgb[..., 0] <= 0).mean()); clip_hi.append((rgb[..., 0] >= 1).mean())

    true, from_r, from_b = map(np.array, (true, from_r, from_b))
    print(f"\n{a.subject}, {len(true)} EC epochs, channel {ch[idx]}")
    print(f"  true TBR range        {true.min():.2f} - {true.max():.2f}")
    print(f"  R channel  corr       {np.corrcoef(true, from_r)[0,1]:.4f}   <- must be > 0.95")
    print(f"  B channel  corr       {np.corrcoef(true, from_b)[0,1]:.4f}   <- the old encoding")
    print(f"  R clipped low/high    {np.mean(clip_lo):.2%} / {np.mean(clip_hi):.2%}   <- widen range if > 1%")
    ok = np.corrcoef(true, from_r)[0,1] > 0.95
    print(f"\n  {'PASS -- TBR is recoverable from the image.' if ok else 'FAIL -- band amplitude is not preserved.'}")

if __name__ == "__main__":
    main()