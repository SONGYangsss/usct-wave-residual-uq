# Wave residuals for case-level uncertainty in learned USCT

Code for

> Y. Song and Y. Zhu, *Wave Residuals for Case-Level Uncertainty Calibration in Learned Ultrasound Computed
> Tomography under Simulated Density and Acquisition Shifts*, Biomedical Signal Processing and Control (2026).

A sound-speed map is reconstructed in a 64-mode PCA latent space through a learned Fourier-neural-operator
forward model F̂. The case-level scale is the normalised residual norm of the measured wave field,
`a_res = ||M(F̂(c) − (y − ȳ))|| / ||M(y − ȳ)||`. It multiplies a fixed relative width `w`, the latent posterior
standard deviation linearised at the prior mean. The interval `c ± q·a_res·w` is calibrated with a
phantom-level conformal quantile `q`. The back-projected residual `a_phys` is the variant scale. Data are
2D OpenBreastUS phantoms with 300 kHz OpenWaves measurements on a 256-element ring.

## Install

```
pip install -r requirements.txt
```

The GPU scripts need TensorFlow 2.10 to 2.15 and were run with TensorFlow 2.10 on an RTX 3060.
`evaluate.py` and `cycle_skipping.py` need only numpy and scipy.

## Data

Download the Zenodo record [10.5281/zenodo.23970986](https://doi.org/10.5281/zenodo.23970986), unzip
`checkpoints.zip` and `runs.zip` in the same folder, and set `USCTUQ_DATA` to it or pass `--data`.

```
data/
  ow_f300.npz, ow_f300_split.npz     200 evaluation phantoms, split into train/calib/test_id/ood (79/40/40/41)
  ow_train.npz, ow_train_b.npz       799 training phantoms for F̂, the PCA prior and the baselines
  checkpoints/                       F̂ (main and 8 controls), PCA prior (pca_d64.npz), 5 U-Net members, diffusion model
  runs/                              the reconstructions and baseline outputs behind every reported number
```

Each data file holds the speed maps (m/s), the 256 × 256 complex receiver matrix and the OpenBreastUS
phantom ids. The measurements were generated with OpenWaves and are derived from OpenBreastUS (CC BY 4.0).

## Reproduce the reported numbers

```
python scripts/evaluate.py --runs data/runs --out results
```

This uses only the run files and takes under an hour on a desktop CPU (`--only main_table,triage` runs a subset). Each section prints its rows and writes
`results/<section>.json`.

| Section | Paper |
|---|---|
| `main_table` | Table 6, unified pipeline comparison (Section 5.3) |
| `triage` | Table 7, Figure 6, triage paragraphs (Section 5.3) and the low-residual count (Section 5.5) |
| `scales` | Table 8, Figure 8, bootstrap intervals, partial Spearman, matched width (Section 5.4) |
| `ablation` | Figure 7 |
| `density_forward_error` | Figure 10 (OOD cutoff) and the forward-error overlap band (Section 5.6) |
| `controls` | Table 9, control surrogates (Section 5.6) |
| `acquisition` | Table 10, transducer gain/phase errors and noise (Section 5.7) |
| `sensitivity` | PCA dimension, mask threshold and case-specific width (Section 5.6) |

## Rerun the pipeline

```
python scripts/train_fno.py --name fno_main --subset all --select test_id
python scripts/reconstruct.py --name main --splits train,calib,test_id,ood --case-w
python scripts/run_baselines.py --name main
```

Controls, acquisition shifts and the PCA dimension use the same two scripts:

```
python scripts/train_fno.py --name fno_lowdens_s0 --subset lowdens --seed 0        # also matched, all; seeds 0-2
python scripts/reconstruct.py --name fno_lowdens_s0 --fno data/checkpoints/fno_lowdens_s0.weights.npz
python scripts/reconstruct.py --name acq_gain10 --perturb gain:0.10,10              # gain:0.02,2 / 0.05,5
python scripts/reconstruct.py --name acq_noise00 --perturb noise:0                  # noise:10 / 20 / 30 / 40
python scripts/run_baselines.py --name acq_gain10 --perturb gain:0.10,10 --methods unet
python scripts/reconstruct.py --name pca128 --pca-dim 128 --mask-std 0.005           # also 32, 64
```

`python scripts/cycle_skipping.py --plot cycle_skipping.png` runs the Helmholtz control of Section 5.5.

Notes:

- The main surrogate kept the epoch with the lowest ID-test forward error (`--select test_id`), as stated in
  Section 4.6. Every control keeps the final epoch.
- GPU arithmetic is not deterministic. Two runs of the same reconstruction differed by up to 9 m/s in single
  pixels, 2% in a_phys and 0.1% in a_res. MC dropout and diffusion sampling are stochastic, so a rerun gives a
  new draw. `runs/` holds the outputs used in the paper.
- Training from scratch gives a new model of similar accuracy, not the same weights.

## Regenerate the measurements

`scripts/prepare_data.py` writes OpenBreastUS speed maps as OpenWaves input (`stage`), extracts the receiver
matrices from the OpenWaves output (`extract`), and rebuilds the density split (`split`). OpenWaves runs in
MATLAB at Simulation Frequency 3.0 (300 kHz).

## Layout

```
usctuq/data.py        loading, measurement normalisation, density score, gain/noise perturbations
usctuq/fno.py         F̂
usctuq/prior.py       PCA prior and breast mask
usctuq/recon.py       latent fit, a_res, a_phys, back-projection, relative width w
usctuq/baselines.py   U-Net (MC dropout, deep ensemble) and conditional diffusion
usctuq/metrics.py     phantom-level conformal calibration, coverage, interval score, ranking statistics
```

## License

Code: MIT. Data and checkpoints on Zenodo: CC BY 4.0. Please also cite OpenBreastUS and OpenWaves.
