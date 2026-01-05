# Phi Coupling Index: Golden Ratio Organization in Human EEG

[![DOI](https://img.shields.io/badge/DOI-pending-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains analysis code and processed data for investigating **golden ratio (φ ≈ 1.618) organization** in human EEG theta-alpha frequency architecture.

**Key Finding**: The Phi Coupling Index (PCI) strongly predicts theta-alpha convergence across 314 subjects (r = 0.638, p < 10⁻³⁷), suggesting that proximity to φ in cross-frequency ratios has functional relevance for neural dynamics.

**Bonus Discovery**: Population mean theta-alpha ratio (1.7221) approximates **e-1 ≈ 1.7183** (difference: 0.0038), suggesting transcendental mathematical constants may characterize default brain states.

## Mathematical Framework

| Constant | Value | Role in Brain Dynamics |
|----------|-------|------------------------|
| **φ (Golden Ratio)** | 1.618 | Optimal flexibility zone - maximal desynchronization |
| **e-1** | 1.718 | Population default attractor - "resting" state |
| **2:1 (Harmonic)** | 2.000 | Rigid phase-locking - processing mode |

The **Phi Coupling Index (PCI)** quantifies where an individual falls on this spectrum:

```
PCI = log(|ratio - 2.0| / |ratio - φ|)
```

- PCI > 0 → φ-organized (closer to golden ratio)
- PCI < 0 → Harmonic-organized (closer to 2:1)

## Repository Contents

### Analysis Scripts

| Script | Description |
|--------|-------------|
| `phi_specificity_analysis.py` | Core PCI computation and φ-specificity testing |
| `phi_prediction_test.py` | Correlation analysis: PCI vs theta-alpha convergence |
| `cross_frequency_analysis.py` | Cross-frequency coupling metrics |
| `comprehensive_analysis.py` | Full pipeline with all metrics |
| `meditation_baseline_analysis.py` | Meditation vs baseline comparisons |
| `coherence_analysis.py` | Inter-electrode coherence |

### Data Files

| File Pattern | Description | N |
|--------------|-------------|---|
| `alpha_s*.mat` | Processed spectral data (Dataset 1) | 20 |
| `alpha_subj_*.mat` | Processed spectral data (Dataset 2) | 18 |
| `subject_*.mat` | Raw spectral matrices | 5 |
| `meditation_spec.mat` | Meditation condition spectra | - |
| `boxplot_*.csv` | Summary statistics for visualization | - |

### Generated Figures

- `phi_specificity_analysis.png` - φ vs 2:1 distance comparison
- `phi_prediction_test.png` - PCI correlation with convergence
- `cross_frequency_coupling.png` - CFC visualization
- `comprehensive_phi_analysis.png` - Multi-panel summary figure

## Key Results

### 1. Population Distribution
- **67.2%** of subjects show φ-organization (PCI > 0)
- Mean ratio: **1.7221** (SE: 0.012)
- Approximates e-1 = 1.7183 (Δ = 0.0038)

### 2. Predictive Power
| Predictor | r | p-value | 95% CI |
|-----------|---|---------|--------|
| PCI (φ-index) | **0.638** | 2.6×10⁻³⁷ | [0.580, 0.690] |
| Distance from φ | -0.745 | <10⁻²⁴ | [-0.79, -0.69] |
| Distance from 2:1 | +0.687 | <10⁻²⁰ | [0.62, 0.74] |

### 3. Aperiodic Robustness
- 99.6% of subjects retained φ-organization after 1/f slope correction
- Effect is property of true oscillations, not spectral artifact

## Installation

```bash
git clone https://github.com/ExeqTer91/open-neuro-data.git
cd open-neuro-data
pip install numpy scipy matplotlib pandas
```

## Usage

```python
# Basic PCI computation
from phi_specificity_analysis import compute_pci

theta_centroid = 5.2  # Hz
alpha_centroid = 9.8  # Hz
pci = compute_pci(theta_centroid, alpha_centroid)
print(f"PCI: {pci:.3f}")  # Positive = φ-organized
```

## Citation

If you use this code or data, please cite:

```bibtex
@article{ursachi2026phi,
  title={Phi Coupling Index Predicts Theta-Alpha Convergence in Human EEG},
  author={Ursachi, Andrei},
  journal={NeuroImage},
  year={2026},
  note={Under review}
}
```

## Related Work

- Klimesch, W. (2013). An algorithm for the EEG frequency architecture of consciousness. *Frontiers in Human Neuroscience*, 7, 766.
- Pletzer, B., Kerschbaum, H., & Klimesch, W. (2010). When frequencies never synchronize: the golden mean and the resting EEG. *Brain Research*, 1335, 91-102.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

- **Author**: Andrei Ursachi
- **Email**: contact@andreiursachi.eu
- **Preprint**: [bioRxiv - pending]

---

*"The golden ratio is not merely a description of default EEG architecture but reflects a functional mechanism for neural flexibility."*
