# Multi-Fidelity PINN with Boundary-Aware Losses (Greenland Bed Topography)

This repository implements Physics-Informed Neural Networks (PINNs) for subglacial bed elevation prediction, combining **multi-fidelity physics residuals** (SIA + reduced-Stokes) with a **boundary-aware weak-form loss** (Neumann traction with optional Dirichlet constraints). It includes baseline, boundary-aware, and multi-fidelity training pipelines.

---

## Features
- Multi-fidelity residual coupling (SIA + reduced-Stokes).
- Boundary-aware weak-form loss (Neumann + optional Dirichlet).
- Curriculum collocation and cosine learning-rate scheduling.
- Baseline and boundary-aware training pipelines with reproducible metrics.

---

## Repository Structure
```

.
├── figs/                     # Final figures and plots
├── main\_code\_multi\_fidelity\_pinn.py
├── pinn\_physics\_tight\_baseline.py
├── pinn\_with\_boudary\_aware.py
├── requirements.txt
└── README.md

````

---

## Installation
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
````

> GPU is optional but recommended (PyTorch ≥ 2.2).

---

## Quick Start

### Baseline PINN (physics-tight)

```bash
python pinn_physics_tight_baseline.py
```

### Multi-fidelity PINN with learned weighting

```bash
python main_code_multi_fidelity_pinn.py
```

### Boundary-aware PINN

```bash
python pinn_with_boudary_aware.py
```

Each script defines editable hyperparameters at the top (epochs, hidden width, learning rate, collocation points, etc.).

---

## Results

Our experiments demonstrate clear benefits of adding boundary-aware losses and multi-fidelity coupling to the baseline PINN:

* **Baseline (Physics-Tight PINN):**
  Achieved good predictive skill (*Test MSE ≈ 0.028, R² ≈ 0.97*) while tightly enforcing PDE constraints.

* **Boundary-Aware PINN:**
  Soft/partial Dirichlet constraints improved boundary fidelity and reduced predictive error.
  Hard Dirichlet enforcement, by contrast, degraded accuracy (over-constrained).

* **Multi-Fidelity PINN:**
  Combining SIA and reduced-Stokes residuals with adaptive weighting (log-variance balancing) yielded the best overall performance.
  Consistently lowered **RMSE** and **MAE**, while improving **R²** compared to both baseline and boundary-aware alone.

**Key Takeaway:**
Boundary-aware and multi-fidelity PINNs produce **more accurate and physically consistent reconstructions** of Greenland subglacial bed topography than baseline PINNs.

---

## Data Availability and Acknowledgements

We utilize five ice sheet surface measurements from four publicly available sources.
These datasets were preprocessed and combined following the methodology in:
**K. Yi et al., "Evaluating Machine Learning and Statistical Models for Greenland Subglacial Bed Topography," 2023.**

### Data Sources

1. **Surface Elevation**
   I. Howat, A. Negrete, and B. Smith, *“MEaSUREs Greenland Ice Mapping Project (GIMP) Digital Elevation Model from GeoEye and WorldView Imagery, Version 1,”* 2017.
   [NSIDC-0715](https://nsidc.org/data/NSIDC-0715/versions/1)

2. **Ice Flow Surface Velocity**
   J. Mouginot, E. Rignot, B. Scheuchl, and R. Millan, *“Comprehensive Annual Ice Sheet Velocity Mapping Using Landsat-8, Sentinel-1, and RADARSAT-2 Data,”* Remote Sensing, vol. 9, no. 4, p. 364, 2017.
   [DOI:10.3390/rs9040364](https://doi.org/10.3390/rs9040364)

3. **Ice Thinning Rates**
   B. Smith, S. Adusumilli, B. M. Csatho, D. Felikson, H. A. Fricker, A. Gardner, N. Holschuh, J. Lee, J. Nilsson, F. S. Paolo, M. R. Siegfried, T. Sutterley, and the ICESat-2 Science Team,
   *“ATLAS/ICESat-2 ATL06 Land Ice Height, Version 5,”* 2021.
   [NSIDC-ATL06](https://nsidc.org/data/ATL06/versions/5)

4. **Surface Mass Balance**
   J. M. van Wessem and M. K. Laffin, *“Regional Atmospheric Climate Model (RACMO2), Version 2.3p2,”* 2020.
   [Zenodo DOI:10.5281/zenodo.3677642](https://doi.org/10.5281/zenodo.3677642)

### Preprocessing & Integration

The above datasets were cleaned and used in:
Katherine Yi, Angelina Dewar, Tartela Tabassum, Jason Lu, Ray Chen, Homayra Alam, Omar Faruque, Sikan Li, Mathieu Morlighem, and Jianwu Wang,
**“Evaluating Machine Learning and Statistical Models for Greenland Subglacial Bed Topography.”**

We gratefully acknowledge these data providers and authors for making their datasets and methodologies publicly available.
