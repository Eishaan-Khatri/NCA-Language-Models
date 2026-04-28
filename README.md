# From Grids to GPT

<p align="center">
  <img src="images/13_interactive-playground-screenshot.png" alt="Interactive NCA Playground" width="900" />
</p>

**An Interactive Re-implementation of _Training Language Models via Neural Cellular Automata_**

This repository provides a visually rich, notebook-driven exploration of Neural Cellular Automata (NCA) for synthetic token generation and pretraining. Below you'll find a compact summary, quick links, and a curated gallery of the figures used in the project.

---

## Live Notebook

Open the live interactive notebook in molab:

> **[Open in molab →](https://molab.marimo.io/notebooks/nb_oXEVG98GdYfmRxmKupEDuD)**

---

## Quick Highlights

- 164M NCA-generated tokens match or exceed performance of 1.6B natural tokens on reasoning benchmarks.
- Interactive NCA playground with sliders and live re-runs.
- Statistical analyses: Zipf checks, spectral (1/f) analysis, state-transition heatmaps, entropy evolution.
- Complexity sweep dashboard and NCA zoo for exploring high-quality rules.

---

## Gallery — Figures (click to enlarge)

Below are the images used in the notebook and README. Files live in `images/`.

### 01 — NCA grid evolution (animated)
<p align="center">
  <img src="images/01_nca-grid-evolution.gif" alt="NCA grid evolution" width="900" />
</p>

### 02 — Spacetime diagram
<p align="center">
  <img src="images/02_spacetime-diagram.png" alt="Spacetime diagram" width="800" />
</p>

### 03 — Token frequency (Zipf) and histogram
<p align="center">
  <img src="images/03_token-frequency-and-histogram.png" alt="Token frequency and histogram" width="800" />
</p>

### 04 — State transition heatmap
<p align="center">
  <img src="images/04_state-transition-heatmap.png" alt="State transition heatmap" width="800" />
</p>

### 05 — Spectral power / 1/f analysis
<p align="center">
  <img src="images/05_spectral-power.png" alt="Spectral power" width="800" />
</p>

### 06 — Entropy evolution
<p align="center">
  <img src="images/06_entropy-evolution.png" alt="Entropy evolution" width="800" />
</p>

### 07 — Cell vs 2×2 patch frequency comparison
<p align="center">
  <img src="images/07_cell-vs-patch-frequencies.png" alt="Cell vs patch frequencies" width="800" />
</p>

### 08 — Seed evolution triptych (example steps)
<p align="center">
  <img src="images/08_mandala-sequence-steps.png" alt="Seed evolution steps" width="800" />
</p>

### 09 — Complexity distribution (violin/box)
<p align="center">
  <img src="images/09_complexity-violin.png" alt="Complexity violin" width="800" />
</p>

### 10 — Token efficiency gains schematic
<p align="center">
  <img src="images/10_token-efficiency-gain.png" alt="Token efficiency gain" width="800" />
</p>

### 11 — Complexity / diversity vs power-law panel
<p align="center">
  <img src="images/11_complexity-diversity-powerlaw.png" alt="Complexity diversity powerlaw" width="800" />
</p>

### 12 — NCA grid snapshot (single step)
<p align="center">
  <img src="images/12_nca-grid-snapshot.png" alt="NCA grid snapshot" width="800" />
</p>

### 13 — Interactive playground screenshot (hero)
<p align="center">
  <img src="images/13_interactive-playground-screenshot.png" alt="Interactive playground screenshot" width="900" />
</p>

---

## Getting Started

Clone and run the notebook locally (no GPU required):

```bash
git clone https://github.com/Eishaan-Khatri/NCA-Language-Models.git
cd NCA-Language-Models
pip install marimo numpy matplotlib pillow
marimo edit nca_prepretraining.py
```

## Dependencies

- marimo >= 0.9.0
- numpy
- matplotlib >= 3.5
- pillow

---

## License

This project is released under the MIT License. See LICENSE for details.

---

If you'd like tweaks to the layout (smaller thumbnails, extra captions, or ordering changes), tell me which adjustments you want and I will update the README.
