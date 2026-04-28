<div align="center">

# 🧬 From Grids to GPT

### Teaching Language Models to Reason with Neural Cellular Automata

[![Open in molab](https://img.shields.io/badge/Open%20in-molab-7c3aed?style=for-the-badge&logo=python&logoColor=white)](https://molab.marimo.io/notebooks/nb_oXEVG98GdYfmRxmKupEDuD)
[![arXiv](https://img.shields.io/badge/arXiv-2603.10055-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.10055)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Built with marimo](https://img.shields.io/badge/Built%20with-marimo-f59e0b?style=for-the-badge)](https://marimo.io)

<br/>

<img src="images/01_nca-grid-evolution.gif" alt="NCA grid evolution" width="720" />

<br/>

**164 million NCA tokens · outperform · 1.6 billion natural language tokens**  
*A 10× data-efficiency gain on reasoning benchmarks*

<br/>

</div>

---

## What This Is

An interactive, visually rich re-implementation of the paper **[Training Language Models via Neural Cellular Automata](https://arxiv.org/abs/2603.10055)** (Lee, Han, Kumar & Agrawal, 2026), built as a [marimo](https://marimo.io) notebook for the [alphaXiv × marimo Notebook Competition](https://marimo.io/pages/events/notebook-competition).

Pre-training LLMs on internet text is hitting a wall — quality data is running out and world knowledge gets tangled with reasoning ability. This paper's answer: generate synthetic token sequences from **Neural Cellular Automata** — simple colored grids evolving under a hidden neural rule. The model must infer that rule from context alone. Doing this at scale trains attention to route information across long distances, a skill that transfers directly to math, code, and logic.

---

## ✨ Highlights

| | |
|---|---|
| 🎮 **Interactive Playground** | Live NCA simulation — every slider re-runs instantly |
| 🧩 **Rule Inference Challenge** | Spot the imposter board — the same task the LLM solves |
| 📊 **Full Statistical Suite** | Zipf, 1/f spectral, entropy, transition heatmaps |
| 🔬 **Baseline Showdown** | NCA vs random noise vs Rule 30 CA vs Dyck-1 |
| 🌊 **Complexity Sweep** | Find high-quality rules + one-click corpus export |
| 🦁 **NCA Zoo** | 8 named rules with live animations |
| 📦 **Export Ready** | Download token sequences as `.npy` for your own training runs |

---

## 🖼 Gallery

<details open>
<summary><strong>Spacetime Diagram — what the LLM actually reads</strong></summary>
<br/>
<p align="center">
  <img src="images/02_spacetime-diagram.png" alt="Spacetime diagram" width="800" />
</p>
<p align="center"><em>Each row is one flattened grid snapshot. Diagonal bands = information propagating at finite speed = long-range dependencies.</em></p>
</details>

<details>
<summary><strong>Interactive Playground</strong></summary>
<br/>
<p align="center">
  <img src="images/13_interactive-playground-screenshot.png" alt="Interactive playground" width="800" />
</p>
</details>

<details>
<summary><strong>Statistical Analysis — Zipf & Token Distribution</strong></summary>
<br/>
<p align="center">
  <img src="images/03_token-frequency-and-histogram.png" alt="Token frequency and histogram" width="800" />
</p>
</details>

<details>
<summary><strong>State Transition Heatmap</strong></summary>
<br/>
<p align="center">
  <img src="images/04_state-transition-heatmap.png" alt="State transition heatmap" width="800" />
</p>
<p align="center"><em>The hidden grammar of each NCA rule. Bright diagonal = sticky states. Off-diagonal structure = what the model must learn.</em></p>
</details>

<details>
<summary><strong>1/f Spectral Power Analysis</strong></summary>
<br/>
<p align="center">
  <img src="images/05_spectral-power.png" alt="Spectral power" width="800" />
</p>
</details>

<details>
<summary><strong>Entropy Evolution Over Time</strong></summary>
<br/>
<p align="center">
  <img src="images/06_entropy-evolution.png" alt="Entropy evolution" width="800" />
</p>
</details>

<details>
<summary><strong>Cell vs 2×2 Patch Tokenization</strong></summary>
<br/>
<p align="center">
  <img src="images/07_cell-vs-patch-frequencies.png" alt="Cell vs patch frequencies" width="800" />
</p>
</details>

<details>
<summary><strong>NCA Zoo — Rule Evolution Steps</strong></summary>
<br/>
<p align="center">
  <img src="images/08_mandala-sequence-steps.png" alt="Seed evolution steps" width="800" />
</p>
</details>

<details>
<summary><strong>Complexity Distribution Across Seeds</strong></summary>
<br/>
<p align="center">
  <img src="images/09_complexity-violin.png" alt="Complexity violin" width="800" />
</p>
<p align="center"><em>Green band = productive gzip range (0.35–0.55). Seeds outside this window lose most of their transfer benefit.</em></p>
</details>

<details>
<summary><strong>Baseline Showdown — Complexity vs Power Law</strong></summary>
<br/>
<p align="center">
  <img src="images/11_complexity-diversity-powerlaw.png" alt="Complexity diversity powerlaw" width="800" />
</p>
</details>

<details>
<summary><strong>Token Efficiency Gain — Schematic</strong></summary>
<br/>
<p align="center">
  <img src="images/10_token-efficiency-gain.png" alt="Token efficiency gain" width="800" />
</p>
</details>

---

## 📋 Results

| Setting | NCA Tokens | Natural Tokens | GSM8K | HumanEval | BIG-Bench |
|---|---|---|---|---|---|
| NCA pre-training only | 164 M | — | +6% | +4% | +5% |
| NCA + fine-tuning | 164 M | 200 M | **best** | **best** | **best** |
| C4 pre-pre-training | 164 M | 200 M | −2% | −1% | +1% |
| Natural language only | — | 1.6 B | baseline | baseline | baseline |

### Domain Calibration

| Domain | Gzip Ratio | Best *n* | Efficiency Gain |
|---|---|---|---|
| Web Text (C4) | ~0.70 | 13–15 | ~31% |
| Mathematics | ~0.58 | 10–12 | ~27% |
| Code | ~0.32 | 5–6 | **~49%** |
| General sweet spot | 0.35–0.55 | 8–12 | ~10× |

---

## 🚀 Getting Started

```bash
git clone https://github.com/Eishaan-Khatri/NCA-Language-Models.git
cd NCA-Language-Models
pip install marimo numpy matplotlib pillow
marimo edit nca_prepretraining.py
```

> **No GPU required.** Everything runs on CPU.

**Dependencies:** `marimo >= 0.9.0` · `numpy` · `matplotlib >= 3.5` · `pillow`

---

## 🔍 Implementation Notes

> **ReLU vs tanh** — The paper's ReLU MLP produces sharp attractor states that force genuine rule inference. Tanh smooths the transitions out and weakens the training signal.

> **Temperature** — Paper uses τ = 0.001 (near-argmax) for corpus generation. The playground defaults to τ = 0.6 for richer visuals. Zipf slope and gzip ratio are stable across this range; the spectral exponent is the most sensitive indicator.

> **Patch tokenization** — At n > 10, modulo-10k hashing of 2×2 patches produces collision rates above 40%. The notebook uses dynamic tokenization throughout.

> **Complexity filtering** — The productive gzip window (0.35–0.55) is narrow. Two hundredths of a point outside it and most transfer benefit disappears. Use the Sweep Dashboard before building a corpus.

---

## 💭 Reflections

When I first read this paper, the premise struck me as implausible. Why should colored grids produce better reasoning signals than carefully curated text?

The answer lies in what the model is forced to do. Natural language pre-training requires simultaneously acquiring world knowledge *and* learning to route information across long contexts — objectives that interfere with each other. NCA sequences strip the knowledge component away entirely, leaving only the structural problem of inferring a hidden rule. The training signal is unusually clean.

Building the pipeline surfaced things the paper mentions but doesn't dwell on: the narrowness of the complexity sweet spot, the severity of patch-tokenization collisions at scale, and how much ReLU versus tanh changes the character of the attractor. None are show-stoppers, but they only become real when you run the code.

---

## 🔭 Future Directions

- Automatic alphabet-size selection conditioned on target corpus statistics
- Hierarchical NCA with coarse-to-fine temporal structure
- Mechanistic analysis of which attention heads specialize in NCA-induced vs language-induced routing
- GPU-scale implementation for full 164 M-token corpus generation

---

## 🙏 Acknowledgements

Built on the research of Lee, Han, Kumar, and Agrawal — a paper that is both conceptually elegant and unusually reproducible.

Developed for the [alphaXiv × marimo Notebook Competition](https://marimo.io/pages/events/notebook-competition) · April 2026

---

## License

Released under the [MIT License](LICENSE). Free to use, modify, and build upon with attribution.

---

<div align="center">
<sub>Made with ❤️ using <a href="https://marimo.io">marimo</a> · April 2026</sub>
</div>
