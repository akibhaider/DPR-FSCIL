# Paper Sources — Literature Review Reading List

All links verified on 2026-04-23. Each paper has at least one **primary source** (official venue / arXiv) and one **backup** where available.

---

## Section A — Why old interpretability was not enough

 Olah et al., *Feature Visualization* (Distill, 2017)
 Adebayo et al., *Sanity Checks for Saliency Maps* (NeurIPS, 2018)
 Koh et al., *Concept Bottleneck Models* (ICML, 2020)
 Elhage et al., *Toy Models of Superposition* (Anthropic, 2022)
 Olah et al., *The Building Blocks of Interpretability* (Distill, 2018)
 Bricken et al., *Towards Monosemanticity: Decomposing Language Models with Dictionary Learning* (Anthropic, 2023)
 Templeton et al., *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet* (Anthropic, 2024)
 Lim et al., *Sparse autoencoders reveal selective remapping of visual concepts during adaptation (PatchSAE)* (ICLR, 2025)
 Pach et al., *Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models* (NeurIPS, 2025)
 Stevens et al., *Sparse Autoencoders for Scientifically Rigorous Interpretation of Vision Models* (2025)
 Saporta et al., *Benchmarking saliency methods for chest X-ray interpretation* (Nature Machine Intelligence, 2022)
 Yuksekgonul et al., *Post-hoc Concept Bottleneck Models*
 Li et al., *LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day* (NeurIPS, 2023)
 Abdulaal et al., *An X-Ray Is Worth 15 Features: Sparse Autoencoders for Interpretable Radiology Report Generation (SAE-Rad)(2024)
 Luo et al., *XBench: A Comprehensive Benchmark for Visual-Language Explanations in Chest Radiography* (2025)
 Jain et al., *RadGraph: Extracting Clinical Entities and Relations from Radiology Reports* (NeurIPS Datasets & Benchmarks, 2021)

---

## Three things to know before downloading

**1. Distill and Transformer Circuits publications are HTML-native.** For Olah's Feature Visualization, Building Blocks, Bricken's Towards Monosemanticity, and Templeton's Scaling Monosemanticity, there is no canonical PDF — the interactive HTML *is* the paper. If you need a local file, open the page in Chrome/Firefox and use "Print → Save as PDF". You will lose the interactive elements but keep the text and figures.

**2. Two corrections from my earlier presentation deck.** I got two authorships/venues wrong and you should fix them in your LaTeX slides before presenting:
- Yuksekgonul et al. *Post-hoc CBM* is **ICLR 2023**, not Nature MI 2023.
- XBench authors are **Luo et al.**, not Roy et al. (The GitHub repo is called `Roypic/Benchmarkingattention` because of the first author's handle, which is probably what confused me.)

**3. arXiv version pinning.** For the papers whose abstracts have been substantially revised (Stevens et al. changed title between v1 and v2; Pach et al. has three versions), cite the version you actually read. Don't assume the current arXiv page matches the version you downloaded months later.

---

## Quick-copy BibTeX-style list (for your references slide)

```
Olah et al. 2017           https://distill.pub/2017/feature-visualization/
Olah et al. 2018           https://distill.pub/2018/building-blocks/
Adebayo et al. 2018        https://arxiv.org/abs/1810.03292
Koh et al. 2020            https://arxiv.org/abs/2007.04612
Elhage et al. 2022         https://arxiv.org/abs/2209.10652
Bricken et al. 2023        https://transformer-circuits.pub/2023/monosemantic-features
Templeton et al. 2024      https://transformer-circuits.pub/2024/scaling-monosemanticity/
Lim et al. 2025 (PatchSAE) https://arxiv.org/abs/2412.05276
Pach et al. 2025           https://arxiv.org/abs/2504.02821
Stevens et al. 2025        https://arxiv.org/abs/2502.06755
Saporta et al. 2022        https://www.nature.com/articles/s42256-022-00536-x
Yuksekgonul et al. 2023    https://arxiv.org/abs/2205.15480
Li et al. 2023 (LLaVA-Med) https://arxiv.org/abs/2306.00890
Abdulaal et al. 2024       https://arxiv.org/abs/2410.03334
Luo et al. 2025 (XBench)   https://arxiv.org/abs/2510.19599
Jain et al. 2021 (RadGraph) https://arxiv.org/abs/2106.14463
```
