# Project Context — Knowledge-Anchored Sparse Autoencoders for Medical VLM Interpretability

> Paste this file at the start of a new Claude conversation to resume work on this project without re-explaining everything. Tell Claude: "This is the context from a prior session. Read it, then help me with X."

---

## 1. Who I am and what I am trying to build

I am **Akib**, BSc Computer Science and Engineering student at Islamic University of Technology (IUT), Bangladesh. I have research background in deep learning, federated learning, medical imaging, and neuro-symbolic methods applied to code analysis (ICSE NLBSE).

I got interested in interpretability research after reading Olah et al.'s *Feature Visualization* (Distill 2017) and *Building Blocks of Interpretability* (Distill 2018). I now want to do original research at the intersection of **mechanistic interpretability + medical vision-language models (VLMs)**.

## 2. My initial (wrong) premise — for the record

I originally believed that "VLMs related applications like VQA, captioning and ViT-transformer architectures have not yet gone through interpretability work." A prior Claude session corrected this: these areas are actively researched. The actual gap is narrower.

## 3. The actual research direction (the thing I am pursuing)

**Working title:** Knowledge-Anchored Sparse Autoencoders for Interpreting Medical Vision-Language Models

**One-sentence version:** Train a sparse autoencoder on a medical VLM's vision encoder, biased by clinical knowledge graph labels (RadGraph / UMLS / SNOMED-CT), and use the resulting features to diagnose why the model hallucinates.

**Why this sits in a real gap:**
- General-domain SAE work on CLIP / LLaVA is crowded (Pach et al. NeurIPS 2025, Zaigrajew et al. ICML 2025).
- Medical VLM work focuses on hallucination *detection*, not hallucination *mechanism*.
- The only close prior work is SAE-Rad (2024), which is purely unsupervised — features still require post-hoc guessing.
- Stevens et al. (2025, "Brazil/lamp" finding) showed that post-hoc guessing of SAE feature meaning is often wrong. In medicine this is unacceptable.
- My opening: inject a clinical knowledge graph directly into the SAE training objective so features are named by construction, not inspection.

## 4. The contribution I must be able to defend

A positive result would support this claim, and no stronger one:

> Adding a clinical-knowledge-graph alignment loss to SAE training on a medical VLM's vision encoder produces features that are (a) more concept-pure against held-out labels, (b) more causally faithful under feature ablation, and (c) more useful for diagnosing specific hallucination types than features from an unsupervised SAE baseline.

## 5. The four verification questions I must answer before claiming novelty

| # | Question | Status |
|---|---|---|
| A | Is the unsupervised-SAE baseline correctly defined and has it been done? | Answered by reading Bricken et al. 2023 and SAE-Rad 2024 |
| B | Is the knowledge-anchoring idea actually new, or a renamed CBM? | Answered by reading Koh et al. 2020 (CBM) and Yuksekgonul et al. 2023 (Post-hoc CBM Medical), plus arXiv search on "supervised SAE" / "concept-guided SAE" |
| C | Do my evaluation metrics measure what they claim? | Answered by reading Adebayo et al. 2018 (sanity checks), Jacovi & Goldberg 2020 (faithfulness), Heimersheim & Nanda 2024 (activation patching) |
| D | What does a positive result actually mean, and what strong claim would be unsupported? | Requires epistemic caution — strong claim = "we explained hallucination mechanisms"; weak claim = "some features correlate with hallucination events". Do not conflate. |

## 6. Critical distinction — SAE vs. Concept Bottleneck Model

The most likely reviewer objection: "this is just a CBM with a different name."

**Rebuttal:** A CBM replaces the prediction layer with concepts — it is a constraint on *output*. An SAE is a post-hoc dictionary over *internal activations* that preserves full reconstruction of the representation. Our method uses concepts to bias a *subset* of SAE features while leaving the rest unsupervised, and critically, the SAE still reconstructs the entire activation vector — the non-anchored features capture whatever concepts aren't in the knowledge graph. CBMs route predictions through concepts; we make a subset of internal features interpretable without routing anything.

## 7. The reading roadmap (9 weeks, ~20 papers, not 200)

### Phase 1 — SAE foundations (3 weeks) ✓
- Olah et al., *Feature Visualization* (Distill 2017)
- Olah et al., *Building Blocks of Interpretability* (Distill 2018)
- **Elhage et al., *Toy Models of Superposition* (Anthropic 2022)** — why SAEs exist
- **Bricken et al., *Towards Monosemanticity* (Anthropic 2023)** — first working SAE
- Templeton et al., *Scaling Monosemanticity* (Anthropic 2024)

### Phase 2 — Vision/VLM SAEs (2 weeks) ✓
- **Pach et al., *SAEs Learn Monosemantic Features in VLMs* (NeurIPS 2025)**
- Zaigrajew et al., *Interpreting CLIP with Hierarchical SAEs* (ICML 2025)
- Stevens et al., *SAEs for Scientifically Rigorous Interpretation* (2025) — Brazil/lamp
- (Optional) Lim et al., *PatchSAE* (ICLR 2025) — patch-level SAEs

### Phase 3 — Medical anchor (2 weeks) ✓
- **Abdulaal et al., *SAE-Rad: An X-Ray Is Worth 15 Features* (2024)** — read twice
- Li et al., *LLaVA-Med* (NeurIPS 2023)
- Jain et al., *RadGraph* (NeurIPS 2021)
- Johnson et al., *MIMIC-CXR* (Scientific Data 2019) — apply for PhysioNet early
- Irvin et al., *CheXpert* (AAAI 2019)

### Phase 4 — Novelty defense (1 week) ✓
- **Koh et al., *Concept Bottleneck Models* (ICML 2020)**
- Kim et al., *TCAV* (ICML 2018)
- Yuksekgonul et al., *Post-hoc CBM* (Nature MI 2023)
- arXiv search: "supervised sparse autoencoder", "concept-guided SAE"

### Phase 5 — Evaluation rigor (1 week) ✓
- **Adebayo et al., *Sanity Checks for Saliency Maps* (NeurIPS 2018)**
- Jacovi & Goldberg, *Towards Faithfully Interpretable NLP Systems* (ACL 2020)
- Heimersheim & Nanda, *How to use and interpret activation patching* (2024)
- Saporta et al., *Benchmarking Saliency for Chest X-Ray* (Nature MI 2022)

## 8. What the five families of interpretability are (reference)

When reading papers, always classify them into one of these:

| Family | Question answered | Example |
|---|---|---|
| 1. Attribution | For this prediction, which pixels mattered? | Grad-CAM, Integrated Gradients, Chefer et al. ViT-LRP |
| 2. Feature visualization | What does neuron N represent? | Olah et al., activation maximization |
| 3. Probing | Is info X decodable from layer L? | TCAV, linear probes |
| 4. Causal intervention | If I ablate component C, does behavior change? | Activation patching, ACDC, circuit discovery |
| 5. Dictionary learning / SAEs | Can I re-describe the activation sparsely? | Bricken, Pach, SAE-Rad |

My project sits in families 4 and 5 (with concept supervision borrowed from family 3).

## 9. Experimental setup (planned, not run)

- **Base model:** LLaVA-Med (primary), CheXagent (secondary)
- **Dataset:** MIMIC-CXR + RadGraph parses (requires PhysioNet credentialed access — CITI training takes weeks, start early)
- **Concept vocabulary:** CheXpert 14-label schema + RadGraph observation/anatomy entities
- **Baselines:** unsupervised SAE, SAE-Rad re-implementation, linear-probe CBM on same activations
- **Hallucination benchmark:** MedVH
- **Compute budget (estimated):** single 40GB A100 feasible; ~1 week of training for a full SAE on LLaVA-Med's vision tower

## 10. Three-axis evaluation protocol

1. **Concept purity.** For each anchored feature, selectivity against held-out RadGraph labels (not the labels used during auxiliary-loss training). Metric: AUROC or precision@k.
2. **Causal faithfulness.** Ablate feature k and measure the specific change in generation behavior. Does ablating the "consolidation" feature specifically reduce mentions of consolidation while leaving other findings intact? This is the most important metric — "concept purity" can be gamed by the auxiliary loss itself.
3. **Hallucination diagnosis.** On MedVH-flagged hallucinations, do anchored features identify which components over/under-fired? Evaluated statistically, not on cherry-picked examples.

## 11. Known risks and how to handle them

- **Scooping risk:** the space is moving weekly. Set arXiv alerts on "supervised SAE", "medical VLM interpretability", "SAE medical imaging". Monitor at least weekly.
- **Compute:** LLaVA-Med 7B vision tower SAE fits on 40GB. If only 24GB available, use quantization or a smaller VLM.
- **Faithfulness pitfall:** do not rely only on correlation-based metrics. Every claim needs a causal-intervention check.
- **Clinical validation gap:** ideally need a radiologist to review ~50 features for plausibility. Start asking IUT / BSMMU / Dhaka hospital contacts early.
- **CBM conflation:** articulate the SAE-vs-CBM distinction crisply every time (see §6).

## 12. Things NOT to do

- Don't claim "first to do X" without an arXiv sweep the same week.
- Don't use only concept-purity as evaluation (circular with training loss).
- Don't propose a 10-VLM, 5-modality, 3-technique paper — single VLM, single modality, focused contribution.
- Don't target CVPR/NeurIPS main track as a first paper. MICCAI or iMIMIC workshop is more realistic given IUT compute profile and the clinical focus.

## 13. Current state (as of this context file)

- **Literature review:** completed per the roadmap above.
- **Presentation:** drafted a beamer deck with title / problem / motivation / objectives / literature-review-taxonomy-and-chronology diagram / gap analysis. Placeholder sections for method, experiments, results exist but are intentionally unfilled until real work is done.
- **PhysioNet access:** (status: apply if not done)
- **Code environment:** (status: set up TransformerLens, SAELens, LLaVA-Med weights)
- **Advisor conversation:** (status: pending — draft email written but superseded; should be based on this project scope)

## 14. Useful session-spanning facts

- My advisor is at IUT; not yet confirmed which lab.
- My prior Claude sessions produced: (a) a reading curriculum, (b) a draft advisor email (deprecated), (c) a presentation deck following the structure of Raiyan et al.'s MWP thesis deck, (d) the taxonomy/chronology figure used in the literature review slide.
- Preferred Claude behavior: terse, directive, honest about uncertainty, willing to disagree. Do not over-format responses with bullets unless asked. Do not hedge excessively. Call out when I'm about to overclaim.

## 15. What to ask me when I come back

If I return without a clear question, ask me:
1. Which of the 5 reading phases I have actually completed (not just "started").
2. Whether I have PhysioNet access yet.
3. Whether I have a first forward pass of LLaVA-Med running end-to-end.
4. What the single most interesting paper from the last week's reading was and how it changes my contribution claim.

Do not start generating slides, code, or proposal text without at least one of these being answered.
