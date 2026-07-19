# Problem Statement (With accumulator-precision dimension, staircase H2, and FP64 accuracy oracle)

## Working title
Block Size Is an Implicit Numerical Parameter: How Paged KV-Cache Block Size Changes the Tokens a Language Model Generates.

## The causal variable, stated correctly

The naive framing "paged vs. contiguous cache layout changes outputs" is wrong, and getting this right is the paper's core precision. Indirection through a block table changes memory *addresses*, not arithmetic. If a paged attention kernel reads the same K/V values in the same order with the same reduction chunking as a contiguous kernel, the outputs are bitwise identical — layout alone is numerically inert.

What actually perturbs numerics is that **block size dictates the chunking of the online-softmax reduction over the KV sequence.** Under limited-precision arithmetic, floating-point addition is non-associative, so a different reduction split produces a different sum, a different logit, and eventually a different argmax. Paging changes outputs only insofar as it forces a different reduction split — and block size is the knob that controls that split. So the causal variable is not "layout"; it is **block size acting as an implicit reduction-split parameter**.

This reframing is the contribution's backbone: it converts a folklore-adjacent observation ("chunking changes numerics") into a precise, characterized claim ("the block-size parameter that serving systems set for memory reasons is silently also a numerical parameter that changes generated tokens, with the following positional structure, dose-response, accuracy consequences, and architectural dependence").

## Critical design variable: accumulator precision (decide before writing any kernel)

"Non-associativity of FP16/BF16" is imprecise, because in production kernels the accumulation usually does **not** happen in the storage dtype. FlashAttention-style kernels keep the online-softmax statistics and the output accumulator in **FP32**, and tensor-core matmuls accumulate in FP32, even when K/V are stored in FP16/BF16. Low-precision rounding enters at specific points (when KV is written to cache, and when intermediates are downcast), not throughout the reduction. This distinction determines the expected effect size and must be an explicit part of the design:

- **Accumulate in FP16/BF16 throughout** → large, dramatic block-size effect, but a reviewer who reads the kernel will correctly object that no production system does this and the effect is inflated.
- **Accumulate in FP32 with low-precision storage (production-realistic)** → divergence comes from FP32 online-softmax rescaling across chunks plus the low-precision rounding points; real, but possibly much smaller.

**Decision:** treat accumulator precision as an explicit experimental dimension (storage dtype × accumulator dtype), and designate low-precision-storage / FP32-accumulate as the *production-realistic reference configuration* whose numbers are the ones that matter for the serving-relevance claim. State this prominently in the paper. Explicitly accept the possible destination that FP32-accumulate divergence is small: "block size perturbs outputs far less than cache-on/off does, provided kernels accumulate in FP32" is a genuinely useful, publishable near-negative result, and walking in aware of it prevents a week-six surprise.

## Relationship to prior work (honest positioning)

- **Chodavarapu & Xu, "The Illusion of Equivalence" (arXiv:2604.15409, April 2026)** show cache-ON vs. cache-OFF FP16 paths diverge deterministically, with FP32 falsification dropping the flip rate to 0.0%. They do not vary block size or study paged execution; their axis is cache-on vs. recompute.
- **Thinking Machines Lab (2025) and the batch-invariance line (Yuan et al., arXiv:2506.09501)** attribute nondeterminism to batch-size / sequence-slicing variation changing kernel reduction strategy, and ship batch-invariant kernels that restore bitwise-identical outputs. This is adjacent: "sequence slicing changes numerics" is close to "block chunking changes numerics." An inference engineer will therefore consider the bare *existence* claim close to obvious.

**Therefore the contribution is explicitly measurement, mechanism localization, and a predictive model — not discovery of a phenomenon.** What the adjacent work has not done is the controlled academic characterization: the positional/functional structure of the perturbation, the block-size dose-response curve, the *accuracy* consequences against a true oracle, a margin-based predictive flip model, the GQA interaction, and quantified argmax-flip regimes on the specific parameter that production serving systems (vLLM, TGI, SGLang) expose and set. Folklore without characterization is precisely what a measurement paper exists to fix; the positioning says so plainly so a reviewer cannot land the "this is known folklore" objection.

## Core research question

Holding model, prompt, precision, sampling policy, and batch size fixed, how does the KV-cache **block size** — as a reduction-split parameter, at a stated accumulator precision — affect the generated token sequence: how large is the per-token logit perturbation, how does it scale with prefix chunk count, how does it affect *accuracy* against a true (FP64) reference, how predictable are token flips from logit margins, and how does grouped-query attention modulate all of this?

## Hypotheses (falsifiable)

- **H1 (mechanism, dispatched quickly — a corollary, not the headline).** With reduction chunking matched, paged and contiguous paths are bitwise identical; divergence appears only when block size changes the reduction split. Demonstrated via the bitwise-identity control, not oversold.
- **H2 (positional structure as a staircase — teacher-forced only).** Per-position single-step logit divergence between a small-block path and a reference grows with the **number of chunks covering the prefix, ⌈t/B⌉**, with discrete step increases at positions where a new block opens — a staircase in t, not spikes at boundaries. Stated as a predicted functional form (divergence vs. chunk count) and compared against classical summation-error expectations. *Measured teacher-forced, never from free-running generation.*
- **H3 (block-size dose-response — a centerpiece).** Argmax-flip rate and mean logit divergence vary with block size; smaller blocks (finer reduction splits, more chunk transitions) produce larger perturbation *relative to the single-block reference*. This dose-response curve is a central result. (Note the accuracy twist in H5b.)
- **H4 (GQA interaction).** Grouped-query attention amplifies block-size-induced divergence relative to a materialized MHA-equivalent control, consistent with the sharp first-layer GQA divergence reported by Chodavarapu & Xu.
- **H5a (precision dose-response).** At fixed accumulator precision, divergence grows as storage precision drops FP32 → FP16 → BF16. BF16 (production default) is expected to give the largest effect.
- **H5b (accuracy, not just sensitivity — the sharpest potential finding).** Measured against an **FP64 oracle**, the single-block reference is *not necessarily the most accurate* configuration; smaller blocks may be *closer* to the FP64 truth (pairwise-style accumulation is often more accurate than one long sequential reduction) even while diverging *more* from the single-block reference. If confirmed: "the serving default block size is not the most numerically accurate choice."
- **H6 (margin-predictive flip model).** A token flips iff the top-1/top-2 logit margin is smaller than the block-size-induced perturbation. Flip probability can therefore be fit as a function of measured margin and block size B, yielding a small predictive model: e.g., "on prose prompts with typical margin distributions, expect X flips per 1,000 tokens at block size 16 under BF16." This predictive sentence is the practitioner-facing takeaway.

## Methodology — two separated measurement modes

**Free-running generation cannot test positional or per-position claims**, because the first argmax flip changes the input to every later step, making all downstream divergence semantic rather than numeric. Therefore:

1. **Teacher-forced single-step divergence (for H2, H3 mechanism, H4, H5, H6).** Feed both paths (differing only in block size / precision) the *same fixed prefix*, no feedback loop, and compare logits position-by-position, the next-token argmax, and the logit margin. The only valid way to attribute perturbation to reduction-split structure. Primary scientific instrument.
2. **Free-running flip statistics (deployment-relevant summary).** Generate normally under each block size and report end-to-end flip/divergence rates across the prompt set. The "what it costs in practice" number, explicitly *not* used for positional attribution.

**Statistics come from the prompt/position distribution, not run repetition.** Both paths are deterministic on fixed hardware; run-to-run repetition is a one-time determinism sanity check, not the statistical engine. Confidence intervals are over prompts (and, for H2, over positions).

## Accuracy oracle

Add an **FP64 reference path** and report each block size's / precision's error against it, alongside divergence from the single-block reference. This is cheap given the existing harness and enables H5b, which is potentially the most quotable result: distinguishing *sensitivity* (difference between configs) from *accuracy* (distance from truth), and showing the production default may be neither the most stable nor the most accurate choice.

## Controls and unit tests

- **Bitwise-identity control (primary correctness gate).** Block size = full sequence length ⇒ paged path *bitwise identical* to contiguous under matched tiling. Stronger than "FP32 divergence collapses," because FP32 is itself non-associative and an indexing bug could survive an FP32 test while failing bitwise identity. Passing this gate is what licenses attributing divergence to reduction split rather than a gather/scatter bug.
- **GQA control.** Qwen2.5-0.5B is GQA. Preferred: materialize the replicated-KV MHA-equivalent form (mathematically identical, different KV grouping) as a same-model control. Cross-check: a genuine MHA model (Pythia or GPT-2) to confirm the effect is architectural. State the subtlety either way.
- **Autotune pinning (the real methods threat).** Clock/thermal throttling does not change kernel numerics — a red herring. The first-order threat is **Triton/backend autotuning selecting kernel configs by timing**, which changes reduction order for reasons unrelated to block size. Pin the autotune config explicitly and report it; otherwise document fixed kernel dispatch.

## External-validity bridge

Add a cheap production bridge: **real vLLM, batch size 1, single sequence, sweep `block_size`, report flip rates on the same prompt set.** Note the constraint: vLLM restricts `block_size` to backend-dependent values — **16 and 32 are the safe sweep; some backends reject 8** — so verify the sweep is actually runnable before promising it in the paper. Changing block size in vLLM also changes kernel dispatch, but that *is* the production question, and it ties the controlled measurements to a deployed system.

## Prompt domains

Flip rates depend heavily on the prompt domain, because logit-margin distributions differ (code prompts have sharper margins than open-ended prose). Use **at least two prompt domains** (e.g., code and prose) and **report margin distributions per domain** rather than pooling — pooling would hide the domain dependence that H6's predictive model is built on.

## Precision and configuration matrix (summary)

Storage dtype {FP32, FP16, BF16} × accumulator dtype {FP32 (production-realistic), low-precision (effect-inflating, for contrast)} × block size {sweep incl. seqlen for the identity gate} × attention grouping {GQA, materialized-MHA control} × prompt domain {≥2}, with an FP64 oracle path for accuracy. Designate low-precision-storage / FP32-accumulate as the reference configuration for headline claims.

## Model and hardware

Qwen2.5-0.5B-Instruct (plus a small MHA cross-check model) on a single consumer GPU. A controlled-isolation study, not a scale study: the contribution is separating block-size-induced (reduction-split) perturbation from the already-documented batch-induced and cache-vs-recompute perturbations, which requires holding many variables fixed across many prompts cheaply. Small scale is the correct instrument.

## Scope, tier, and kill condition

- **Tier:** an ML-systems / efficiency / reproducibility workshop paper (ICML/NeurIPS/MLSys workshops) or a clean, citable arXiv preprint. Not a flagship-conference systems paper; the value is rigorous novel measurement plus a small predictive model, not a new system.
- **Kill-condition sweep (before building):** search arXiv *and* "batch invariance," "paged attention determinism," and the vLLM / SGLang issue trackers for anyone who has already characterized block-size dose-response on output tokens. If the general framing is taken, the narrowest still-open slices are H2 (staircase functional form), H5b (accuracy-vs-oracle), and H6 (margin-predictive model) — the least likely to be covered.
- **Implementation prerequisite:** the bitwise-identity control must pass before any divergence number is trusted; accumulator precision must be fixed and stated before any effect size is reported.

## One-sentence version

Serving systems choose a KV-cache block size for memory reasons, unaware that block size is also a reduction-split parameter that, at production-realistic precision, changes the tokens the model generates; this work characterizes that effect — its staircase growth with prefix chunk count (teacher-forced), its block-size dose-response, its *accuracy* against an FP64 oracle (where the serving default may not be the most accurate choice), its predictability from logit margins, and its amplification under grouped-query attention — on a controlled small-model testbed, and bridges to production via a vLLM block-size sweep.
