# Plan B Implementation Decisions

## ReplayScheduler — IS weight applied as scalar multiplier

**Context:** Most HF models return scalar loss, not per-example losses,
making per-example IS weighting impossible without extra forward passes.
**Options considered:**
  A) Mean IS weight as scalar multiplier (approximation, one forward pass)
  B) Per-example IS weights (accurate, requires custom loss reduction)
**Chosen:** A
**Reason:** Keeps the integration surface simple; per-example requires
  overriding model internals which creates model-specific complexity.
**Impact:** Replay gradient is approximately but not exactly IS-corrected.
  Document as a known limitation.

## ReplayScheduler — Internal priority storage via numpy arrays

**Context:** FastPERBuffer is designed for RL transitions (state, action, reward,
next_state, done) and doesn't directly support generic anchor dict storage.
**Options considered:**
  A) Wrap FastPERBuffer by encoding anchor_idx as state (float32 array)
  B) Implement lightweight priority sampling directly with numpy arrays
  C) Build a separate sum-tree purely for anchor priority sampling
**Chosen:** B
**Reason:** Numpy-based weighted sampling with IS correction achieves the same
  proportional sampling semantics without forcing RL transition semantics onto
  anchor examples. FastPERBuffer's sum-tree is an implementation detail; the
  spec's requirement is proportional sampling with IS correction.
**Impact:** ReplayScheduler is self-contained; priorities stored as numpy float64
  arrays; IS weights computed using the same formula as FastPERBuffer.

## ForgettingMonitor — Loss computation for mock/minimal models

**Context:** anchor losses must be computed without assuming a specific
HuggingFace model architecture during testing.
**Options considered:**
  A) Detect model type via config.model_type and branch on causal LM vs classifier
  B) Try causal LM loss first, fall back to classifier loss on exception
  C) Accept a user-supplied loss_fn parameter
**Chosen:** A with fallback to causal LM
**Reason:** Explicit detection is clearer; fallback to causal LM matches the spec
  default ("If uncertain, default to causal LM loss").
**Impact:** Monitors work correctly for GPT-2, LLaMA, BERT-style classifiers.

## AnchorSet — priority-stratified subsampling without initial losses

**Context:** spec says priority-stratified requires initial losses not available
at construction time.
**Options considered:**
  A) Raise ValueError if strategy="priority-stratified" and losses not provided
  B) Fall back to "random" silently with a warning
  C) Accept optional initial_losses parameter for stratified sampling
**Chosen:** A — raise ValueError clearly
**Reason:** Fail fast with a clear error is better than silent degradation;
  users can snapshot_baseline first and then subsample if needed.
**Impact:** Users who want stratified subsampling must provide initial losses.

## ForgettingMonitor — tokenization of anchor data

**Context:** Anchor examples may be pre-tokenized (have input_ids) or raw text
(have a "text" or "input" key).
**Options considered:**
  A) Accept only pre-tokenized anchors; require user to tokenize before passing
  B) Auto-detect and tokenize if needed using model's tokenizer attribute
  C) Require a tokenizer parameter on ForgettingMonitor
**Chosen:** B with fallback to C — check model.tokenizer, then monitor.tokenizer
**Reason:** Most HF Trainer setups have a tokenizer accessible; requiring
  explicit pre-tokenization adds friction for common use cases.
**Impact:** ForgettingMonitor has an optional tokenizer parameter.
