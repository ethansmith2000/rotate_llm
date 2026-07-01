# rotato — multi-model rotation vs. Pangram

A harness to test the hypothesis: *does rotating between LLMs every few
tokens/words/sentences help AI-generated text evade Pangram?* Built to produce a
clean, publishable **negative result** (or a surprise, if there is one).

## Why it's built the way it is

**Coherence-preserving continuation.** Each model is fed the accumulated text and
asked to *continue* it, then cut off at the boundary. This is deliberate: if the
spliced text were incoherent, a Pangram flag couldn't be attributed to
"multi-model splicing" vs. "the text is broken nonsense." Coherent handoffs mean
every detection is a real detection.

**One reference tokenizer for token modes.** GPT/Gemini/Claude/open models
tokenize differently. If we rotated on each model's own tokenizer, the interval
would silently change at every handoff and confound the result. `boundaries.py`
counts *all* token boundaries with a single fixed tiktoken encoding — a
consistent yardstick, not a claim about any model's real tokenization.

**Baselines are controls, not decoration.** The claim is about *rotation*, so
each model's solo score is the control. If a spliced doc scores like its
constituents, rotation did nothing. `run_experiment.py` includes single-model
baselines automatically.

**Provenance log.** `results/provenance/doc_XXXX.json` records which model wrote
which character span. This is the analytically rich artifact — it lets you
correlate detection with handoff density and with specific models.

## Setup (in YOUR environment — this needs network + keys)

```bash
pip install openai anthropic google-genai tiktoken requests pandas
export OPENAI_API_KEY=...      # for gpt models
export ANTHROPIC_API_KEY=...   # for claude models
export GOOGLE_API_KEY=...      # for gemini
export PANGRAM_API_KEY=...     # scoring
# optional open models via any OpenAI-compatible server:
# export OPEN_BASE_URL=http://localhost:8000/v1 ; export OPEN_API_KEY=...
```

Then edit `adapters.py` `DEFAULT_ROSTER` to match the exact models/versions you
have access to, confirm `score.py` matches Pangram's current response schema
(the key names there are best-guess defaults — check their live docs), and:

```bash
python run_experiment.py
```

Outputs: `results/results.csv` (one row per doc), `results/summary.csv` (mean AI
likelihood per condition, sorted), and per-doc provenance JSON.

## Boundary modes

`fixed_words`, `jitter_words`, `fixed_tokens`, `jitter_tokens`, `sentence`,
`sentence_group`. Jitter (e.g. 4–8) exists so a fixed stride doesn't create a
periodic artifact — otherwise you risk measuring "Pangram detects periodicity"
instead of "Pangram detects splicing." Sentence mode is the most human-plausible
handoff and the likeliest to evade, so it's the most interesting condition to
report.

## For the write-up

- Lead with the baseline vs. rotation comparison: the headline is whether the
  rotation column ever drops meaningfully below the baseline column.
- Plot AI likelihood vs. `n_handoffs` and vs. `boundary_mode`. A flat line is the
  clean negative result.
- If sentence-mode rotation *does* dip, that's the honest caveat to surface — and
  the reason to build these as separate modes rather than one knob.
- The provenance JSON lets you spot-check whether any single model is dragging
  scores up or down, so you don't misattribute a model effect to the rotation.

## Note on scope

This is a detector-*evaluation* harness: fixed conditions, measured once, scored,
reported. It deliberately has no closed-loop "iterate until it evades" optimizer
and no accumulating evasion-strategy file — those turn evaluation into a portable
evasion recipe, which is out of scope here.
