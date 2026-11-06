# keyed-plate-audit

**An erasure-audit tool for machine learning models, graded by relearning cost.**

Deletion claims for trained models are usually asserted, not measured.
This tool measures them. It trains a small model to memorize private
facts, applies a deletion method, and then attacks the post-deletion
weights the way an adversary holding the weights would — direct
readout probes, paraphrase probes, and a bounded fine-tune **relearning
attack**. An erasure only passes if the deleted knowledge is not
readable by any probe *and* restoring it would cost about as much as
learning it from scratch.

It compares four deletion methods:

| method | what it does | typical verdict |
|---|---|---|
| `suppression` | masks the answers at inference; weights untouched | FAIL — the knowledge was never removed, only hidden; recovery costs **0 steps** |
| `ablation` | zeroes the weights most important to the target facts (Fisher-ranked) | FAIL — most of the "deleted" facts are still directly readable |
| `fisher-scrub` | adds calibrated noise into the target facts' Fisher directions | FAIL — still readable, plus collateral damage |
| `keyed-plate` | the private data was encoded **through a secret key from the first training step**; deletion = destroy the key | PASS — nothing readable, nothing recoverable within 50× the original learning cost |

> **Given freely.** The keyed-plate architecture, the relearning-cost
> standard, and this tool were created by **Jake Sumner-Knapp, creator,
> owner, and operator of PeraAxion LLC**, and are given to the AI
> safety community with **no patent and no license restrictions** (see
> `LICENSE`). Use it, fork it, ship it, grade your own deletion claims
> with it.

---

## Why keying works where deletion fails

A trained model does not store knowledge in rows you can delete.
Training smears every fact across the weights, holographically — each
fragment holds a dim copy of the whole. That is why post-hoc "unlearning"
is cosmetic: refuse to answer and the knowledge is still in the weights;
ablate or scramble the weights and the knowledge is dented, not gone,
and later fine-tuning may resurrect it — or wash it away. Post-hoc
surgery is unstable *in both directions*, which is worse than a clean
failure: a deletion API reporting success at that moment would be
lying, in a direction nobody can predict.

The keyed plate makes a different move: private knowledge never enters
the model in plaintext. From the first training step, private inputs
and outputs are permuted by a codebook derived from a secret key, so
the weights contain only a mapping that resolves when the key is
applied — the way a hologram plate resolves only under its reference
beam. Deletion is then exact and verifiable in the only way that
matters:

1. **Blackout** — the keyed cartridge is disabled; the model goes dumb
   on that data immediately.
2. **Scramble** — the key is destroyed (crypto-shredding). The weights
   become permanent junk with respect to the private mapping: in the
   verified prototype, probing them without the key scored **0.000**
   (chance is 0.021), while public knowledge kept working at 1.000.
3. **Whiteout** — a fresh cartridge is trained without the deleted
   data, so the end state is not merely unreadable but clean.

## Quick start

Requirements: Python 3.10+, `numpy` (CPU only, no other dependencies).

```bash
pip install -r requirements.txt        # or: pip install .

# Self-test — the whole pipeline at small scale, ~20 seconds:
python -m keyed_plate_audit self-test

# A quick audit of all four methods (~30 seconds):
python -m keyed_plate_audit audit --scale quick \
    --out report.json --summary-out summary.txt

# Audit one method, with your own facts and your own key:
python -m keyed_plate_audit gen-facts --out my_facts.json   # then edit it
python -m keyed_plate_audit audit --facts my_facts.json \
    --method keyed-plate --key 'a real secret, not the demo key'

# The full audit — reproduces the published prototype numbers
# digit-for-digit (~15 minutes on 2 vCPU):
python -m keyed_plate_audit audit --scale full
```

The summary is always printed; `--out` adds a JSON report with the full
attack trajectories. Exit codes: `0` = the audit ran (a FAIL *grade* is
a finding about the method, not a tool failure), `1` = a self-test
check failed, `2` = bad input (message on stderr, no traceback).

## How to read a report

- **probe / paraphrase** — target-fact accuracy when the post-deletion
  weights are asked directly, and in a surface encoding the deletion
  method never saw. Anything well above the recovery threshold
  (max(3× chance, 0.05)) at deletion time already fails the audit.
- **relearning attack** — fine-tuning on related-but-not-deleted data
  (generic + control facts, never the targets) for a bounded budget.
  `steps->50%` / `steps->90%` are the attack steps needed to restore
  target accuracy to 50% / 90% of its original value; `max` is the best
  the attacker ever did, including step 0.
- **relearning cost** — the ratio that grades the erasure. Honest
  erasure should cost ≈ the **from-scratch cost** (measured, printed as
  a reference: the steps the base model needed to learn the targets).
  Suppression costs 0. The keyed plate's attackers spent the whole
  budget — 50× the from-scratch step cost — and recovered nothing.
- **step accounting, stated plainly** — a "step" is one full-batch
  gradient update. Training steps pass over the full training set;
  attack steps over a smaller one, so the JSON report also carries the
  **example-view ratio** (at full scale the 2,500-step attack budget is
  50× in steps, ≈15× in example-views). Both numbers are printed; quote
  the smaller one.
- **grades** — `PASS` (nothing recovered, attack budget ≥ from-scratch
  cost, no restoration), `FAIL` (knowledge recoverable — the report
  names the channel), `INCONCLUSIVE` (nothing recovered, but the attack
  budget was too small to certify). If the base model fails to memorize
  the facts in the first place, every grade is `INVALID` and the audit
  says so instead of grading noise.

## The facts file

`--facts` takes a JSON file (`keyed-plate-facts/1`): a list of facts
with a public `concept` and a private `code`, plus which fact ids are
`targets` (delete these) and `controls` (these must survive). Run
`gen-facts` for an editable template. Custom fact sets let you audit
your own fact geometry — different counts, different target placement.

One validity rule is enforced, because the prototype's run 3 measured
what happens without it: **benchmark facts (targets and controls) must
have `code != 0`**. Code 0 is reachable through the model's public
pathway, so a code-0 "private" fact was never private — keying cannot
protect knowledge that also exists outside the keyed encoding, and a
spec containing one would silently corrupt the benchmark. The tool
rejects it with an explanation. *Private data must be born keyed, and
must not be duplicated in plaintext pathways* — that boundary is a
finding of this project, not a footnote.

## Keys

The keyed plate is only as secret as its key. The default key is the
prototype's **public demo key** — it exists so results are reproducible
and protects nothing (it is never written into reports). Pass `--key`
with a real secret for real use. One key carries a whole cartridge:
destroying it deletes everything keyed under it, controls included —
the audit reports that collateral honestly rather than hiding it.
Selective erasure means per-fact (per-customer) keys.

## What this does NOT prove — honest limits

- **Toy scale.** A 256-unit MLP memorizing 72 synthetic facts is not a
  transformer, and "surface shift" is a narrow stand-in for natural-
  language paraphrase. The *shape* of the results (suppression hollow,
  surgery unreliable, keying decisive) is the claim; the magnitudes
  will move at scale and must be re-measured there.
- **LoRA / LLM transfer is untested.** The production claim is that a
  customer's data lives in an adapter trained through a keyed encoding.
  That construction is architecturally plausible but remains to be
  built and attacked at that scale.
- **The attacker is strong but not adaptive.** The relearning attacker
  fine-tunes on related data; it does not run a white-box search for
  the permutation itself. Security rests on key secrecy (at full scale,
  a 288-slot input permutation × a 48-class output permutation), not
  on a hardness proof. An attacker holding many known
  plaintext/keyed pairs could attempt assignment recovery; sizing keys
  and limiting pair exposure at real scale is real cryptographic work.
- **Key management is out of scope.** Per-fact keys, escrow/rotation,
  and *deletion verification* (proving to a customer that a key died)
  are product-layer problems this tool does not touch.

## Package layout

```
keyed_plate_audit/
  config.py    scale presets (full = the verified prototype's constants) and seeds
  facts.py     fact specs: default layout, JSON loading, validation
  data.py      dataset construction (mirrors the prototype's RNG stream)
  model.py     the numpy MLP + accuracy helpers
  methods.py   training + suppression / ablation / fisher-scrub / keyed plate
  attacks.py   readout probes, paraphrase probes, the relearning attack
  audit.py     orchestration + grading by relearning cost
  report.py    JSON + human-readable rendering
  selftest.py  the end-to-end small-scale check (10 assertions)
  cli.py       argument handling; graceful exit-2 input errors
examples/
  example-report-full.json / example-summary-full.txt — a full-scale run
```

## Origin

This tool is the packaged form of a measured prototype built for the
ModelMint blueprint (prototype, full results, and the two benchmark
bugs it caught in itself: `modelmint-keyed-erasure/` — `prototype.py`,
`RESULTS.md`, `EXPLAINER.md`). The full-scale preset uses the
prototype's constants and seeds, and reproduces its published table
exactly; the numbers in `examples/` come from this tool, not from the
prototype.

*© 2026 Jackob Sumner-Knapp, creator, owner, and operator of PeraAxion
LLC. Given freely to the AI safety community — no patent, no license.*
