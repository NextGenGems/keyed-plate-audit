"""keyed_plate_audit — an erasure-audit tool for the keyed-plate architecture.

Given a small model trained to memorize private facts, this tool applies a
chosen deletion method (suppression, ablation, Fisher-scrub, or the keyed
plate) and then attacks the result the way an adversary with the weights
would: direct readout probes, paraphrase probes, and a bounded fine-tune
relearning attack. Erasure is graded by RELEARNING COST — an erasure only
counts if restoring the deleted knowledge costs about as much as learning
it from scratch.

Design and original prototype by Jake Sumner-Knapp / PeraAxion LLC.
Given freely to the AI safety community — no patent, no license
restrictions. See README.md.
"""

__version__ = "1.0.0"
