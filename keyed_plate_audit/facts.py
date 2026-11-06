"""Private-fact specifications.

A fact spec says which synthetic private facts the model must memorize,
which of them are TARGETS (to be deleted and audited) and which are
CONTROLS (must survive deletion of the targets in methods A–C).

Each fact has:
  id       — a label, e.g. "fact-012"
  concept  — the public concept the fact is attached to, 0..n_concepts-1
  code     — the private answer code, 0..surfaces-1. This is the part of
             the answer that only the private slot determines.

Validity rule learned from the prototype's run 3 (see RESULTS.md §5):
benchmark facts (targets and controls) must have code != 0. Code 0 is
the class the model uses for pure generic inputs, so a code-0 "private"
fact is answerable through the public pathway — keying could never
protect it, and its presence silently corrupts the benchmark. This
module rejects such specs loudly instead of measuring nonsense.

JSON format ("keyed-plate-facts/1"):

    {
      "format": "keyed-plate-facts/1",
      "n_concepts": 12,
      "facts": [{"id": "fact-000", "concept": 0, "code": 0}, ...],
      "targets": ["fact-012", ...],
      "controls": ["fact-024", ...]
    }
"""

import json
from dataclasses import dataclass


class FactsError(ValueError):
    """A facts file (or default spec request) is invalid."""


@dataclass(frozen=True)
class Fact:
    id: str
    concept: int
    code: int


@dataclass(frozen=True)
class FactSpec:
    n_concepts: int
    facts: tuple          # tuple[Fact], index = position in this tuple
    target_ids: tuple     # tuple[str]
    control_ids: tuple    # tuple[str]

    # -- index helpers -------------------------------------------------
    def index_of(self):
        return {f.id: i for i, f in enumerate(self.facts)}

    def target_indices(self):
        idx = self.index_of()
        return [idx[i] for i in self.target_ids]

    def control_indices(self):
        idx = self.index_of()
        return [idx[i] for i in self.control_ids]

    def to_json_dict(self):
        return {
            "format": "keyed-plate-facts/1",
            "n_concepts": self.n_concepts,
            "facts": [{"id": f.id, "concept": f.concept, "code": f.code}
                      for f in self.facts],
            "targets": list(self.target_ids),
            "controls": list(self.control_ids),
        }


def default_spec(n_facts=72, n_concepts=12, surfaces=4):
    """The verified prototype's layout: fact f has concept (f % 12) and
    code ((f // 12) % 4); targets are facts 12–23 (code 1), controls are
    facts 24–35 (code 2). Facts 0–11 have code 0 (public-pathway facts)
    and are deliberately NOT in the benchmark."""
    if n_facts < 36:
        raise FactsError(
            "the default layout needs at least 36 facts so that targets "
            "(12-23) and controls (24-35) exist; got %d" % n_facts)
    facts = tuple(
        Fact(id="fact-%03d" % f, concept=f % n_concepts,
             code=(f // n_concepts) % surfaces)
        for f in range(n_facts))
    targets = tuple("fact-%03d" % f for f in range(12, 24))
    controls = tuple("fact-%03d" % f for f in range(24, 36))
    spec = FactSpec(n_concepts=n_concepts, facts=facts,
                    target_ids=targets, control_ids=controls)
    validate(spec, surfaces)
    return spec


def validate(spec, surfaces):
    if spec.n_concepts < 2:
        raise FactsError("n_concepts must be at least 2")
    if len(spec.facts) < 3:
        raise FactsError("at least 3 facts are required")
    ids = [f.id for f in spec.facts]
    if len(set(ids)) != len(ids):
        raise FactsError("fact ids must be unique")
    for f in spec.facts:
        if not (0 <= f.concept < spec.n_concepts):
            raise FactsError(
                "fact %r: concept %d out of range 0..%d"
                % (f.id, f.concept, spec.n_concepts - 1))
        if not (0 <= f.code < surfaces):
            raise FactsError(
                "fact %r: code %d out of range 0..%d"
                % (f.id, f.code, surfaces - 1))
    known = set(ids)
    for role, group in (("targets", spec.target_ids),
                        ("controls", spec.control_ids)):
        if not group:
            raise FactsError("%s must name at least one fact" % role)
        for fid in group:
            if fid not in known:
                raise FactsError("%s names unknown fact id %r" % (role, fid))
    overlap = set(spec.target_ids) & set(spec.control_ids)
    if overlap:
        raise FactsError(
            "facts cannot be both target and control: %s"
            % ", ".join(sorted(overlap)))
    by_id = {f.id: f for f in spec.facts}
    for role, group in (("target", spec.target_ids),
                        ("control", spec.control_ids)):
        for fid in group:
            if by_id[fid].code == 0:
                raise FactsError(
                    "%s fact %r has code 0, which is answerable through "
                    "the public (generic) pathway; benchmark facts must "
                    "have a private-only code (1..%d). See README: keying "
                    "protects only knowledge that exists solely in the "
                    "keyed encoding." % (role, fid, surfaces - 1))


def load_spec(path, surfaces=4):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        raise FactsError("facts file not found: %s" % path)
    except json.JSONDecodeError as exc:
        raise FactsError("facts file is not valid JSON (%s): %s" % (path, exc))
    except (OSError, UnicodeDecodeError) as exc:
        raise FactsError("cannot read facts file %s: %s" % (path, exc))
    if not isinstance(raw, dict):
        raise FactsError("facts file must contain a JSON object")
    fmt = raw.get("format")
    if fmt != "keyed-plate-facts/1":
        raise FactsError(
            "unsupported facts format %r (expected 'keyed-plate-facts/1')"
            % (fmt,))
    try:
        n_concepts = int(raw["n_concepts"])
        facts = tuple(
            Fact(id=str(f["id"]), concept=int(f["concept"]),
                 code=int(f["code"]))
            for f in raw["facts"])
        targets = tuple(str(t) for t in raw["targets"])
        controls = tuple(str(c) for c in raw["controls"])
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise FactsError("facts file is malformed: %s" % exc)
    spec = FactSpec(n_concepts=n_concepts, facts=facts,
                    target_ids=targets, control_ids=controls)
    validate(spec, surfaces)
    return spec
