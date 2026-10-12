"""Primate taxonomy resolver built on the Mammal Diversity Database (MDD).

The resolver maps free-text scientific names (iNaturalist, GBIF, Bing folder
names, legacy literature names) onto the MDD primate backbone at the finest
level that can be justified: species, genus, or family.

Resolution steps, in order:

1. Manual aliases (:data:`MANUAL_ALIASES`) for names where iNaturalist or GBIF
   diverge from MDD and the MDD synonym table does not cover the usage.
2. Exact accepted MDD species (trinomials are reduced to the binomial unless
   the trinomial is listed as a synonym of another species).
3. MDD synonym table (original combinations, apostrophes and diacritics
   removed). A synonym that points to several accepted species resolves to
   their shared genus or family.
4. Elevated subspecies (``Genus species subspecies`` where
   ``Genus subspecies`` is an accepted MDD species).
5. Epithet transfer within a family (e.g. iNaturalist ``Callicebus cupreus``
   -> MDD ``Plecturocebus cupreus``), allowing for Latin gender agreement.
6. Genus-only fallback (accepted or synonymised genus).
7. Unique MDD common names (``Rhesus Macaque``). Common names are tried
   last so that folder names such as ``indri`` resolve to the genus.

Hybrids (``x`` / ``×``) resolve to the shared genus, or the shared family for
intergeneric hybrids. Open nomenclature (``sp.``, ``spp.``, ``cf.``, ``aff.``)
resolves to genus.

``Homo sapiens`` is part of MDD v2.5 and is kept in the taxonomy, so
``human`` and ``homo`` resolve to Homo (species/genus level). Downstream code
that wants non-human primates only should drop ``genus == "Homo"``.

CLI::

    python -m species_benchmark.taxonomy build --mdd-dir C:/pfspecies/mdd/MDD \
        --out C:/pfspecies/data/taxonomy_mdd_v2.5_primates.json
    python -m species_benchmark.taxonomy audit --taxonomy <json> \
        --names-file names.txt [names2.txt ...] --report report.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

MDD_SPECIES_CSV = "MDD_v2.5_6904species.csv"
MDD_SYNONYM_CSV = "Species_Syn_Current_v2.5.csv"
LEVELS: tuple[str, ...] = ("species", "genus", "family", "unresolved")

# Manual aliases for iNaturalist/GBIF usages that diverge from MDD v2.5 and
# are not recoverable from the MDD synonym table or the epithet-transfer rule.
# Keys are normalised (lowercase, single spaces); values are accepted MDD
# names: "Genus epithet" (species), "Genus" (genus), or "family:Name".
MANUAL_ALIASES: dict[str, str] = {
    # Humans (MDD v2.5 includes Homo sapiens).
    "human": "Homo sapiens",
    "humans": "Homo sapiens",
    "homo": "Homo",
    # Bing folder-name artifacts (not a taxon).
    "adobe infant": "",
    # Genera recognised by iNaturalist/GBIF but sunk in MDD v2.5. The MDD
    # synonym table maps Leontocebus to several callitrichid genera (older
    # combinations), so it would otherwise collapse to family; MDD v2.5
    # places all saddleback tamarins in Saguinus.
    "leontocebus": "Saguinus",
    # Pseudopotto martini (Schwartz 1996) is not recognised by MDD v2.5 and
    # is generally regarded as falling within Perodicticus variation.
    "pseudopotto": "Perodicticus",
    "pseudopotto martini": "Perodicticus",
    # Higher-rank labels occasionally used as scientific names.
    "cebidae": "family:Cebidae",
    "cercopithecidae": "family:Cercopithecidae",
    "lemuridae": "family:Lemuridae",
    "hylobatidae": "family:Hylobatidae",
    "atelidae": "family:Atelidae",
    "callitrichidae": "family:Callitrichidae",
    "pitheciidae": "family:Pitheciidae",
    "galagidae": "family:Galagidae",
    "lorisidae": "family:Lorisidae",
    "indriidae": "family:Indriidae",
    "cheirogaleidae": "family:Cheirogaleidae",
    "hominidae": "family:Hominidae",
    "tarsiidae": "family:Tarsiidae",
    "aotidae": "family:Aotidae",
}

_QUALIFIERS = {"sp", "spp", "ssp", "subsp", "cf", "aff", "nr", "var", "f", "indet", "?"}
_HYBRID_TOKENS = {"x", "×"}
_AUTHOR_PARTICLES = {"de", "da", "du", "van", "von", "der", "den", "la", "le", "d", "in", "and", "&", "et"}
_EPITHET_RE = re.compile(r"^[a-z][a-z\-]*$")
_SSP_RE = re.compile(r"_[A-Z]\. [a-z]\. ([a-z\-]+)_")


@dataclass(frozen=True)
class Resolution:
    """Result of resolving a name against the taxonomy.

    Attributes:
        family: Accepted MDD family, or None if unresolved.
        genus: Accepted MDD genus, or None if resolved only to family.
        species: Accepted MDD binomial ("Genus epithet"), or None.
        level: One of 'species', 'genus', 'family', 'unresolved'.
        method: Short tag naming the rule that produced the resolution.
    """

    family: str | None
    genus: str | None
    species: str | None
    level: str
    method: str


UNRESOLVED_EMPTY = Resolution(None, None, None, "unresolved", "empty")


def normalize_name(name: str) -> str:
    """Normalise a name for lookup.

    Folds diacritics to ASCII, maps underscores to spaces, removes apostrophes,
    lowercases, and collapses whitespace. The hybrid sign ``×`` is preserved
    as a separate token.

    Args:
        name: Raw name.

    Returns:
        Normalised lowercase name.
    """
    text = str(name).replace("×", " × ").replace("_", " ")
    folded = "".join(
        ch if ch == "×" else unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode("ascii")
        for ch in text
    )
    folded = folded.replace("'", "").replace("’", "")
    return " ".join(folded.lower().split())


def epithet_stem(epithet: str) -> str:
    """Strip common Latin gender endings so agreement variants compare equal.

    Args:
        epithet: Lowercase specific epithet.

    Returns:
        Stem used for gender-insensitive matching.
    """
    for suffix in ("ensis", "ense", "us", "um", "a", "is", "e", "i"):
        if epithet.endswith(suffix) and len(epithet) - len(suffix) >= 3:
            return epithet[: -len(suffix)]
    if epithet.endswith("er") and len(epithet) > 5:
        return epithet[:-2]
    if epithet.endswith("ra") and len(epithet) > 5:
        return epithet[:-2]
    return epithet


def _parse_tokens(raw: str) -> tuple[list[str], bool, bool]:
    """Split a raw name into genus/epithet tokens.

    Args:
        raw: Raw name (not normalised).

    Returns:
        Tuple (tokens, has_qualifier, is_hybrid). ``tokens`` holds the
        lowercase genus followed by lowercase epithets; author strings, years
        and parenthetical content are dropped. For hybrids, tokens are the
        full normalised token list including the hybrid marker.
    """
    text = re.sub(r"\(.*?\)", " ", str(raw).replace("_", " "))
    text = text.replace("×", " × ")
    parts = text.split()
    if not parts:
        return [], False, False
    if text == text.upper():
        text = text.lower()
    all_lower = text == text.lower()
    tokens: list[str] = []
    qualifier = False
    hybrid = False
    for i, part in enumerate(parts):
        bare = part.strip(".,;:")
        low = normalize_name(bare) if bare != "×" else "×"
        if i == 0:
            tokens.append(low)
            continue
        if low in _HYBRID_TOKENS:
            hybrid = True
            tokens.append("×")
            continue
        if low in _QUALIFIERS or bare.lower().rstrip(".") in _QUALIFIERS:
            qualifier = True
            continue
        if hybrid:
            tokens.append(low)
            continue
        if any(ch.isdigit() for ch in bare) or (not all_lower and bare[:1].isupper()):
            break
        if low in _AUTHOR_PARTICLES:
            break
        if not _EPITHET_RE.match(low):
            break
        tokens.append(low)
        if all_lower and len(tokens) >= 3:
            break
    return tokens, qualifier, hybrid


class PrimateTaxonomy:
    """Primate backbone (families, genera, species) with name resolution.

    Attributes:
        species: Mapping accepted binomial -> {'genus', 'family', 'common_name'}.
        genus_to_family: Mapping accepted genus -> family.
        synonyms: Mapping normalised synonym -> target string ("Genus epithet",
            "Genus", or "family:Name").
        genus_synonyms: Mapping normalised synonym genus -> target string.
        aliases: Manual alias mapping (normalised key -> target string).
        source: Free-text description of the data source.
    """

    def __init__(
        self,
        species: dict[str, dict[str, str | None]],
        genus_to_family: dict[str, str],
        synonyms: dict[str, str] | None = None,
        genus_synonyms: dict[str, str] | None = None,
        subspecies: dict[str, str] | None = None,
        aliases: dict[str, str] | None = None,
        source: str = "",
    ) -> None:
        """Initialise from pre-built tables.

        Args:
            species: Accepted binomial -> metadata dict.
            genus_to_family: Accepted genus -> family.
            synonyms: Normalised synonym name -> target.
            genus_synonyms: Normalised synonym genus -> target.
            subspecies: Normalised accepted trinomial -> accepted binomial.
            aliases: Manual aliases; defaults to :data:`MANUAL_ALIASES`.
            source: Data source description.
        """
        self.species = dict(species)
        self.genus_to_family = dict(genus_to_family)
        self.synonyms = dict(synonyms or {})
        self.genus_synonyms = dict(genus_synonyms or {})
        self.subspecies = dict(subspecies or {})
        self.aliases = dict(MANUAL_ALIASES if aliases is None else aliases)
        self.source = source
        self._species_lc = {normalize_name(s): s for s in self.species}
        self._genus_lc = {g.lower(): g for g in self.genus_to_family}
        self._family_lc = {f.lower(): f for f in self.families}
        stems: dict[tuple[str, str], list[str]] = defaultdict(list)
        for sp, meta in self.species.items():
            stems[(str(meta["family"]), epithet_stem(sp.split()[1]))].append(sp)
        self._family_stems = dict(stems)
        common: dict[str, list[str]] = defaultdict(list)
        for sp, meta in self.species.items():
            if meta.get("common_name"):
                common[normalize_name(str(meta["common_name"]))].append(sp)
        self._common = {k: v[0] for k, v in common.items() if len(v) == 1}

    # ------------------------------------------------------------------ build
    @property
    def families(self) -> list[str]:
        """Sorted list of accepted families."""
        return sorted(set(self.genus_to_family.values()))

    @property
    def genera(self) -> list[str]:
        """Sorted list of accepted genera."""
        return sorted(self.genus_to_family)

    @property
    def species_list(self) -> list[str]:
        """Sorted list of accepted binomials."""
        return sorted(self.species)

    @property
    def species_to_genus(self) -> dict[str, str]:
        """Mapping accepted binomial -> genus."""
        return {s: str(m["genus"]) for s, m in self.species.items()}

    @property
    def common_names(self) -> dict[str, str | None]:
        """Mapping accepted binomial -> MDD main common name."""
        return {s: m.get("common_name") for s, m in self.species.items()}

    @classmethod
    def from_mdd(
        cls,
        species_csv: Path,
        synonym_csv: Path | None = None,
        include_extinct: bool = False,
    ) -> PrimateTaxonomy:
        """Build the primate taxonomy from MDD CSV exports.

        Args:
            species_csv: Path to ``MDD_v2.5_6904species.csv`` (or equivalent).
            synonym_csv: Optional path to ``Species_Syn_Current_v2.5.csv``.
            include_extinct: Keep species flagged extinct in MDD.

        Returns:
            Populated :class:`PrimateTaxonomy`.
        """
        df = _read_csv(
            Path(species_csv),
            usecols=["sciName", "order", "family", "genus", "specificEpithet", "mainCommonName", "extinct", "subspecies"],
        )
        df = df[df["order"].astype(str).str.upper() == "PRIMATES"]
        if not include_extinct:
            df = df[pd.to_numeric(df["extinct"], errors="coerce").fillna(0).astype(int) == 0]
        species: dict[str, dict[str, str | None]] = {}
        genus_to_family: dict[str, str] = {}
        subspecies: dict[str, str] = {}
        for row in df.itertuples(index=False):
            binomial = f"{row.genus} {row.specificEpithet}"
            common = row.mainCommonName if isinstance(row.mainCommonName, str) else None
            species[binomial] = {"genus": row.genus, "family": row.family, "common_name": common}
            genus_to_family[row.genus] = row.family
            if isinstance(row.subspecies, str):
                for ssp in _SSP_RE.findall(row.subspecies):
                    subspecies[f"{binomial} {ssp}".lower()] = binomial
        synonyms: dict[str, str] = {}
        genus_synonyms: dict[str, str] = {}
        if synonym_csv is not None:
            synonyms, genus_synonyms = _build_synonyms(Path(synonym_csv), species, genus_to_family)
        return cls(
            species,
            genus_to_family,
            synonyms,
            genus_synonyms,
            subspecies,
            source=f"MDD ({Path(species_csv).name}; {Path(synonym_csv).name if synonym_csv else 'no synonyms'})",
        )

    def to_json(self, path: Path) -> None:
        """Write the taxonomy (including synonym tables and aliases) to JSON.

        Args:
            path: Output path.
        """
        payload = {
            "source": self.source,
            "n_families": len(self.families),
            "n_genera": len(self.genera),
            "n_species": len(self.species),
            "families": self.families,
            "genus_to_family": dict(sorted(self.genus_to_family.items())),
            "species": dict(sorted(self.species.items())),
            "subspecies": dict(sorted(self.subspecies.items())),
            "synonyms": dict(sorted(self.synonyms.items())),
            "genus_synonyms": dict(sorted(self.genus_synonyms.items())),
            "aliases": dict(sorted(self.aliases.items())),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> PrimateTaxonomy:
        """Load a taxonomy written by :meth:`to_json`.

        Args:
            path: JSON path.

        Returns:
            Loaded :class:`PrimateTaxonomy`. Manual aliases in the current
            code are merged over those stored in the file.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        aliases = dict(data.get("aliases", {}))
        aliases.update(MANUAL_ALIASES)
        return cls(
            data["species"],
            data["genus_to_family"],
            data.get("synonyms", {}),
            data.get("genus_synonyms", {}),
            data.get("subspecies", {}),
            aliases=aliases,
            source=data.get("source", ""),
        )

    # ---------------------------------------------------------------- resolve
    def _target(self, target: str, method: str) -> Resolution:
        """Convert a stored target string into a Resolution.

        Args:
            target: "Genus epithet", "Genus", "family:Name", or "" (unresolved).
            method: Method tag.

        Returns:
            Resolution for the target.
        """
        if not target:
            return Resolution(None, None, None, "unresolved", method)
        if target.startswith("family:"):
            fam = target.split(":", 1)[1]
            return Resolution(fam, None, None, "family", method)
        if " " in target:
            meta = self.species[target]
            return Resolution(str(meta["family"]), str(meta["genus"]), target, "species", method)
        return Resolution(self.genus_to_family[target], target, None, "genus", method)

    def _genus_of(self, token: str) -> tuple[str, str] | None:
        """Resolve a lowercase genus token to an accepted genus or family target.

        Args:
            token: Lowercase genus token.

        Returns:
            Tuple (target, method) or None.
        """
        if token in self._genus_lc:
            return self._genus_lc[token], "genus"
        if token in self.genus_synonyms:
            return self.genus_synonyms[token], "genus_synonym"
        if token in self._family_lc:
            return "family:" + self._family_lc[token], "family_name"
        return None

    def _resolve_binomial(self, tokens: Sequence[str]) -> Resolution | None:
        """Resolve genus + epithet (+ subspecies) tokens to species if possible.

        Args:
            tokens: Lowercase tokens, at least two.

        Returns:
            Species- or higher-level Resolution, or None.
        """
        binom = " ".join(tokens[:2])
        full = " ".join(tokens[:3])
        if len(tokens) >= 3:
            if full in self.subspecies:
                return self._target(self.subspecies[full], "subspecies")
            if full in self.synonyms:
                return self._target(self.synonyms[full], "synonym_trinomial")
        if binom in self._species_lc:
            return self._target(self._species_lc[binom], "exact" if len(tokens) == 2 else "subspecies")
        if binom in self.synonyms:
            return self._target(self.synonyms[binom], "synonym")
        if len(tokens) >= 3:
            elevated = f"{tokens[0]} {tokens[2]}"
            if elevated in self._species_lc:
                return self._target(self._species_lc[elevated], "subspecies_elevated")
        genus = self._genus_of(tokens[0])
        if genus is not None and not genus[0].startswith("family:"):
            g = genus[0]
            fam = self.genus_to_family[g]
            cands = self._family_stems.get((fam, epithet_stem(tokens[1])), [])
            same_genus = [c for c in cands if c.split()[0] == g]
            pick = same_genus if same_genus else cands
            if len(pick) == 1:
                return self._target(pick[0], "epithet_transfer" if not same_genus else "epithet_gender")
        return None

    def resolve(self, name: str | None) -> Resolution:
        """Resolve a free-text name to the finest supported MDD level.

        Args:
            name: Scientific name, common name, or folder name. Case,
                underscores, author strings and years are tolerated.

        Returns:
            :class:`Resolution`. ``level == 'unresolved'`` if nothing matched.

        Example:
            >>> tax.resolve("Trachypithecus geei Khajuria, 1956").species
            'Trachypithecus geei'
        """
        if name is None or not str(name).strip():
            return UNRESOLVED_EMPTY
        norm = normalize_name(name)
        if norm in self.aliases:
            return self._target(self.aliases[norm], "alias")
        tokens, qualifier, hybrid = _parse_tokens(name)
        if not tokens:
            return Resolution(None, None, None, "unresolved", "unparsed")
        key = " ".join(tokens)
        if key in self.aliases:
            return self._target(self.aliases[key], "alias")
        if hybrid:
            return self._resolve_hybrid(tokens)
        if len(tokens) >= 2 and not qualifier:
            res = self._resolve_binomial(tokens)
            if res is not None:
                return res
        genus = self._genus_of(tokens[0])
        if genus is not None:
            method = "qualified_genus" if qualifier else ("genus_only" if len(tokens) == 1 else "genus_fallback")
            if genus[1] != "genus":
                method = genus[1]
            return self._target(genus[0], method)
        if norm in self._common:
            return self._target(self._common[norm], "common_name")
        return Resolution(None, None, None, "unresolved", "no_match")

    def _resolve_hybrid(self, tokens: Sequence[str]) -> Resolution:
        """Resolve a hybrid formula to the shared genus or family.

        Args:
            tokens: Lowercase tokens containing the hybrid marker '×'.

        Returns:
            Genus-, family-, or unresolved-level Resolution.
        """
        sides: list[list[str]] = [[]]
        for tok in tokens:
            if tok == "×":
                sides.append([])
            else:
                sides[-1].append(tok)
        first_genus = sides[0][0] if sides[0] else None
        parts: list[Resolution] = []
        for side in sides:
            if not side:
                continue
            if self._genus_of(side[0]) is None and first_genus is not None:
                side = [first_genus] + side
            parts.append(self.resolve(" ".join(side)))
        genera = {p.genus for p in parts}
        families = {p.family for p in parts}
        if not parts or None in families:
            return Resolution(None, None, None, "unresolved", "hybrid")
        if len(genera) == 1 and None not in genera:
            g = next(iter(genera))
            return Resolution(self.genus_to_family[g], g, None, "genus", "hybrid")
        if len(families) == 1:
            return Resolution(next(iter(families)), None, None, "family", "hybrid")
        return Resolution(None, None, None, "unresolved", "hybrid")


def _read_csv(path: Path, usecols: Sequence[str]) -> pd.DataFrame:
    """Read an MDD CSV, tolerating legacy encodings.

    Args:
        path: CSV path.
        usecols: Columns to load.

    Returns:
        DataFrame with the requested columns.
    """
    try:
        return pd.read_csv(path, usecols=list(usecols), low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, usecols=list(usecols), low_memory=False, encoding="latin-1")


def _collapse(targets: Iterable[str], species: dict[str, dict[str, str | None]]) -> str | None:
    """Collapse a set of accepted binomials to the finest shared rank.

    Args:
        targets: Accepted binomials.
        species: Accepted species metadata.

    Returns:
        Binomial, genus, "family:Name", or None if no shared family.
    """
    tset = set(targets)
    if len(tset) == 1:
        return next(iter(tset))
    genera = {species[t]["genus"] for t in tset}
    if len(genera) == 1:
        return str(next(iter(genera)))
    families = {species[t]["family"] for t in tset}
    if len(families) == 1:
        return f"family:{next(iter(families))}"
    return None


def _build_synonyms(
    synonym_csv: Path,
    species: dict[str, dict[str, str | None]],
    genus_to_family: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Build name- and genus-level synonym maps from the MDD synonym table.

    Args:
        synonym_csv: Path to ``Species_Syn_Current_v2.5.csv``.
        species: Accepted species metadata.
        genus_to_family: Accepted genus -> family.

    Returns:
        Tuple (synonyms, genus_synonyms) with normalised keys.
    """
    cols = ["MDD_species", "MDD_order", "MDD_original_combination", "MDD_normalized_original_combination", "MDD_original_rank"]
    syn = _read_csv(synonym_csv, usecols=cols)
    syn = syn[syn["MDD_order"].astype(str).str.upper() == "PRIMATES"]
    accepted_lc = {normalize_name(s) for s in species}
    by_key: dict[str, set] = defaultdict(set)
    by_genus: dict[str, Counter] = defaultdict(Counter)
    for row in syn.itertuples(index=False):
        target = str(row.MDD_species).replace("_", " ")
        if target not in species:
            continue
        for raw in (row.MDD_original_combination, row.MDD_normalized_original_combination):
            if not isinstance(raw, str):
                continue
            words = normalize_name(raw).replace("-", "").split()
            words = [w for w in words if w not in {"var", "subsp", "ssp", "f", "forma"}]
            if len(words) < 2:
                continue
            by_key[" ".join(words[:3])].add(target)
            if len(words) >= 3 and words[1] == words[2]:
                by_key[" ".join(words[:2])].add(target)
            by_genus[words[0]][target] += 1
    synonyms: dict[str, str] = {}
    for key, targets in by_key.items():
        if key in accepted_lc:
            continue
        collapsed = _collapse(targets, species)
        if collapsed is not None:
            synonyms[key] = collapsed
    genus_lc = {g.lower() for g in genus_to_family}
    genus_synonyms: dict[str, str] = {}
    for gkey, counter in by_genus.items():
        if gkey in genus_lc:
            continue
        collapsed = _collapse(counter.keys(), species)
        if collapsed is not None and " " not in collapsed:
            genus_synonyms[gkey] = collapsed
    return synonyms, genus_synonyms


# --------------------------------------------------------------------- audit
def read_names_file(path: Path) -> Counter:
    """Read a names file into a name -> count Counter.

    Each line is either ``name`` (count 1 per occurrence) or
    ``name<TAB>count``.

    Args:
        path: Text file path.

    Returns:
        Counter of names.
    """
    counts: Counter = Counter()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if "\t" in line:
            name, _, cnt = line.rpartition("\t")
            try:
                counts[name.strip()] += int(cnt)
                continue
            except ValueError:
                pass
        counts[line.strip()] += 1
    return counts


def audit_names(tax: PrimateTaxonomy, counts: Counter) -> dict[str, object]:
    """Summarise resolution rates for a name -> count table.

    Args:
        tax: Taxonomy used for resolution.
        counts: Counter of names (weights are row counts).

    Returns:
        Report dict with per-level unique-name and row-weighted rates,
        at-least-genus rates, method counts, and unresolved names.
    """
    total_rows = sum(counts.values())
    level_names: Counter = Counter()
    level_rows: Counter = Counter()
    method_rows: Counter = Counter()
    unresolved: list[tuple[str, int]] = []
    family_only: list[tuple[str, int]] = []
    for name, cnt in counts.items():
        res = tax.resolve(name)
        level_names[res.level] += 1
        level_rows[res.level] += cnt
        method_rows[res.method] += cnt
        if res.level == "unresolved":
            unresolved.append((name, cnt))
        elif res.level == "family":
            family_only.append((name, cnt))
    n_names = len(counts)

    def _rate(c: Counter, total: int, levels: Sequence[str]) -> float:
        return round(sum(c[lv] for lv in levels) / total, 6) if total else 0.0

    return {
        "n_unique_names": n_names,
        "n_rows": total_rows,
        "levels": {
            lv: {
                "unique_names": level_names[lv],
                "rows": level_rows[lv],
                "rate_unique": _rate(level_names, n_names, [lv]),
                "rate_weighted": _rate(level_rows, total_rows, [lv]),
            }
            for lv in LEVELS
        },
        "at_least_genus": {
            "rate_unique": _rate(level_names, n_names, ["species", "genus"]),
            "rate_weighted": _rate(level_rows, total_rows, ["species", "genus"]),
        },
        "methods_rows": dict(method_rows.most_common()),
        "family_only": [{"name": n, "count": c} for n, c in sorted(family_only, key=lambda x: -x[1])],
        "unresolved": [{"name": n, "count": c} for n, c in sorted(unresolved, key=lambda x: -x[1])],
    }


def _main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument list (defaults to sys.argv).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="Build primate taxonomy JSON from MDD CSVs.")
    b.add_argument("--mdd-dir", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--include-extinct", action="store_true")
    a = sub.add_parser("audit", help="Report resolution rates for a names file.")
    a.add_argument("--taxonomy", type=Path, required=True)
    a.add_argument("--names-file", type=Path, nargs="+", required=True)
    a.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.cmd == "build":
        tax = PrimateTaxonomy.from_mdd(
            args.mdd_dir / MDD_SPECIES_CSV, args.mdd_dir / MDD_SYNONYM_CSV, include_extinct=args.include_extinct
        )
        tax.to_json(args.out)
        print(
            f"families={len(tax.families)} genera={len(tax.genera)} species={len(tax.species)} "
            f"synonyms={len(tax.synonyms)} genus_synonyms={len(tax.genus_synonyms)} -> {args.out}"
        )
        return 0

    tax = PrimateTaxonomy.from_json(args.taxonomy)
    combined: Counter = Counter()
    per_file: dict[str, object] = {}
    for f in args.names_file:
        counts = read_names_file(f)
        combined.update(counts)
        per_file[str(f)] = audit_names(tax, counts)
    report = {"taxonomy": str(args.taxonomy), "combined": audit_names(tax, combined), "per_file": per_file}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    for label, rep in [("combined", report["combined"])] + list(per_file.items()):
        assert isinstance(rep, dict)
        lv = rep["levels"]
        print(
            f"{label}: names={rep['n_unique_names']} rows={rep['n_rows']} "
            + " ".join(f"{k}={lv[k]['rate_weighted']:.4f}" for k in LEVELS)
            + f" >=genus={rep['at_least_genus']['rate_weighted']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
