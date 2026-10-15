"""Tests for species_benchmark.taxonomy."""

from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from species_benchmark.taxonomy import (
    PrimateTaxonomy,
    _main,
    audit_names,
    epithet_stem,
    normalize_name,
    read_names_file,
)

MDD_DIR = Path("C:/pfspecies/mdd/MDD")

SPECIES_ROWS = [
    # sciName, order, family, genus, epithet, common, extinct, subspecies
    ("Macaca_mulatta", "Primates", "Cercopithecidae", "Macaca", "mulatta", "Rhesus Macaque", 0, None),
    ("Macaca_fascicularis", "Primates", "Cercopithecidae", "Macaca", "fascicularis", "Crab-eating Macaque", 0,
     "_M. f. fascicularis_ (Raffles, 1821); _M. f. philippensis_ Geoffroy, 1843"),
    ("Chlorocebus_aethiops", "Primates", "Cercopithecidae", "Chlorocebus", "aethiops", "Grivet", 0, None),
    ("Trachypithecus_geei", "Primates", "Cercopithecidae", "Trachypithecus", "geei", "Gee's Golden Langur", 0, None),
    ("Cercopithecus_mitis", "Primates", "Cercopithecidae", "Cercopithecus", "mitis", "Blue Monkey", 0, None),
    ("Cercopithecus_kandti", "Primates", "Cercopithecidae", "Cercopithecus", "kandti", "Golden Monkey", 0, None),
    ("Plecturocebus_cupreus", "Primates", "Pitheciidae", "Plecturocebus", "cupreus", "Coppery Titi", 0, None),
    ("Callicebus_personatus", "Primates", "Pitheciidae", "Callicebus", "personatus", "Atlantic Titi", 0, None),
    ("Cheracebus_lugens", "Primates", "Pitheciidae", "Cheracebus", "lugens", "Black Titi", 0, None),
    ("Mico_argentatus", "Primates", "Callitrichidae", "Mico", "argentatus", "Silvery Marmoset", 0, None),
    ("Pan_troglodytes", "Primates", "Hominidae", "Pan", "troglodytes", "Chimpanzee", 0, None),
    ("Homo_sapiens", "Primates", "Hominidae", "Homo", "sapiens", "Human", 0, None),
    ("Indri_indri", "Primates", "Indriidae", "Indri", "indri", "Indri", 0, None),
    ("Eulemur_fulvus", "Primates", "Lemuridae", "Eulemur", "fulvus", "Brown Lemur", 0, None),
    ("Varecia_rubra", "Primates", "Lemuridae", "Varecia", "rubra", "Red Ruffed Lemur", 0, None),
    ("Archaeolemur_edwardsi", "Primates", "Archaeolemuridae", "Archaeolemur", "edwardsi", "Extinct Lemur", 1, None),
    ("Canis_lupus", "Carnivora", "Canidae", "Canis", "lupus", "Wolf", 0, None),
]

SYN_ROWS = [
    # MDD_species, order, original combination, normalized original, rank
    ("Chlorocebus aethiops", "Primates", "Cercopithecus Aethiops", "Cercopithecus aethiops", "species"),
    ("Plecturocebus cupreus", "Primates", "Callithrix cuprea", "Callithrix cuprea", "species"),
    ("Cheracebus lugens", "Primates", "Callithrix lugens", "Callithrix lugens", "species"),
    ("Mico argentatus", "Primates", "Simia argentata", "Simia argentata", "species"),
    # Ambiguous name pointing to two species in different genera, same family.
    ("Chlorocebus aethiops", "Primates", "Papio ambiguus", "Papio ambiguus", "species"),
    ("Macaca mulatta", "Primates", "Papio ambiguus", "Papio ambiguus", "species"),
    # Trinomial synonym.
    ("Plecturocebus cupreus", "Primates", "Callicebus moloch discolor", "Callicebus moloch discolor", "subspecies"),
    ("Canis lupus", "Carnivora", "Canis familiaris", "Canis familiaris", "species"),
]


@pytest.fixture()
def tax(tmp_path: Path) -> PrimateTaxonomy:
    """Build a toy taxonomy from miniature MDD-format CSVs."""
    sp = pd.DataFrame(
        SPECIES_ROWS,
        columns=["sciName", "order", "family", "genus", "specificEpithet", "mainCommonName", "extinct", "subspecies"],
    )
    syn = pd.DataFrame(
        SYN_ROWS,
        columns=["MDD_species", "MDD_order", "MDD_original_combination",
                 "MDD_normalized_original_combination", "MDD_original_rank"],
    )
    sp_path, syn_path = tmp_path / "species.csv", tmp_path / "syn.csv"
    sp.to_csv(sp_path, index=False)
    syn.to_csv(syn_path, index=False)
    return PrimateTaxonomy.from_mdd(sp_path, syn_path)


def test_backbone_filters_order_and_extinct(tax: PrimateTaxonomy) -> None:
    assert "Canis lupus" not in tax.species
    assert "Archaeolemur edwardsi" not in tax.species
    assert "Archaeolemur" not in tax.genera
    assert tax.species_to_genus["Macaca mulatta"] == "Macaca"
    assert tax.genus_to_family["Pan"] == "Hominidae"
    assert tax.common_names["Pan troglodytes"] == "Chimpanzee"
    assert "Hominidae" in tax.families


def test_json_roundtrip(tax: PrimateTaxonomy, tmp_path: Path) -> None:
    out = tmp_path / "tax.json"
    tax.to_json(out)
    loaded = PrimateTaxonomy.from_json(out)
    assert loaded.species_list == tax.species_list
    assert loaded.synonyms == tax.synonyms
    assert loaded.resolve("Cercopithecus aethiops") == tax.resolve("Cercopithecus aethiops")


@pytest.mark.parametrize(
    "name",
    ["Macaca mulatta", "macaca_mulatta", "  Macaca   mulatta ", "MACACA MULATTA", "Macaca mulatta (Zimmermann, 1780)"],
)
def test_exact_variants(tax: PrimateTaxonomy, name: str) -> None:
    res = tax.resolve(name)
    assert (res.species, res.level) == ("Macaca mulatta", "species")


def test_author_and_year(tax: PrimateTaxonomy) -> None:
    res = tax.resolve("Trachypithecus geei Khajuria, 1956")
    assert res.species == "Trachypithecus geei"
    assert res.family == "Cercopithecidae"


def test_subspecies(tax: PrimateTaxonomy) -> None:
    assert tax.resolve("Macaca fascicularis philippensis").species == "Macaca fascicularis"
    assert tax.resolve("Pan troglodytes verus").species == "Pan troglodytes"


def test_subspecies_elevated(tax: PrimateTaxonomy) -> None:
    # Binomial not accepted but genus + subspecific epithet is.
    res = tax.resolve("Cercopithecus doggetti kandti")
    assert res.species == "Cercopithecus kandti"
    assert res.method == "subspecies_elevated"


def test_hybrids(tax: PrimateTaxonomy) -> None:
    res = tax.resolve("Macaca fascicularis × mulatta")
    assert (res.genus, res.species, res.level) == ("Macaca", None, "genus")
    res = tax.resolve("Macaca mulatta x Macaca fascicularis")
    assert res.level == "genus"
    res = tax.resolve("Eulemur fulvus × Varecia rubra")
    assert (res.family, res.genus, res.level) == ("Lemuridae", None, "family")
    res = tax.resolve("Pan troglodytes × Eulemur fulvus")
    assert res.level == "unresolved"


@pytest.mark.parametrize("name", ["Macaca sp.", "Macaca spp", "Macaca cf. mulatta", "Macaca aff. mulatta"])
def test_open_nomenclature(tax: PrimateTaxonomy, name: str) -> None:
    res = tax.resolve(name)
    assert (res.genus, res.species, res.level) == ("Macaca", None, "genus")


@pytest.mark.parametrize("name,genus", [("macaca", "Macaca"), ("pan", "Pan"), ("Indri", "Indri"), ("Macaca", "Macaca")])
def test_genus_only(tax: PrimateTaxonomy, name: str, genus: str) -> None:
    res = tax.resolve(name)
    assert (res.genus, res.species, res.level) == (genus, None, "genus")


def test_unknown_epithet_falls_back_to_genus(tax: PrimateTaxonomy) -> None:
    res = tax.resolve("Macaca nonexistens")
    assert (res.genus, res.level, res.method) == ("Macaca", "genus", "genus_fallback")


def test_synonyms(tax: PrimateTaxonomy) -> None:
    res = tax.resolve("Cercopithecus aethiops")
    assert (res.genus, res.species, res.method) == ("Chlorocebus", "Chlorocebus aethiops", "synonym")
    assert tax.resolve("Simia argentata").species == "Mico argentatus"
    assert tax.resolve("Callicebus moloch discolor").species == "Plecturocebus cupreus"


def test_ambiguous_synonym_collapses_to_family(tax: PrimateTaxonomy) -> None:
    res = tax.resolve("Papio ambiguus")
    assert (res.family, res.genus, res.level) == ("Cercopithecidae", None, "family")


def test_callicebus_split_epithet_transfer(tax: PrimateTaxonomy) -> None:
    # Callicebus cupreus is not in the toy synonym table; resolved by
    # epithet transfer within Pitheciidae.
    res = tax.resolve("Callicebus cupreus")
    assert res.species == "Plecturocebus cupreus"
    assert res.method == "epithet_transfer"
    assert tax.resolve("Callicebus lugens").species == "Cheracebus lugens"
    # Gender agreement (argentata vs argentatus) within the same genus.
    assert tax.resolve("Mico argentata").species == "Mico argentatus"


def test_human_maps_to_homo(tax: PrimateTaxonomy) -> None:
    assert tax.resolve("human").species == "Homo sapiens"
    assert tax.resolve("homo").genus == "Homo"


def test_common_name_and_unresolved(tax: PrimateTaxonomy) -> None:
    assert tax.resolve("Rhesus Macaque").species == "Macaca mulatta"
    assert tax.resolve("Canis lupus").level == "unresolved"
    assert tax.resolve("").level == "unresolved"
    assert tax.resolve(None).level == "unresolved"
    assert tax.resolve("adobe_infant").level == "unresolved"


def test_helpers() -> None:
    assert normalize_name("Cercopithecus_l'hoesti ") == "cercopithecus lhoesti"
    assert normalize_name("Pithécia") == "pithecia"
    assert epithet_stem("argentatus") == epithet_stem("argentata")


def test_audit(tax: PrimateTaxonomy, tmp_path: Path) -> None:
    names = tmp_path / "names.txt"
    names.write_text("Macaca mulatta\t8\nmacaca\t1\nCanis lupus\t1\n", encoding="utf-8")
    counts = read_names_file(names)
    assert counts == Counter({"Macaca mulatta": 8, "macaca": 1, "Canis lupus": 1})
    rep = audit_names(tax, counts)
    assert rep["at_least_genus"]["rate_weighted"] == pytest.approx(0.9)
    assert rep["levels"]["species"]["rate_unique"] == pytest.approx(1 / 3)
    assert rep["unresolved"] == [{"name": "Canis lupus", "count": 1}]


@pytest.mark.skipif(not (MDD_DIR / "MDD_v2.5_6904species.csv").exists(), reason="MDD v2.5 not available")
def test_real_mdd_cli(tmp_path: Path) -> None:
    out = tmp_path / "tax.json"
    assert _main(["build", "--mdd-dir", str(MDD_DIR), "--out", str(out)]) == 0
    real = PrimateTaxonomy.from_json(out)
    assert len(real.species) > 450
    assert real.resolve("Cercopithecus aethiops").genus == "Chlorocebus"
    assert real.resolve("Callicebus cupreus").genus == "Plecturocebus"
    assert real.resolve("Callicebus torquatus").genus == "Cheracebus"
    assert real.resolve("Leontocebus fuscicollis").species == "Saguinus fuscicollis"
    assert real.resolve("Cebus imitator").species == "Cebus capucinus"
    assert real.resolve("indri").level == "genus"
    names = tmp_path / "n.txt"
    names.write_text("Macaca mulatta\t3\npseudopotto\t1\n", encoding="utf-8")
    report = tmp_path / "r.json"
    assert _main(["audit", "--taxonomy", str(out), "--names-file", str(names), "--report", str(report)]) == 0
    assert report.exists()
