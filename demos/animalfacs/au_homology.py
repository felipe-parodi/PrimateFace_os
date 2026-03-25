"""Cross-species AU homology mapping.

Maps Action Unit numbers across primate FACS systems based on shared
musculature and published homology tables (Vick et al. 2007; Parr et al. 2010;
Waller et al. 2012; Caeiro et al. 2013).
"""

from typing import Dict, List, Set, Tuple

# ============================================================================
# Per-species AU catalogues
# Keys are canonical AU integers. Values are muscle name / description.
# Sources: published FACS manuals for each species.
# ============================================================================

CHIMP_AUS: Dict[int, str] = {
    1: "Inner Brow Raise (frontalis, pars medialis)",
    2: "Outer Brow Raise (frontalis, pars lateralis)",
    5: "Upper Lid Raise (levator palpebrae superioris)",
    6: "Cheek Raise (orbicularis oculi, pars orbitalis)",
    7: "Lid Tightener (orbicularis oculi, pars palpebralis)",
    8: "Lips Toward Each Other",
    9: "Nose Wrinkle (levator labii superioris alaeque nasi)",
    10: "Upper Lip Raise (levator labii superioris)",
    12: "Lip Corner Pull (zygomaticus major)",
    16: "Lower Lip Depress (depressor labii inferioris)",
    17: "Chin Raise (mentalis)",
    18: "Lip Pucker (incisivii labii / orbicularis oris)",
    22: "Lip Funnel (orbicularis oris)",
    24: "Lip Press (orbicularis oris)",
    25: "Lips Part",
    26: "Jaw Drop (masseter; relaxation of pterygoids)",
    27: "Mouth Stretch",
    43: "Eyes Closed (relaxation of levator palpebrae)",
    45: "Blink",
}

MACAQUE_AUS: Dict[int, str] = {
    1: "Inner Brow Raise (frontalis, pars medialis)",
    2: "Outer Brow Raise (frontalis, pars lateralis)",
    5: "Upper Lid Raise (levator palpebrae superioris)",
    6: "Cheek Raise (orbicularis oculi, pars orbitalis)",
    7: "Lid Tightener (orbicularis oculi, pars palpebralis)",
    8: "Lips Toward Each Other",
    9: "Nose Wrinkle (levator labii superioris alaeque nasi)",
    10: "Upper Lip Raise (levator labii superioris)",
    12: "Lip Corner Pull (zygomaticus major)",
    16: "Lower Lip Depress (depressor labii inferioris)",
    17: "Chin Raise (mentalis)",
    18: "Lip Pucker (incisivii labii / orbicularis oris)",
    22: "Lip Funnel (orbicularis oris)",
    24: "Lip Press (orbicularis oris)",
    25: "Lips Part",
    26: "Jaw Drop",
    27: "Mouth Stretch",
    45: "Blink",
    # MaqFACS-specific ADs (action descriptors)
    # 160: "Lip corner depress" — AD, not canonical AU
}

GIBBON_AUS: Dict[int, str] = {
    1: "Inner Brow Raise (frontalis, pars medialis)",
    2: "Outer Brow Raise (frontalis, pars lateralis)",
    5: "Upper Lid Raise (levator palpebrae superioris)",
    6: "Cheek Raise (orbicularis oculi, pars orbitalis)",
    7: "Lid Tightener (orbicularis oculi, pars palpebralis)",
    10: "Upper Lip Raise (levator labii superioris)",
    12: "Lip Corner Pull (zygomaticus major)",
    16: "Lower Lip Depress (depressor labii inferioris)",
    18: "Lip Pucker (incisivii labii / orbicularis oris)",
    25: "Lips Part",
    26: "Jaw Drop",
    27: "Mouth Stretch",
    45: "Blink",
}

ORANGUTAN_AUS: Dict[int, str] = {
    1: "Inner Brow Raise (frontalis, pars medialis)",
    2: "Outer Brow Raise (frontalis, pars lateralis)",
    5: "Upper Lid Raise (levator palpebrae superioris)",
    6: "Cheek Raise (orbicularis oculi, pars orbitalis)",
    7: "Lid Tightener (orbicularis oculi, pars palpebralis)",
    9: "Nose Wrinkle (levator labii superioris alaeque nasi)",
    10: "Upper Lip Raise (levator labii superioris)",
    12: "Lip Corner Pull (zygomaticus major)",
    16: "Lower Lip Depress (depressor labii inferioris)",
    17: "Chin Raise (mentalis)",
    18: "Lip Pucker (incisivii labii / orbicularis oris)",
    22: "Lip Funnel (orbicularis oris)",
    24: "Lip Press (orbicularis oris)",
    25: "Lips Part",
    26: "Jaw Drop",
    27: "Mouth Stretch",
    43: "Eyes Closed",
    45: "Blink",
}

MARMOSET_AUS: Dict[int, str] = {
    1: "Inner Brow Raise (frontalis, pars medialis)",
    2: "Outer Brow Raise (frontalis, pars lateralis)",
    5: "Upper Lid Raise (levator palpebrae superioris)",
    7: "Lid Tightener (orbicularis oculi, pars palpebralis)",
    10: "Upper Lip Raise (levator labii superioris)",
    12: "Lip Corner Pull (zygomaticus major)",
    16: "Lower Lip Depress (depressor labii inferioris)",
    18: "Lip Pucker",
    25: "Lips Part",
    26: "Jaw Drop",
    27: "Mouth Stretch",
    45: "Blink",
}

GORILLA_AUS: Dict[int, str] = {
    1: "Inner Brow Raise (frontalis, pars medialis)",
    2: "Outer Brow Raise (frontalis, pars lateralis)",
    5: "Upper Lid Raise (levator palpebrae superioris)",
    6: "Cheek Raise (orbicularis oculi, pars orbitalis)",
    7: "Lid Tightener (orbicularis oculi, pars palpebralis)",
    9: "Nose Wrinkle (levator labii superioris alaeque nasi)",
    10: "Upper Lip Raise (levator labii superioris)",
    12: "Lip Corner Pull (zygomaticus major)",
    16: "Lower Lip Depress (depressor labii inferioris)",
    17: "Chin Raise (mentalis)",
    18: "Lip Pucker",
    22: "Lip Funnel (orbicularis oris)",
    24: "Lip Press (orbicularis oris)",
    25: "Lips Part",
    26: "Jaw Drop",
    27: "Mouth Stretch",
    43: "Eyes Closed",
    45: "Blink",
}

# Master lookup
SPECIES_AU_CATALOGUES: Dict[str, Dict[int, str]] = {
    "chimp": CHIMP_AUS,
    "macaque": MACAQUE_AUS,
    "gibbon": GIBBON_AUS,
    "orangutan": ORANGUTAN_AUS,
    "marmoset": MARMOSET_AUS,
    "gorilla": GORILLA_AUS,
}

# ============================================================================
# AU → landmark proxy mapping
# Which PrimateFace kinematics features are proxies for which AUs.
# ============================================================================

AU_LANDMARK_PROXIES: Dict[int, List[str]] = {
    1: ["brow_height_left", "brow_height_right"],  # Inner brow raise
    2: ["brow_height_left", "brow_height_right"],  # Outer brow raise
    5: ["eye_aperture_left", "eye_aperture_right"],  # Upper lid raise
    6: ["eye_aperture_left", "eye_aperture_right"],  # Cheek raise (narrows eye)
    7: ["eye_aperture_left", "eye_aperture_right"],  # Lid tightener
    9: ["nose_length"],  # Nose wrinkle
    10: ["mouth_aperture", "nose_length"],  # Upper lip raise
    12: ["mouth_width"],  # Lip corner pull
    16: ["mouth_aperture"],  # Lower lip depress
    17: ["mouth_aperture", "face_height"],  # Chin raise
    18: ["mouth_width", "mouth_aspect_ratio"],  # Lip pucker
    25: ["mouth_aperture", "mouth_aspect_ratio"],  # Lips part
    26: ["mouth_aperture", "jaw_width", "mouth_aspect_ratio"],  # Jaw drop
    27: ["mouth_aperture", "mouth_aspect_ratio"],  # Mouth stretch
    45: ["eye_aperture_left", "eye_aperture_right"],  # Blink
}


def get_shared_aus(species_list: List[str]) -> Set[int]:
    """Return AUs present in ALL listed species.

    Args:
        species_list: List of species IDs (e.g. ["chimp", "macaque"]).

    Returns:
        Set of AU integers shared across all listed species.
    """
    if not species_list:
        return set()
    sets = [set(SPECIES_AU_CATALOGUES[s].keys()) for s in species_list
            if s in SPECIES_AU_CATALOGUES]
    if not sets:
        return set()
    return sets[0].intersection(*sets[1:])


def get_species_aus(species_id: str) -> Set[int]:
    """Return AU set for a single species.

    Args:
        species_id: Species identifier.

    Returns:
        Set of AU integers for the species.
    """
    return set(SPECIES_AU_CATALOGUES.get(species_id, {}).keys())


def build_homology_matrix() -> Tuple[List[int], List[str], List[List[bool]]]:
    """Build a species × AU presence/absence matrix.

    Returns:
        Tuple of (au_list, species_list, matrix) where matrix[i][j]
        indicates whether species j has AU au_list[i].
    """
    all_aus: Set[int] = set()
    species_list = sorted(SPECIES_AU_CATALOGUES.keys())
    for cat in SPECIES_AU_CATALOGUES.values():
        all_aus.update(cat.keys())
    au_list = sorted(all_aus)
    matrix = []
    for au in au_list:
        row = [au in SPECIES_AU_CATALOGUES[sp] for sp in species_list]
        matrix.append(row)
    return au_list, species_list, matrix
