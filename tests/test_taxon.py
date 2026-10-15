"""Tests for the taxon classifier API (uses a tiny randomly initialised model)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from primateface.taxon import EXPORT_FORMAT, TaxonClassifier, TaxonPrediction, _build_net, square_crop  # noqa: E402

SPACE = {
    "family": ["Cercopithecidae", "Hominidae"],
    "genus": ["Macaca", "Pan"],
    "species": ["Macaca fuscata", "Macaca mulatta", "Pan troglodytes"],
    "species_to_genus": {"Macaca fuscata": "Macaca", "Macaca mulatta": "Macaca", "Pan troglodytes": "Pan"},
    "genus_to_family": {"Macaca": "Cercopithecidae", "Pan": "Hominidae"},
    "common": {"Pan troglodytes": "Chimpanzee"},
}


@pytest.fixture()
def ckpt_path(tmp_path):
    net = _build_net("resnet10t", {k: len(SPACE[k]) for k in ("family", "genus", "species")}, cosine=False)
    path = tmp_path / "taxon_test.pt"
    torch.save(
        {
            "format": EXPORT_FORMAT,
            "backbone": "resnet10t",
            "img_size": 64,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "cosine": False,
            "crop_scale": 1.25,
            "temperature": {"family": 1.0, "genus": 1.0, "species": 1.0},
            "thresholds": {"species": 0.5, "genus": 0.3},
            "label_space": SPACE,
            "state_dict": net.state_dict(),
        },
        path,
    )
    return path


def test_square_crop_pads_out_of_bounds():
    img = np.full((50, 80, 3), 255, np.uint8)
    crop = square_crop(img, [70, 40, 90, 60], scale=1.25, size=32)
    assert crop.shape == (32, 32, 3)
    assert crop.min() == 0 and crop.max() == 255  # padded region is black


def test_predict_bboxes_returns_consistent_predictions(ckpt_path):
    clf = TaxonClassifier(ckpt_path)
    img = np.random.randint(0, 255, (120, 160, 3), np.uint8)
    preds = clf.predict_bboxes(img, [[10, 10, 60, 60], [80, 20, 150, 100]])
    assert len(preds) == 2
    for p in preds:
        assert isinstance(p, TaxonPrediction)
        assert p.level in ("species", "genus", "family")
        assert p.family[0] in SPACE["family"] and p.genus[0] in SPACE["genus"]
        assert abs(sum(prob for _, prob in p.top_genus) - 1.0) < 1e-4
        assert set(p.to_dict()) >= {"label", "level", "confidence"}


def test_rollup_thresholds(ckpt_path):
    strict = TaxonClassifier(ckpt_path, species_threshold=1.01, genus_threshold=1.01)
    p = strict.predict_crops([np.zeros((64, 64, 3), np.uint8)])[0]
    assert p.level == "family"
    loose = TaxonClassifier(ckpt_path, species_threshold=0.0, genus_threshold=0.0)
    p = loose.predict_crops([np.zeros((64, 64, 3), np.uint8)])[0]
    assert p.level in ("species", "genus")  # genus if species head disagrees with genus head


def test_rejects_wrong_format(tmp_path):
    bad = tmp_path / "bad.pt"
    torch.save({"format": "other"}, bad)
    with pytest.raises(ValueError):
        TaxonClassifier(bad)


def test_default_taxon_model_is_registered_tinyvit():
    from primateface._model_registry import DEFAULT_TAXON_MODEL, TAXON_MODELS
    from primateface.taxon import resolve_taxon_name

    assert DEFAULT_TAXON_MODEL == "taxon_tinyvit5m"
    assert DEFAULT_TAXON_MODEL in TAXON_MODELS
    assert resolve_taxon_name(True) == DEFAULT_TAXON_MODEL
    assert resolve_taxon_name("default") == DEFAULT_TAXON_MODEL
    assert resolve_taxon_name("taxon_convnext_atto") == "taxon_convnext_atto"
    assert resolve_taxon_name(None) is None and resolve_taxon_name(False) is None


def test_cli_bare_taxon_flag_uses_default():
    import argparse

    from primateface import cli

    seen = {}

    def fake_analyze(args: argparse.Namespace) -> int:
        seen["taxon_model"] = args.taxon_model
        return 0

    cli._cmd_analyze, orig = fake_analyze, cli._cmd_analyze
    try:
        cli.main(["analyze", "x.jpg", "--taxon-model"])
        assert seen["taxon_model"] == "default"
        cli.main(["analyze", "x.jpg"])
        assert seen["taxon_model"] is None
    finally:
        cli._cmd_analyze = orig


def test_face_taxon_property_requires_model():
    from primateface.face import Face

    f = Face(
        bbox=np.array([0, 0, 10, 10], np.float32),
        score=0.9,
        keypoints=np.zeros((68, 3), np.float32),
        _image=np.zeros((20, 20, 3), np.uint8),
        _image_size=(20, 20),
    )
    with pytest.raises(RuntimeError):
        _ = f.taxon
    assert "taxon" not in f.to_dict()
