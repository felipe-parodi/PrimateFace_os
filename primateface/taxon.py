"""Primate taxon (family / genus / species) classification from face crops.

Models are hierarchical classifiers trained on face crops (see
``species_benchmark/`` and ``docs/plans/species-classifier_261012.md``).
Low-confidence species predictions roll up to genus, then family.

Example:
    >>> from primateface.taxon import TaxonClassifier
    >>> clf = TaxonClassifier.from_pretrained("taxon_convnext_atto")
    >>> pred = clf.predict_bbox(bgr_image, [x1, y1, x2, y2])
    >>> pred.label, pred.level, pred.confidence
    ('Macaca fuscata', 'species', 0.93)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

LEVELS = ("family", "genus", "species")
EXPORT_FORMAT = "primateface-taxon-v1"
CLIP_HUBS = {"bioclip2": "hf-hub:imageomics/bioclip-2", "bioclip": "hf-hub:imageomics/bioclip"}


@dataclass
class TaxonPrediction:
    """Hierarchical taxon prediction for one face.

    Attributes:
        label: Most specific confident name (species, genus, or family).
        level: Level of ``label`` (``"species"``, ``"genus"``, ``"family"``).
        confidence: Probability of ``label`` at its level.
        family: Top family and its probability.
        genus: Top genus and its probability.
        species: Top species and its probability (``None`` if the model has
            no species head entry).
        top_species: Top-k ``(species, prob)`` pairs.
        top_genus: Top-k ``(genus, prob)`` pairs.
    """

    label: str
    level: str
    confidence: float
    family: Tuple[str, float]
    genus: Tuple[str, float]
    species: Optional[Tuple[str, float]]
    top_species: List[Tuple[str, float]] = field(default_factory=list)
    top_genus: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        """Serialize to plain Python types."""
        return {
            "label": self.label,
            "level": self.level,
            "confidence": self.confidence,
            "family": list(self.family),
            "genus": list(self.genus),
            "species": list(self.species) if self.species else None,
            "top_species": [list(t) for t in self.top_species],
            "top_genus": [list(t) for t in self.top_genus],
        }


def square_crop(image: np.ndarray, bbox: Sequence[float], scale: float = 1.25, size: int = 224) -> np.ndarray:
    """Square face crop centred on ``bbox`` with context, zero padded at borders.

    Matches the crop used to build the training set.

    Args:
        image: HxWx3 image (any channel order; returned unchanged).
        bbox: ``[x1, y1, x2, y2]`` in pixels.
        scale: Side length as a multiple of ``max(w, h)``.
        size: Output side length.

    Returns:
        ``size x size x 3`` uint8 crop.
    """
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * scale
    half = side / 2
    sx1, sy1 = int(round(cx - half)), int(round(cy - half))
    sx2, sy2 = int(round(cx + half)), int(round(cy + half))
    h, w = image.shape[:2]
    pad = max(0, -sx1, -sy1, sx2 - w, sy2 - h)
    if pad:
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        sx1, sy1, sx2, sy2 = sx1 + pad, sy1 + pad, sx2 + pad, sy2 + pad
    crop = image[sy1:sy2, sx1:sx2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def _build_net(backbone: str, n_classes: Dict[str, int], cosine: bool):
    """Recreate the training-time network (without downloading weights)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if backbone in CLIP_HUBS:
                import open_clip

                clip = open_clip.create_model(CLIP_HUBS[backbone], pretrained=None)
                self.backbone = nn.Module()
                self.backbone.visual = clip.visual
                dim = clip.visual.output_dim
                self._clip = True
            else:
                import timm

                self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
                dim = self.backbone.num_features
                self._clip = False
            self.heads = nn.ModuleDict({k: nn.Linear(dim, n, bias=not cosine) for k, n in n_classes.items()})
            self.logit_scale = nn.Parameter(torch.zeros(())) if cosine else None
            self.cosine = cosine

        def forward(self, x):
            f = self.backbone.visual(x) if self._clip else self.backbone(x)
            if self.cosine:
                f = F.normalize(f, dim=-1)
                s = self.logit_scale.exp()
                return {k: s * F.linear(f, F.normalize(h.weight, dim=-1)) for k, h in self.heads.items()}
            return {k: h(f) for k, h in self.heads.items()}

    return _Net()


class TaxonClassifier:
    """Hierarchical primate taxon classifier for face crops.

    Args:
        checkpoint: Path to an exported ``primateface-taxon-v1`` checkpoint.
        device: Torch device string.
        species_threshold: Minimum species probability to report a species;
            otherwise the prediction rolls up to genus.
        genus_threshold: Minimum genus probability to report a genus;
            otherwise the prediction rolls up to family.
    """

    def __init__(
        self,
        checkpoint: Union[str, Path],
        device: str = "cpu",
        species_threshold: Optional[float] = None,
        genus_threshold: Optional[float] = None,
    ) -> None:
        import torch

        ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        if ckpt.get("format") != EXPORT_FORMAT:
            raise ValueError(f"{checkpoint} is not a {EXPORT_FORMAT} checkpoint")
        self.space = ckpt["label_space"]
        self.backbone = ckpt["backbone"]
        self.img_size = int(ckpt["img_size"])
        self.crop_scale = float(ckpt.get("crop_scale", 1.25))
        self.mean = np.asarray(ckpt["mean"], dtype=np.float32)
        self.std = np.asarray(ckpt["std"], dtype=np.float32)
        self.temperature = ckpt.get("temperature", {k: 1.0 for k in LEVELS})
        th = ckpt.get("thresholds", {})
        self.species_threshold = species_threshold if species_threshold is not None else th.get("species", 0.5)
        self.genus_threshold = genus_threshold if genus_threshold is not None else th.get("genus", 0.3)
        self.common_names: Dict[str, str] = self.space.get("common", {})
        n_classes = {k: len(self.space[k]) for k in LEVELS}
        self.net = _build_net(self.backbone, n_classes, bool(ckpt["cosine"]))
        self.net.load_state_dict(ckpt["state_dict"])
        self.device = device
        self.net.eval().to(device)

    @classmethod
    def from_pretrained(cls, name: str = "taxon_convnext_atto", device: str = "cpu", **kwargs) -> "TaxonClassifier":
        """Load a registered model by name or a local checkpoint path.

        Args:
            name: Registry name (see ``primateface models list``) or a path.
            device: Torch device.
            **kwargs: Passed to the constructor.

        Returns:
            Loaded classifier.
        """
        p = Path(name)
        if p.suffix in (".pt", ".pth") and p.exists():
            return cls(p, device=device, **kwargs)
        from ._model_manager import resolve_taxon_checkpoint

        return cls(resolve_taxon_checkpoint(name), device=device, **kwargs)

    def _preprocess(self, crops_rgb: Sequence[np.ndarray]):
        import torch

        arr = np.stack([cv2.resize(c, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA) for c in crops_rgb])
        arr = (arr.astype(np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(arr.transpose(0, 3, 1, 2)).to(self.device)

    def predict_logits(self, crops_rgb: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
        """Raw temperature-scaled logits per level for RGB square crops."""
        import torch

        with torch.no_grad():
            out = self.net(self._preprocess(crops_rgb))
        return {k: out[k].float().cpu().numpy() / float(self.temperature.get(k, 1.0)) for k in LEVELS}

    def predict_crops(self, crops_rgb: Sequence[np.ndarray], topk: int = 5) -> List[TaxonPrediction]:
        """Classify pre-cropped RGB face images.

        Args:
            crops_rgb: Square RGB crops (see :func:`square_crop`).
            topk: Number of alternatives to report.

        Returns:
            One :class:`TaxonPrediction` per crop.
        """
        if len(crops_rgb) == 0:
            return []
        logits = self.predict_logits(crops_rgb)
        probs = {k: _softmax(v) for k, v in logits.items()}
        preds = []
        for i in range(len(crops_rgb)):
            pf, pg, ps = probs["family"][i], probs["genus"][i], probs["species"][i]
            fam = (self.space["family"][int(pf.argmax())], float(pf.max()))
            gen = (self.space["genus"][int(pg.argmax())], float(pg.max()))
            sp = (self.space["species"][int(ps.argmax())], float(ps.max())) if len(ps) else None
            if sp and sp[1] >= self.species_threshold and self.space["species_to_genus"][sp[0]] == gen[0]:
                label, level, conf = sp[0], "species", sp[1]
            elif gen[1] >= self.genus_threshold:
                label, level, conf = gen[0], "genus", gen[1]
            else:
                label, level, conf = fam[0], "family", fam[1]
            ts = [(self.space["species"][j], float(ps[j])) for j in np.argsort(-ps)[:topk]] if len(ps) else []
            tg = [(self.space["genus"][j], float(pg[j])) for j in np.argsort(-pg)[:topk]]
            preds.append(TaxonPrediction(label, level, conf, fam, gen, sp, ts, tg))
        return preds

    def predict_bbox(self, image_bgr: np.ndarray, bbox: Sequence[float], topk: int = 5) -> TaxonPrediction:
        """Classify the face at ``bbox`` in a BGR image."""
        return self.predict_bboxes(image_bgr, [bbox], topk)[0]

    def predict_bboxes(self, image_bgr: np.ndarray, bboxes: Sequence[Sequence[float]], topk: int = 5) -> List[TaxonPrediction]:
        """Classify several faces in one BGR image (batched)."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        crops = [square_crop(rgb, b, self.crop_scale, self.img_size) for b in bboxes]
        return self.predict_crops(crops, topk)


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)
