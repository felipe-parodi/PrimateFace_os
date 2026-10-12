"""Hierarchical taxon classifier: one backbone, family/genus/species heads."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_BACKBONES = {
    "bioclip2": "hf-hub:imageomics/bioclip-2",
    "bioclip": "hf-hub:imageomics/bioclip",
}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
LEVELS = ("family", "genus", "species")


class _ClipVisual(nn.Module):
    """Wrap an open_clip visual tower so it returns projected embeddings."""

    def __init__(self, visual: nn.Module) -> None:
        super().__init__()
        self.visual = visual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)


def build_backbone(
    name: str, pretrained: bool = True, img_size: Optional[int] = None
) -> Tuple[nn.Module, int, Dict[str, object]]:
    """Create a feature extractor.

    Args:
        name: timm model name, or a key of ``CLIP_BACKBONES``.
        pretrained: Load pretrained weights.
        img_size: Input size for fixed-size ViTs (positional embeddings are resized).

    Returns:
        (module mapping images to features, feature dim, data config with
        ``input_size``, ``mean``, ``std``).
    """
    if name in CLIP_BACKBONES:
        import open_clip

        clip, _, _ = open_clip.create_model_and_transforms(CLIP_BACKBONES[name] if pretrained else None)
        visual = clip.visual
        dim = visual.output_dim
        cfg = {"input_size": (3, 224, 224), "mean": CLIP_MEAN, "std": CLIP_STD}
        return _ClipVisual(visual), dim, cfg
    import timm

    m = create_timm(name, pretrained, img_size)
    dc = timm.data.resolve_data_config({}, model=m)
    cfg = {"input_size": dc["input_size"], "mean": dc["mean"], "std": dc["std"]}
    return m, feature_dim(m), cfg


def create_timm(name: str, pretrained: bool, img_size: Optional[int] = None) -> nn.Module:
    """Create a headless timm model, passing ``img_size`` when the model accepts it."""
    import timm

    if img_size is not None:
        try:
            return timm.create_model(name, pretrained=pretrained, num_classes=0, img_size=img_size)
        except TypeError:
            pass
    return timm.create_model(name, pretrained=pretrained, num_classes=0)


def feature_dim(m: nn.Module) -> int:
    """Pooled output width of a headless timm model (``num_features`` can differ,
    e.g. MobileNetV4's conv head)."""
    hidden = getattr(m, "head_hidden_size", None)
    return int(hidden) if hidden else int(m.num_features)


class TaxonClassifier(nn.Module):
    """Backbone plus linear heads for each taxonomic level.

    Args:
        backbone: Name passed to :func:`build_backbone`.
        n_classes: Mapping level -> number of classes.
        pretrained: Load pretrained backbone weights.
        cosine_heads: Use L2-normalised features and weights with a learnable scale
            (matches CLIP geometry; lets heads be initialised from text embeddings).
        img_size: Training/inference resolution (needed by fixed-size ViTs).
    """

    def __init__(
        self,
        backbone: str,
        n_classes: Dict[str, int],
        pretrained: bool = True,
        cosine_heads: Optional[bool] = None,
        img_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.backbone, self.feat_dim, self.data_config = build_backbone(backbone, pretrained, img_size)
        self.cosine = backbone in CLIP_BACKBONES if cosine_heads is None else cosine_heads
        self.heads = nn.ModuleDict(
            {lvl: nn.Linear(self.feat_dim, n, bias=not self.cosine) for lvl, n in n_classes.items()}
        )
        self.logit_scale = nn.Parameter(torch.tensor(30.0).log()) if self.cosine else None
        self.drop = nn.Dropout(0.1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone features (normalised when using cosine heads)."""
        f = self.backbone(x)
        return F.normalize(f, dim=-1) if self.cosine else f

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return logits per level."""
        f = self.drop(self.forward_features(x))
        if self.cosine:
            s = self.logit_scale.exp()
            return {lvl: s * F.linear(f, F.normalize(h.weight, dim=-1)) for lvl, h in self.heads.items()}
        return {lvl: h(f) for lvl, h in self.heads.items()}

    @torch.no_grad()
    def init_heads_from_text(self, class_names: Dict[str, List[str]], prompts: Dict[str, List[str]]) -> None:
        """Initialise cosine heads from BioCLIP text embeddings (zero-shot init).

        Args:
            class_names: Level -> class name list (same order as the head).
            prompts: Level -> list of full text prompts aligned with ``class_names``.
        """
        import open_clip

        hub = CLIP_BACKBONES[self.backbone_name]
        clip, _, _ = open_clip.create_model_and_transforms(hub)
        tok = open_clip.get_tokenizer(hub)
        clip = clip.eval().to(next(self.parameters()).device)
        for lvl, texts in prompts.items():
            embs = []
            for i in range(0, len(texts), 256):
                t = tok(texts[i : i + 256]).to(next(self.parameters()).device)
                embs.append(F.normalize(clip.encode_text(t).float(), dim=-1))
            self.heads[lvl].weight.copy_(torch.cat(embs))
        del clip


def taxon_prompts(space: Dict[str, object], common: Optional[Dict[str, str]] = None) -> Dict[str, List[str]]:
    """BioCLIP-style taxonomic prompts for each class.

    Args:
        space: Label space with ``family``, ``genus``, ``species`` lists and parent maps.
        common: Optional species -> common name.

    Returns:
        Level -> prompt list aligned with the class lists.
    """
    common = common or {}
    g2f = space["genus_to_family"]
    s2g = space["species_to_genus"]
    base = "a photo of Animalia Chordata Mammalia Primates"
    out = {
        "family": [f"{base} {f}." for f in space["family"]],
        "genus": [f"{base} {g2f[g]} {g}." for g in space["genus"]],
    }
    sp = []
    for s in space["species"]:
        g = s2g[s]
        epithet = s.split(" ", 1)[1] if " " in s else s
        txt = f"{base} {g2f[g]} {g} {epithet}"
        if s in common:
            txt += f" with common name {common[s]}"
        sp.append(txt + ".")
    out["species"] = sp
    return out
