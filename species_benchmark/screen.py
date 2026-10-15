"""Zero-shot content screen for face crops (SigLIP 2, Apache-2.0).

Scores each crop as human / non-human primate / drawing / other so that human faces
and non-photographic or non-primate crops can be separated from the primate label pool.

Example:
    python -m species_benchmark.screen --crops-from C:/pfspecies/data/phash.parquet \
        C:/pfspecies/data/yt_pool_manifest.parquet --out C:/pfspecies/data/screen.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

MODEL = ("ViT-B-16-SigLIP2", "webli")
PROMPTS: Dict[str, List[str]] = {
    "human": ["a photo of a human face", "a photo of a person's face", "a close-up photo of a man or woman"],
    "primate": [
        "a photo of a monkey's face",
        "a photo of an ape's face",
        "a photo of a lemur's face",
        "a close-up photo of a wild primate",
    ],
    "drawing": ["a drawing or illustration of a monkey", "a cartoon of an animal", "a painting of an animal"],
    "other": [
        "a photo of a dog, cat, or other non-primate animal",
        "a blurry photo of leaves and branches with no face",
        "a toy or stuffed animal",
        "a photo of a statue",
    ],
}


class _Crops(Dataset):
    def __init__(self, paths: List[str], tf) -> None:
        self.paths, self.tf = paths, tf

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def screen(paths: List[str], bs: int = 256, workers: int = 10) -> pd.DataFrame:
    """Category probabilities for each crop.

    Args:
        paths: Crop image paths.
        bs: Batch size.
        workers: DataLoader workers.

    Returns:
        DataFrame with ``p_human``, ``p_primate``, ``p_drawing``, ``p_other``.
    """
    import open_clip

    model, _, pre = open_clip.create_model_and_transforms(MODEL[0], pretrained=MODEL[1])
    tok = open_clip.get_tokenizer(MODEL[0])
    model = model.cuda().eval()
    texts, owners = [], []
    for cat, ps in PROMPTS.items():
        texts += ps
        owners += [cat] * len(ps)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        t = model.encode_text(tok(texts).cuda()).float()
    t = torch.nn.functional.normalize(t, dim=-1)
    scale = model.logit_scale.exp().float()
    cats = list(PROMPTS)
    owner_idx = torch.tensor([cats.index(o) for o in owners]).cuda()
    out = []
    for x in DataLoader(_Crops(paths, pre), batch_size=bs, num_workers=workers, pin_memory=True):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = model.encode_image(x.cuda(non_blocking=True)).float()
        logits = scale * torch.nn.functional.normalize(f, dim=-1) @ t.T
        p = torch.softmax(logits, -1)
        agg = torch.zeros(len(p), len(cats), device=p.device).index_add_(1, owner_idx, p)
        out.append(agg.cpu().numpy())
    arr = np.concatenate(out)
    return pd.DataFrame({f"p_{c}": arr[:, i] for i, c in enumerate(cats)})


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crops-from", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    frames = [pd.read_parquet(p, columns=["crop_id", "crop_path"]) for p in args.crops_from]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("crop_id")
    df = df[df["crop_path"].map(lambda q: Path(q).exists())].reset_index(drop=True)
    res = pd.concat([df, screen(df["crop_path"].tolist())], axis=1)
    res["category"] = res[[c for c in res.columns if c.startswith("p_")]].idxmax(axis=1).str[2:]
    res.to_parquet(args.out, index=False)
    print(res["category"].value_counts().to_dict())


if __name__ == "__main__":
    main()
