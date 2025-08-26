# PrimateFace Dataset

## Overview
The PrimateFace dataset contains facial images and annotations for various primate species, including humans and non-human primates. This dataset supports research in comparative facial analysis, behavior recognition, and cross-species computer vision applications.

## Dataset Access

### Current Status
The full PrimateFace dataset will be released in **September 2025** on Hugging Face.

**Coming Soon:** [https://huggingface.co/datasets/fparodi/PrimateFace](https://huggingface.co/datasets/fparodi/PrimateFace)

### What's Available Now
- Pre-trained models: Available via `demos/download_models.py`
- Sample images: Included in `demos/` for testing
- Annotation format: COCO JSON with 68-point landmarks
- Documentation: Complete API and format specifications

### Dataset Preview
The complete dataset will include:
- **10,000+ annotated images** across 15+ primate species
- **68-point facial landmarks** for each face
- **Species labels** and metadata
- **Train/validation/test splits** for reproducible research
- **Diverse conditions**: Multiple poses, expressions, and environments

## Data Usage and Licensing

### Important Notice
**Users must comply with the original licensing terms and usage restrictions of each source dataset.** The PrimateFace dataset aggregates data from multiple sources, each with their own licensing requirements and usage policies.

### Usage Guidelines
1. **Research Use Only**: This dataset is intended for academic and research purposes
2. **Attribution Required**: Please cite both PrimateFace and the original data sources
3. **Privacy Considerations**: Some datasets may contain human subjects - follow appropriate ethical guidelines
4. **Commercial Use**: Check individual source dataset licenses for commercial usage restrictions

### Data Sources
The dataset includes contributions from various research institutions and publicly available datasets. Each subset maintains its original licensing terms:

- Human facial data: Subject to privacy and ethical use restrictions
- Non-human primate data: Various institutional data sharing agreements apply
- Public domain imagery: No restrictions beyond attribution

### Ethical Considerations
- Respect privacy of human subjects in the dataset
- Follow institutional guidelines for research involving primates
- Do not use for surveillance or identification without proper authorization
- Consider the ethical implications of cross-species facial analysis

## Dataset Structure

The dataset follows COCO format with extensions for facial landmarks:
- Images: JPEG/PNG format
- Annotations: JSON files with bounding boxes and 68-point facial landmarks
- Metadata: Species, data source, collection context

## Citation
If you use this dataset, please cite:

```bibtex
@article{parodi2025primateface,
title={PrimateFace: A Machine Learning Resource for Automated Face Analysis in Human and Non-human Primates},
author={Parodi, Felipe and Matelsky, Jordan and Lamacchia, Alessandro and Segado, Melanie and Jiang, Yaoguang and Regla-Vargas, Alejandra and Sofi, Liala and Kimock, Clare and Waller, Bridget M and Platt, Michael and others},
journal={bioRxiv},
pages={2025--08},
year={2025},
publisher={Cold Spring Harbor Laboratory}
}
```

## Contact
For questions about data usage or licensing:
- Email: primateface@gmail.com
- Felipe Parodi: fparodi@upenn.edu

## Disclaimer
The authors and maintainers of PrimateFace make no warranties about the suitability of this data for any particular purpose and accept no liability for any use of the dataset.