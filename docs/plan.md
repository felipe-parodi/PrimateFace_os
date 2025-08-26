# PrimateFace Documentation Plan

## Overview

This document outlines the comprehensive documentation structure for the PrimateFace project, ensuring clear navigation between quick-start guides, API references, and workflow tutorials.

## Documentation Philosophy

- **Quick Start First**: Each module has a clean README focused on getting started
- **Technical Depth Available**: Detailed documentation in module-specific `*_docs.md` files
- **Unified API Reference**: Centralized in `docs/api/` with links to technical docs
- **Practical Workflows**: Step-by-step guides for common tasks in `docs/guides/`

## Directory Structure

```
PrimateFace/
├── README.md                     # Project overview & quick start
├── demos/
│   ├── README.md                # Clean quick start (✅ Done)
│   └── demo_docs.md            # Technical details (✅ Done)
├── dinov2/
│   ├── README.md                # Clean quick start (✅ Done)
│   └── dinov2_docs.md          # Technical details (✅ Done)
├── landmark-converter/
│   ├── README.md                # Clean quick start (✅ Done)
│   └── converter_docs.md       # Technical details (✅ Done)
├── gui/
│   ├── README.md                # Clean quick start (✅ Done)
│   └── gui_docs.md             # Technical details (✅ Done)
├── evals/
│   ├── README.md                # Already comprehensive
│   └── eval_docs.md            # Technical details (🔄 To do)
└── docs/
    ├── index.md                 # Main documentation homepage
    ├── installation.md          # Installation guide
    ├── api/                     # API Reference
    │   ├── index.md            # API overview (✅ Done)
    │   ├── demos.md            # Demos API (✅ Done)
    │   ├── dinov2.md           # DINOv2 API (✅ Done)
    │   ├── converter.md        # Landmark Converter API (✅ Done)
    │   ├── gui.md              # GUI API (✅ Done)
    │   └── evaluation.md       # Evaluation API (🔄 To do)
    ├── guides/                  # Workflow Guides
    │   ├── index.md            # Guides overview (✅ Done)
    │   ├── inference.md        # Using pretrained models
    │   ├── subset-selection.md # DINOv2-guided selection
    │   ├── landmark-training.md # Training landmark converter
    │   ├── pseudo-labeling.md  # GUI pseudo-labeling workflow
    │   └── framework-training.md # Training in different frameworks
    ├── tutorials/               # Interactive tutorials (existing)
    ├── frameworks/              # Framework integration (existing)
    └── tools/                   # Tool documentation (existing)
```

## Module Documentation Status

### ✅ Completed
1. **demos/**
   - README.md: Reduced from 278 to 162 lines (42% reduction)
   - demo_docs.md: Created with 346 lines of technical detail
   - docs/api/demos.md: Created API reference

2. **dinov2/**
   - README.md: Reduced from 260 to 198 lines (24% reduction)
   - dinov2_docs.md: Created with 423 lines of technical detail
   - docs/api/dinov2.md: Created API reference

3. **landmark-converter/**
   - README.md: Reduced from 213 to 160 lines (25% reduction)
   - converter_docs.md: Created with 380+ lines of technical detail
   - docs/api/converter.md: Created API reference

4. **gui/**
   - README.md: Updated to 193 lines with clean structure
   - gui_docs.md: Created with 400+ lines of technical detail
   - docs/api/gui.md: Created API reference

### 🔄 To Do
1. **evals/**
   - Create eval_docs.md for deep technical content
   - Keep comprehensive README (already well-structured)
   - Create docs/api/evaluation.md

2. **Workflow Guides** (content creation)
   - guides/inference.md
   - guides/subset-selection.md
   - guides/landmark-training.md
   - guides/pseudo-labeling.md
   - guides/framework-training.md

## Workflow Guides Structure

### 1. Inference with Pretrained Models (`guides/inference.md`)
- Quick start with demos
- Model zoo overview
- Performance benchmarks
- Troubleshooting

### 2. DINOv2-Guided Image Selection (`guides/subset-selection.md`)
- Theory and motivation
- Step-by-step workflow
- Comparison with random sampling
- Integration with training pipelines

### 3. Training Landmark Converters (`guides/landmark-training.md`)
- Data preparation
- Model selection guide
- Training best practices
- Evaluation and deployment

### 4. Pseudo-Labeling Workflow (`guides/pseudo-labeling.md`)
- GUI setup and configuration
- Batch processing strategies
- Quality control
- Export formats

### 5. Framework-Specific Training (`guides/framework-training.md`)
- MMPose/MMDetection setup
- DeepLabCut integration
- SLEAP workflows
- YOLO training
- Performance comparison

## Documentation Standards

### README.md Guidelines
- **Length**: 100-200 lines maximum
- **Structure**:
  1. Brief description (2-3 lines)
  2. Visual (if applicable)
  3. Key features (bullet points)
  4. Quick Start (3-4 essential steps)
  5. Python API (brief example)
  6. Common options (table format)
  7. Documentation links
  8. Troubleshooting (3-4 common issues)
  9. Next steps

### Technical Documentation (*_docs.md)
- **Comprehensive**: Full API reference
- **Examples**: Multiple usage patterns
- **Advanced**: Performance optimization, custom implementations
- **Testing**: Unit test examples
- **Architecture**: Implementation details

### API Documentation (docs/api/*.md)
- **Quick Reference**: Import statements
- **Main Classes**: Core functionality
- **Common Patterns**: 2-3 examples
- **Integration**: With other modules
- **Links**: To technical docs for details

## Implementation Priority

1. **High Priority** (Core functionality)
   - landmark-converter documentation
   - guides/index.md structure
   - guides/inference.md

2. **Medium Priority** (User workflows)
   - gui documentation
   - guides/pseudo-labeling.md
   - guides/landmark-training.md

3. **Lower Priority** (Advanced usage)
   - evals documentation refinement
   - guides/framework-training.md
   - guides/subset-selection.md

## Quality Checklist

- [ ] Each module has clean README + technical docs
- [ ] All READMEs follow consistent structure
- [ ] API docs link to technical documentation
- [ ] Workflow guides cover common use cases
- [ ] Cross-references between related modules
- [ ] Code examples are tested and working
- [ ] Tables used for options/parameters
- [ ] Visual assets included where helpful

## Next Steps

1. Complete landmark-converter documentation
2. Create guides/index.md with navigation
3. Update main docs/index.md with new structure
4. Create first workflow guide (inference.md)
5. Review and refine based on user feedback

## Future Enhancements

### 📦 Model Zoo Page
Create a dedicated model zoo page (`docs/models.md` enhancement or separate `docs/model-zoo.md`) with:
- Table of all pretrained models with download links
- Performance metrics for each model
- Hardware requirements
- Example usage for each model
- Version compatibility matrix

### 📹 Video Tutorials
Plan for embedded video tutorials in documentation:
- Quick start videos for each module
- Workflow walkthroughs
- GUI tool demonstrations
- Framework-specific training tutorials
- Can embed directly in MkDocs pages using HTML/iframe

### 📊 API Versioning
Implement API versioning documentation:
- Changelog page (`docs/changelog.md`)
- Migration guides between versions
- Deprecation warnings
- API stability guarantees
- Version compatibility matrix

### 🚀 Additional Ideas

1. **Interactive Code Playground**: Embed runnable code examples using services like Binder or Google Colab
2. **Performance Benchmarks Page**: Comprehensive benchmarks across different hardware configurations
3. **FAQ Section**: Common questions and troubleshooting organized by topic
4. **Community Showcase**: Gallery of projects using PrimateFace
5. **Developer Guide**: Contributing guidelines, architecture overview, plugin development

## Success Metrics

- User can go from installation to inference in <5 minutes
- Each workflow guide is self-contained and completable
- Technical users can find deep implementation details easily
- Documentation reduces GitHub issues by providing clear answers
- Consistent style makes navigation intuitive