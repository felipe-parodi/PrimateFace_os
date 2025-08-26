# Contributing to PrimateFace

Thank you for your interest in contributing to PrimateFace! We welcome contributions from the community to help improve and expand this resource for primate facial analysis.

## Ways to Contribute

### 1. Report Issues
- Use the [GitHub Issues](https://github.com/PrimateFace/primateface_oss/issues) page
- Check if the issue already exists before creating a new one
- Provide detailed information including:
  - Your environment (OS, Python version, package versions)
  - Steps to reproduce the issue
  - Expected vs actual behavior
  - Error messages and stack traces

### 2. Contribute Code

#### Getting Started
1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/primateface_oss.git`
3. Create a new branch: `git checkout -b feature-name`
4. Set up your development environment:
   ```bash
   pip install -e .[dev]
   ```

#### Code Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Write unit tests for new functionality
- Keep commits atomic and write clear commit messages

#### Testing
Before submitting a PR, ensure:
```bash
# Run tests
pytest tests/

# Check code style
black --check .
flake8 .

# Type checking
mypy .
```

### 3. Contribute Data or Models

If you have:
- **New primate facial datasets**: Contact us about data sharing agreements
- **Trained models**: Submit via pull request with documentation
- **Annotations**: Follow our COCO format guidelines

Please ensure:
- You have appropriate permissions to share the data
- Data is properly anonymized if needed
- Include documentation about data collection and annotation procedures

### 4. Improve Documentation

- Fix typos or clarify existing documentation
- Add examples and tutorials
- Translate documentation to other languages
- Create video tutorials or blog posts

## Pull Request Process

1. **Before submitting:**
   - Ensure your code follows our style guidelines
   - Add/update tests as needed
   - Update documentation if you changed functionality
   - Add your changes to CHANGELOG.md

2. **PR Description should include:**
   - What changes were made and why
   - Link to any relevant issues
   - Screenshots/videos for UI changes
   - Performance impact if applicable

3. **Review process:**
   - PRs require at least one maintainer approval
   - Address all review comments
   - Keep your branch up to date with main

## Development Setup

### Environment Setup
```bash
# Clone the repo
git clone https://github.com/PrimateFace/primateface_oss.git
cd primateface_oss

# Create conda environment
conda create -n primateface-dev python=3.10
conda activate primateface-dev

# Install in development mode
pip install -e .[dev]
```

### Running Tests
```bash
# All tests
pytest

# Specific module
pytest tests/test_detection.py

# With coverage
pytest --cov=primateface tests/
```

## Code of Conduct

### Our Standards
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards other community members

### Unacceptable Behavior
- Harassment, discrimination, or offensive comments
- Personal or political attacks
- Public or private harassment
- Publishing others' private information
- Other unprofessional conduct

## Questions?

- Email: primateface@gmail.com
- GitHub Discussions: [Link](https://github.com/PrimateFace/primateface_oss/discussions)
- Issues: [Link](https://github.com/PrimateFace/primateface_oss/issues)

## License

By contributing to PrimateFace, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

Thank you to all our contributors! Your efforts help advance primate behavioral research and computer vision.