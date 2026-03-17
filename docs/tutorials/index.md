# Tutorials

PrimateFace provides practical tutorials demonstrating various applications of primate facial analysis across different species and use cases.

## Available Tutorials

### 🐵 Application Notebooks

Our Jupyter notebooks demonstrate real-world applications:

#### 1. [Lemur Face Visibility Time-Stamping](lemur_timestamping.md)
Track when lemur faces are visible in video footage for behavioral analysis.

#### 2. [Macaque Face Recognition](macaque_recognition.md)  
Identify individual macaques using facial landmarks and features.

#### 3. [Howler Vocal-Motor Coupling](howler_vocal.md)
Analyze facial movements during vocalizations in howler monkeys.

#### 4. [Gaze Following Analysis](gaze_following.md)
Track primate gaze direction for social behavior studies.

#### 5. [Data-Driven Discovery of Facial Actions](facial_actions.md)
Use unsupervised methods to discover facial action patterns.

#### 6. [Cross-Subject Neural Decoding](neural_decoding.md)
Decode facial actions from neural recordings across subjects.

## Running the Notebooks

### Local Setup

```bash
# Clone the repository
git clone https://github.com/KordingLab/PrimateFace.git
cd PrimateFace

# Install dependencies
pip install -e .

# Launch Jupyter
jupyter notebook demos/notebooks/
```

### Notebooks

Run locally with auto-downloading models from [HuggingFace](https://huggingface.co/fparodi/primateface-models):

| Tutorial | Notebook |
|----------|----------|
| Lemur Video Timestamping | [`lemur_video_timestamping.ipynb`](../notebooks/lemur_video_timestamping.ipynb) |
| Macaque Face Recognition | [`macaque_face_recognition.ipynb`](../notebooks/macaque_face_recognition.ipynb) |
| Howler Vocal-Motor Coupling | [`howler_vocal_motor_coupling.ipynb`](../notebooks/howler_vocal_motor_coupling.ipynb) |
| Macaque Gaze Following | [`macaque_gaze_following.ipynb`](../notebooks/macaque_gaze_following.ipynb) |
| Landmark Demographics | [`landmark_demographics.ipynb`](../notebooks/landmark_demographics.ipynb) |

## Framework-Specific Examples

- [Using with DeepLabCut](../frameworks/deeplabcut.md)
- [Using with SLEAP](../frameworks/sleap.md)
- [Using with MMPose](../frameworks/mmpose.md)
- [Using with Ultralytics](../frameworks/ultralytics.md)

## Need Help?

- Check our [FAQ](../faq.md)
- Report issues on [GitHub](https://github.com/KordingLab/PrimateFace/issues)
- Contact us at [primateface@gmail.com](mailto:primateface@gmail.com)