# LungLens

LungLens is a PyTorch convolutional neural network for classifying tuberculosis in chest X-ray images, served through a Streamlit web app where you can upload an image (or pick a preset example) and get an instant prediction.

> **Disclaimer:** this is a research/educational project, not a medical device. Its predictions must not be used for actual diagnosis.

## Table of Contents

- [Overview](#overview)
- [Data](#data)
- [Project Structure](#project-structure)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference (Web App)](#inference-web-app)
- [Docker Deployment](#docker-deployment)
- [Research Notes](#research-notes)
- [Results](#results)
- [License](#license)

## Overview

- **Model** — a CNN with four convolutional blocks (two convolutional layers per block) followed by four fully connected layers for binary classification.
- **Loss** — `BCEWithLogitsLoss` with an increased positive-class weight to counter class imbalance, plus a reduce-on-plateau learning-rate callback.
- **App** — a Streamlit interface that loads the trained weights and classifies uploaded or example images, with the weights path configurable via the `MODEL_WEIGHTS_PATH` environment variable.

## Data

The model is trained on the public **Tuberculosis (TB) Chest X-ray Database** (available on Kaggle): roughly 3,500 normal images and 700 TB images — a significant class imbalance that shapes both training and evaluation (see [Research Notes](#research-notes)).

## Project Structure

```
.
├── app.py                       # Streamlit inference app
├── tuberculosis_recognition/    # Model package (retains its original module name)
│   ├── cnn_model.py             # CNN architecture
│   ├── inference.py             # Image recognition wrapper
│   └── paths.py                 # Path configuration
├── scripts/
│   ├── data_preparation.py      # Dataset extraction and preprocessing
│   ├── train.py                 # Training loop
│   └── evaluation.py            # Test-set evaluation
├── source/                      # Data, example images, and model weights
├── Dockerfile / docker-compose.yml / Caddyfile
├── DEPLOY.md                    # Deployment guide
└── requirements.txt
```

## Training

1. Download the TB Chest X-ray dataset archive from Kaggle.
2. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

3. Run data preparation, pointing it at the downloaded archive:

   ```bash
   python -m scripts.data_preparation C:\Path\To\Archive\TB_database.zip
   ```

4. Train the model:

   ```bash
   python -m scripts.train --epochs 20 --lr 0.0003 --bs 32
   ```

The trained weights are saved to `./source/model/model_weights.pth`.

## Evaluation

Evaluate the trained model on the test split:

```bash
python -m scripts.evaluation --bs 32
```

Given the 5:1 class imbalance, plain accuracy is misleading (predicting "normal" for everything already scores ~83%). F1-score is the primary metric.

## Inference (Web App)

Place trained weights at `./source/model/model_weights.pth` (or set the `MODEL_WEIGHTS_PATH` environment variable to their location), then run:

```bash
streamlit run app.py
```

In the app you can upload your own chest X-ray or choose a preset image to test the model.

## Docker Deployment

The repository includes a `Dockerfile`, `docker-compose.yml`, and `Caddyfile` for containerized deployment behind a reverse proxy:

```bash
docker compose up --build
```

See [DEPLOY.md](DEPLOY.md) for the full deployment guide.

## Research Notes

### Data preparation

Normal images in the dataset are grayscale, while TB images vary in color spectrum — and about half of the TB images have only one channel. To prevent the network from using color as a shortcut feature, all images are converted to a single channel during preparation.

The class imbalance (3,500 normal vs. 700 TB) is addressed in two ways:

- **Training** — data augmentation and an increased loss weight for the TB class.
- **Evaluation** — F1-score instead of accuracy as the primary metric.

### Dataset caveats

The near-perfect scores likely overstate real-world performance, because the TB and normal images come from different sources (different medical organizations), leaving class-correlated artifacts the model can exploit:

- Different color spectrums between TB and normal images (mitigated during preparation)
- White rectangles present only in some TB images
- "L"/"R" position markers on a significant share of TB images

None of these artifacts are medically meaningful, but each is statistically predictive in this dataset. A model trained here may perform noticeably worse on images without these characteristics — a classic example of dataset bias in medical imaging.

## Results

| Metric | Test set |
|---|---|
| Accuracy | 99.29% |
| F1-score | 97.87% |

Test results match the validation results observed during training. Interpret them with the [dataset caveats](#dataset-caveats) above in mind.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
