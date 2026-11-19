
from pathlib import Path
import re

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib

from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

# Configuration
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "outputs"

RANDOM_STATE = 42
BATCH_SIZE = 16

# Select computing device
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# Load images and corresponding labels
def load_dataset():
    paths = []
    labels = []

    folders = [
        ("Authentic", 0),
        ("Counterfeit", 1)
    ]

    extensions = {
        ".png", ".jpg", ".jpeg",
        ".bmp", ".tif", ".tiff"
    }

    for folder, label in folders:
        directory = DATASET_DIR / folder

        if not directory.is_dir():
            raise FileNotFoundError(directory)

        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                paths.append(path)
                labels.append(label)

    return np.array(paths), np.array(labels)


# Group front and back images of the same IC
def get_chip_id(path):
    return re.sub(
        r"(\d+)[FB](?=-|$)",
        r"\1",
        path.stem,
        flags=re.IGNORECASE
    )


# Load pretrained ResNet50
def load_feature_extractor():
    weights = ResNet50_Weights.DEFAULT

    model = resnet50(weights=weights)

    # Remove final classification layer
    model.fc = nn.Identity()

    # Freeze pretrained weights
    model.requires_grad_(False)
    model = model.to(device)
    model.eval()

    return model, weights.transforms()


# Extract 2048 features from each image
def extract_features(paths, model, preprocess):
    features = []

    with torch.inference_mode():
        for start in range(0, len(paths), BATCH_SIZE):
            batch = []

            for path in paths[start:start + BATCH_SIZE]:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    batch.append(preprocess(image))

            batch = torch.stack(batch).to(device)

            output = model(batch)
            features.append(output.cpu().numpy())

            print(
                f"Processed "
                f"{min(start + BATCH_SIZE, len(paths))}"
                f"/{len(paths)} images"
            )

    return np.concatenate(features)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Using device:", device)

    # Load development dataset
    paths, labels = load_dataset()

    print("Total images:", len(paths))
    print("Authentic:", np.sum(labels == 0))
    print("Counterfeit:", np.sum(labels == 1))

    if len(paths) == 0 or len(np.unique(labels)) != 2:
        raise ValueError("Both image classes are required.")

    # Prevent front/back images from crossing splits
    groups = np.array([
        get_chip_id(path) for path in paths
    ])

    # Approximately 80% training, 20% validation
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    train_idx, val_idx = next(
        splitter.split(paths, labels, groups)
    )

    assert not (
        set(groups[train_idx]) &
        set(groups[val_idx])
    ), "Data leakage detected"

    train_paths = paths[train_idx]
    val_paths = paths[val_idx]

    y_train = labels[train_idx]
    y_val = labels[val_idx]

    if len(np.unique(y_train)) != 2 or len(np.unique(y_val)) != 2:
        raise ValueError("Both splits must contain both classes.")

    print("\nTraining images:", len(train_paths))
    print("Validation images:", len(val_paths))

    # Initialize ResNet50
    model, preprocess = load_feature_extractor()

    # Extract training and validation features
    print("\nExtracting training features...")
    X_train = extract_features(
        train_paths, model, preprocess
    )

    print("\nExtracting validation features...")
    X_val = extract_features(
        val_paths, model, preprocess
    )

    # Train logistic regression
    print("\nTraining logistic regression...")

    classifier = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    classifier.fit(X_train, y_train)

    # Automatically classify validation images
    predictions = classifier.predict(X_val)

    # Export classification results
    results = pd.DataFrame({
        "image_identifier": [p.name for p in val_paths],
        "predicted_label": predictions
    })

    results.to_csv(
        OUTPUT_DIR / "classification_results.csv",
        index=False
    )

    # Save data for Task 1(c)
    np.savez(
        OUTPUT_DIR / "evaluation_data.npz",
        y_true=y_val,
        y_pred=predictions,
        image_paths=np.array(
            [str(p.relative_to(ROOT)) for p in val_paths]
        )
    )

    # Save split for Tasks 2 and 3
    split = pd.DataFrame({
        "image_identifier": [
            str(p.relative_to(ROOT)) for p in paths
        ],
        "true_label": labels,
        "chip_id": groups,
        "split": [
            "validation" if i in set(val_idx)
            else "train"
            for i in range(len(paths))
        ]
    })

    split.to_csv(
        OUTPUT_DIR / "dataset_split.csv",
        index=False
    )

    # Save trained classifier
    joblib.dump(
        classifier,
        OUTPUT_DIR / "baseline_classifier.joblib"
    )

    print("\nClassification complete!")
    print(results.to_string(index=False))
    print("\nResults saved to outputs/")


if __name__ == "__main__":
    main()