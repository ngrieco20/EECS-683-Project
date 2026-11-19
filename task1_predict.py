
from pathlib import Path
import argparse

import joblib
import pandas as pd

from task1_baseline import (
    load_feature_extractor,
    extract_features,
    OUTPUT_DIR
)


def main():
    parser = argparse.ArgumentParser(
        description="Classify unseen IC images"
    )

    parser.add_argument(
        "input_dir",
        help="Folder containing IC images"
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir)

    extensions = {
        ".png", ".jpg", ".jpeg",
        ".bmp", ".tif", ".tiff"
    }

    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)

    paths = sorted([
        path for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
    ])

    if not paths:
        raise ValueError("No images found.")

    print("Total images:", len(paths))

    # Load trained logistic regression model
    classifier = joblib.load(
        OUTPUT_DIR / "baseline_classifier.joblib"
    )

    # Load pretrained ResNet50
    model, preprocess = load_feature_extractor()

    # Extract image features
    features = extract_features(
        paths, model, preprocess
    )

    # Generate predictions
    predictions = classifier.predict(features)

    results = pd.DataFrame({
        "image_identifier": [
            str(p.relative_to(input_dir))
            for p in paths
        ],
        "predicted_label": predictions
    })

    output_file = OUTPUT_DIR / "unseen_predictions.csv"

    results.to_csv(output_file, index=False)

    print("\nClassification Results:")
    print(results.to_string(index=False))

    print("\nPredicted authentic:", sum(predictions == 0))
    print("Predicted counterfeit:", sum(predictions == 1))

    print("\nResults saved to:", output_file)


if __name__ == "__main__":
    main()