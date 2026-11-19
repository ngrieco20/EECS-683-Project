Task 1(a): Selected AI Methodology

Method: We use a pretrained ResNet-50 to extract features from each IC image, then train a logistic regression classifier on those features. The code is in task1_baseline.py.

How it works:

Each image is resized to 224×224 and fed into ResNet-50. We remove its last layer, so instead of a class label it outputs 2048 numbers that describe the image (edges, textures, shapes). We do not retrain ResNet-50.
A logistic regression model is trained on these numbers using the labeled development images. It learns which patterns go with counterfeit parts.
For a new image, the same steps are applied and the model outputs 0 (authentic) or 1 (counterfeit).

Why it is suitable for counterfeit ICs:

Counterfeit chips often look different on the surface, such as re-marked or sanded packages, uneven texture, or odd printing. CNN features are good at picking up texture and pattern differences like these.
The pretrained network already knows how to describe images, so we only need a small labeled dataset and do not have to train a deep model from scratch.
It is fast and simple to run, and it can classify many images automatically, which makes it a good baseline to improve on later.

## Task 1(b): Baseline Implementation

We implemented the selected method in `task1_baseline.py`. The dataset contains 112 images, with 70 authentic and 42 counterfeit images. We used 88 images for training and 24 for validation. Front and back images of the same chip were grouped together based on their filenames to avoid data leakage.

The program extracts features using ResNet-50 and trains a logistic regression classifier. It automatically classifies multiple images as authentic (0) or counterfeit (1) and saves the results in a CSV file.

We also created `task1_predict.py`, which uses the trained model to classify new images without having to train it again.

## Task 1(c): Baseline Performance Evaluation

We used `task1_evaluate.py` to calculate the performance of our model on the 24 validation images.

**Results:**

| Metric | Score |
|---|---:|
| Accuracy | 87.50% |
| Precision | 100.00% |
| Recall | 70.00% |
| F1 Score | 82.35% |

**Confusion Matrix:**

| Actual / Predicted | Authentic | Counterfeit |
|---|---:|---:|
| Authentic | 14 | 0 |
| Counterfeit | 3 | 7 |

**Strengths:**

The model correctly classified 21 out of 24 images. All authentic images were classified correctly. Using a pretrained model also saved training time and allowed us to work with a small dataset.

**Limitations:**

The model incorrectly classified three counterfeit images as authentic. All three appear to be back-side images, which may contain fewer markings that help identify counterfeit chips. Another limitation is the small dataset, which may affect how well the model performs on new images.

## How to Run

Install dependencies:

`python3 -m pip install -r requirements.txt`

Train the model:

`python3 task1_baseline.py`

Evaluate the model:

`python3 task1_evaluate.py`

Classify new images:

`python3 task1_predict.py IMAGE_FOLDER`

Replace `IMAGE_FOLDER` with the actual path to the images. All results are saved in the `outputs/` folder.