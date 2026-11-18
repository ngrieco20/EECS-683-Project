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