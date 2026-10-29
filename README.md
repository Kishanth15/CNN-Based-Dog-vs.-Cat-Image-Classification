# CNN-Based Dog vs. Cat Image Classification

This project classifies cat and dog images with a convolutional neural network
based on **MobileNetV2** and ImageNet transfer learning. Training uses random
flips, rotations, zoom, and contrast augmentation. The model first trains a
classification head, then fine-tunes the final backbone layers.

The script evaluates a separate held-out test split and reports accuracy,
macro precision, macro recall, macro F1, ROC-AUC, and per-class metrics. It
also generates a chart with training/validation accuracy and loss, a confusion
matrix, and an ROC curve. It saves the trained Keras model alongside the chart.

## Dataset

Download **`train.zip`** from the Kaggle [Dogs vs. Cats competition data page](https://www.kaggle.com/competitions/dogs-vs-cats/data)
(Kaggle account and acceptance of the competition rules are required). The
archive contains labeled images named `cat.*` and `dog.*`. Do not use Kaggle's
unlabeled `test1.zip` for this supervised training workflow.

Arrange labeled images into these folders, with the class names spelled exactly
as shown:

```text
data/
  train/
    cat/  dog/
  validation/
    cat/  dog/
  test/
    cat/  dog/
```

Use distinct images in each split. An 80/10/10 stratified split is a reasonable
starting point. The script expects the folders to be prepared; it does not
download Kaggle data automatically.

## Run locally

```bash
python -m pip install -r requirements.txt
python cnn_dog_cat_classifier.py --data-dir ./data
```

Optional arguments:

```bash
python cnn_dog_cat_classifier.py --data-dir ./data --epochs 12 --batch-size 32 --output-dir ./results
```

The first run downloads pretrained MobileNetV2 ImageNet weights. Outputs are
`dog_cat_cnn_results.png` and `dog_cat_cnn.keras` in the selected output folder.
The script checks whether test accuracy reaches 85%, but actual performance
depends on the images and labels; no result is claimed until it is run on a
prepared dataset.

## Google Colab

Open [`dog_cat_colab.ipynb`](dog_cat_colab.ipynb) in Google Colab for an
end-to-end workflow. It uploads Kaggle's `train.zip`, splits it into train,
validation, and test folders, runs the training script, and downloads the
generated chart and model. The notebook uses an 80/10/10 split with a fixed
random seed.
