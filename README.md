# Deja-Vu Features

This repository contains the network architecture, losses and pretrained feature 
models from [Deja-Vu](https://www.researchgate.net/publication/340273695_Same_Features_Different_Day_Weakly_Supervised_Feature_Learning_for_Seasonal_Invariance).

## Introduction
"Like night and day" is a commonly used expression to imply that two things are completely different. 
Unfortunately, this tends to be the case for current visual feature representations of the same scene across varying seasons or times of day. 
The aim of this paper is to provide a dense feature representation that can be used to perform localization, sparse matching or image retrieval, regardless of the current seasonal or temporal appearance. 

Recently, there have been several proposed methodologies for deep learning dense feature representations. 
These methods make use of ground truth pixel-wise correspondences between pairs of images and focus on the spatial properties of the features. 
As such, they don't address temporal or seasonal variation. 
Furthermore, obtaining the required pixel-wise correspondence data to train in cross-seasonal environments is highly complex in most scenarios. 

We propose Deja-Vu, a weakly supervised approach to learning season invariant features that does not require pixel-wise ground truth data. 
The proposed system only requires coarse labels indicating if two images correspond to the same location or not. 
From these labels, the network is trained to produce "similar" dense feature maps for corresponding locations despite environmental changes.


## Prerequisites
- Python >= 3.6
- PyTorch >= 0.4
- Numpy
- Skimage

## Usage
A simple [script](main.py) is provided as an example of running the network and computing the similarity between feature maps.
```
# Create and load model
model = DejaVu.from_ckpt(ckpt_file).to(device)
# model.norm = False  # Removes L2 feature normalization
```

The loss should be created as such, with `is_similarity` indicating that we wish to compute the similarity, rather than the actual loss
```
loss = ContextualLoss(offset=1.0, bandwidth=0.5, is_similarity=True).to(device)
```

It is recommended to downsample the images/feature maps in order to speed up similarity computation.
The actual similarity values and their range will differ depending on each model.
Pretrained models are included in the [ckpts](ckpts) directory, which can be directly loaded with `DejaVu.from_ckpt`.


## Citation
Please cite the following paper if you find Deja-Vu useful in your research:
```
@inproceedings{spencer2020,
  title={Same Features, Different Day: Weakly Supervised Feature Learning for Seasonal Invariance},
  author={Spencer, Jaime  and Bowden, Richard and Hadfield, Simon},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

## Contact

You can contact me at [jaime.spencer@surrey.ac.uk](mailto:jaime.spencer@surrey.ac.uk)



