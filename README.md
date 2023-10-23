# DAGI

> This repo holds code for [Imputing Brain Measurements Across Data Sets via Graph Neural Networks](https://link.springer.com/chapter/10.1007/978-3-031-46005-0_15#citeas). (MICCAI PRIME 2023 Accepted)

## Prepare data and train your model
```
python main.py
```
> We are using NCANDA dataset, with the Freesurfer score for each of the 34 bilateral cortical regions. Each of the regions consists of 5 regional measurements: average thickness, surface area, gray matter volume, mean curvature, and Gaussian curvature. We use the first three measurements to impute the rest curvature scores.

> Please replace the file_path in main.py with your own file path.


### Citation
If you find this paper, code useful for your research, please cite our paper:
```
@inproceedings{wang2023imputing,
  title={Imputing Brain Measurements Across Data Sets via Graph Neural Networks},
  author={Wang, Yixin and Peng, Wei and Tapert, Susan F and Zhao, Qingyu and Pohl, Kilian M},
  booktitle={International Workshop on PRedictive Intelligence In MEdicine},
  pages={172--183},
  year={2023},
  organization={Springer}
}

