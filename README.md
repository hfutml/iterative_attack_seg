# Iterative Adversarial Attack on Image-guided Story Ending Generation
This is the repository for the Iterative-attack on multimodal text generation tasks.

## Dependencies
- python 3.8.16
- torch 1.12.1
- transformers 4.12.5
- sentence-transformers 2.2.2
- sacrebleu 2.3.1
- pycocoevalcap 1.2
- stanfordcorenlp 3.9.1.1

## Usage
- Datasets
    - VIST-E: download the text annotation (SIS-with-labels.tar.gz) and all images from https://visionandlanguage.net/VIST/dataset.html. 
    - LSMDC-E: download LSMDC 2021 version from https://sites.google.com/site/describingmovies/home.

    - Multi30k: download the dataset from https://github.com/multi30k/dataset.

     Note: Please refer to the above papers for specific processing details of the datasets.

- Victim models
    - Image-guided story ending generation 
        - MGCL: https://github.com/VILAN-Lab/MGCL
        - MMT: https://github.com/LivXue/MMT
    - Multimodal machine translation
        - https://github.com/QAQ-v/MMT

- Run `iterative_attack_run.py` to attack the victim models.

## Citation
```
@ARTICLE{10366855,
  author={Wang, Youze and Hu, Wenbo and Hong, Richang},
  journal={IEEE Transactions on Multimedia}, 
  title={Iterative Adversarial Attack on Image-guided Story Ending Generation}, 
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TMM.2023.3345167}}

```

