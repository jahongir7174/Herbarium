[Herbarium 2022](https://www.kaggle.com/competitions/herbarium-2022-fgvc9) Kaggle contest source code

### Install

* `pip install timm`

### Train

* `bash ./main.sh $ --train` for training, `$` is number of GPUs

### Results

|    Model     | LR Schedule | Epochs | Top1 |
|:------------:|:-----------:|-------:|-----:|
| ECAResNet-T  |    Step     |    300 | 81.8 |

### Reference

* https://github.com/rwightman/pytorch-image-models