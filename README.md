# [ICCV 2025] Learning to See in the Extremely Dark [[Paper]](https://arxiv.org/pdf/2506.21132)
<h4 align="center">Hai Jiang<sup>1</sup>, Binhao Guan<sup>2</sup>, Zhen Liu<sup>2</sup>, Xiaohong Liu<sup>3</sup>, Jian Yu<sup>4</sup>, Zheng Liu<sup>4</sup>, Songchen Han<sup>1</sup>, Shuaicheng Liu<sup>2</sup></center>
<h4 align="center">1.Sichuan University,</center></center>
<h4 align="center">2.University of Electronic Science and Technology of China,</center></center>
<h4 align="center">3.Shanghai Jiaotong University,</center></center>
<h4 align="center">4.National Innovation Center for UHD Video Technology</center></center>

## Dataset synthesis pipeline
![](./Figure/syn_pipe.png)

## Framework pipeline
![](./Figure/framework.png)

## Dependencies
```
pip install -r requirements.txt
````

## Download the raw training and evaluation datasets
### SIED dataset
Our SIED dataset is available at [[OneDrive]](https://1drv.ms/f/c/e379fe7c770e3033/Ejd2sO7WNMlGiAVSqFtu1KkBF8UL_RU9unCds1Mu8z8IPw?e=gyBKoy) and [[Baidu Yun (extracted code:4y4w)]](https://pan.baidu.com/s/13DpBAePEHpV0k4Mj96fgrw). Please see the txt files in ```data``` folder for the training set and evaluation set split. 

### SID dataset

## Pre-trained Models 
You can download our pre-trained model from [[OneDrive]]() and [[Baidu Yun (extracted code:)]]()

## How to train?
You need to modify ```datasets/dataset.py``` slightly for your environment, and then
```
python train.py  
```

## How to test?
```
python evaluate.py
```

## Visual comparison
![](./Figure/visual_canon.png)
![](./Figure/visual_sony.png)
## Citation
If you use this code or ideas from the paper for your research, please cite our paper:
```

```

## Acknowledgement
Part of the code is adapted from previous works: [WeatherDiff](https://github.com/IGITUGraz/WeatherDiffusion) and [MIMO-UNet](https://github.com/chosj95/MIMO-UNet). We thank all the authors for their contributions.

