# [ICCV 2025] Learning to See in the Extremely Dark [[Paper]]([https://arxiv.org/pdf/2407.08939](https://arxiv.org/pdf/2506.21132))
<h4 align="center">Hai Jiang<sup>1</sup>, Binhao Guan<sup>2</sup>, Zhen Liu<sup>2</sup>, Xiaohong Liu<sup>3</sup>, Jian Yu<sup>4</sup>, Zheng Liu<sup>4</sup>, Songchen Han<sup>1</sup>, Shuaicheng Liu<sup>2</sup></center>
<h4 align="center">1.Sichuan University,</center></center>
<h4 align="center">2.University of Electronic Science and Technology of China,</center></center>
<h4 align="center">3.Shanghai Jiaotong University,</center></center>
<h4 align="center">4.National Innovation Center for UHD Video Technology</center></center>

## Dataset synthesis pipeline
![](./Figure/syn_pipe.jpg)

## Framework pipeline
![](./Figure/framework.jpg)

## Dependencies
```
pip install -r requirements.txt
````

## Download the raw training and evaluation datasets
### SIED dataset
Coming soon!
### SID dataset

## Pre-trained Models 
You can download our pre-trained model from [[Google Drive]]() and [[Baidu Yun (extracted code:)]]()

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
![](./Figure/visual.jpg)

## Citation
If you use this code or ideas from the paper for your research, please cite our paper:
```

```

## Acknowledgement
Part of the code is adapted from previous works: [WeatherDiff](https://github.com/IGITUGraz/WeatherDiffusion) and [MIMO-UNet](https://github.com/chosj95/MIMO-UNet). We thank all the authors for their contributions.

