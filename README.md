# P<sup>3</sup>ID: A Privacy-Preserving Person Identification Framework Towards Multi-Environments Based on Transfer Learning

This is the official implementation of the paper: <a href="https://ieeexplore.ieee.org/abstract/document/10679703" title="P<sup>3</sup>ID: A Privacy-Preserving Person Identification Framework Towards Multi-Environments Based on Transfer Learning">P<sup>3</sup>ID: A Privacy-Preserving Person Identification Framework Towards Multi-Environments Based on Transfer Learning</a>

## About P<sup>3</sup>ID

We propose a multi-environments person identification framework based on transfer learning using Impulse-Radio Ultra-Wideband (IR-UWB) radar dataset. A neural network is devised for mapping signals from distinct environments into a unified feature space and further align them, enabling the model to extract environment-insensitive features. 

## Getting Started

### Dependencies

```python
pytorch==1.12.1
tensorboard==2.10.1
torchvision==0.13.1
configargparse==1.4
numpy==1.21.5
scikit_learn==1.1.2
timm==0.5.4 
```

### Dataset

Using a real IR-UWB radar testbed, we build a dataset with 22,264 samples from three environments, varying in testing distance and occlusion condition. The directory structure is:

```
│path/to/dataset/
├──A_train/
│  ├── p1-A-0.5m-F-1
│  │   ├── 1.png
│  │   ├── 1_mw.png
│  │   ├── 1_pt.png
│  │   ├── ......
│  ├── ......
├──A_valid/
│  ├── p1-A-0.5m-F-1
│  │   ├── 2.png
│  │   ├── 3.png
│  │   ├── ......
│  ├── ......
├──A_test/
│  ├── p1-A-0.5m-F-1
│  │   ├── 4.png
│  │   ├── 5.png
│  │   ├── ......
│  ├── ......
```
<p1-A-0.5m-F-1> means **Person 1** conducted the **first** experiment in **Environment A** at a **distance of 0.5 meters** from the radar equipment, with **no obstructions**.

The anthropometric data of individuals is:

| Person ID | Height (cm) | Weight (kg) | Gender |
| :-------: | :---------: | :---------: | :----: |
|     1     |     182     |     75      |  Male  |
|     2     |     180     |     74      |  Male  |
|     3     |     175     |     65      |  Male  |
|     4     |     168     |     65      |  Male  |
|     5     |     170     |     65      |  Male  |
|     6     |     175     |     75      |  Male  |
|     7     |     186     |     92      |  Male  |
|     8     |     170     |     55      |  Male  |
|     9     |     162     |     48      | Female |
|    10     |     160     |     70      | Female |

It is worth noting that the initial version of our work provides a part of dataset , while the remaining dataset will be made available in subsequent versions.

### Usage

Frist, clone the repository locally:

```bash
git clone https://github.com/hxhebit/P3ID.git
```

Then, install Pytorch, tensorboard, and other dependencies:

```
pip3 install -r requirements.txt
```

Next, configure parameters at [file1](https://github.com/hxhebit/P3ID/blob/master/run/args/task1.yaml) or [file2](https://github.com/hxhebit/P3ID/blob/master/run/args/task2.yaml).

Finally, to train and test on a single node with 8 GPUs, run:

```bash
bash run/runs/task1.sh <FOLDER_NAME>
```

To customize your dataset, simply substitute the [dataset](https://github.com/hxhebit/P3ID/tree/master/dataset) with an identical directory structure.

---

***This project is continuously being updated.*** 
