# P<sup>3</sup>ID: A Privacy-Preserving Person Identification Framework Towards Multi-Environments Based on Transfer Learning

## About P<sup>3</sup>ID

We propose a transfer learning framework for achieving multi-environments person identification. A neural network is devised for  mapping signals from distinct environments into a unified feature space and further align them, enabling the model to extract environment-insensitive features. 

### Dataset

Using a real Impulse Radio Ultra-WideBand (IR-UWB) radar testbed, we build a dataset with 22,264 samples from three environments, varying in testing distance and occlusion condition. The directory structure is:

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
It is worth noting that the initial version of our work provides a part of dataset , while the remaining dataset will be made available in subsequent versions.

---

***This project is continuously being updated.*** 
