# Self-Supervised Anomaly Segmentation

### Intorduction:

This is a PyToch implementation of [A Novel Self-supervised Learning Task Designed for Anomaly Segmentation]().

Our pseudo mask generator and anomaly segmentation framework:

 <img src="./demo/cocomask.png" width="300" />     <img src="./demo/DBAD.png" width="480" />

### Contributions:

- [x] we propose a novel self-supervised learning pretext task, which is different from generation-based methods or commonly contrastive leanring, it generat pseudo mask from other labeled dataset such as CoCo, and every suitable for pixelwise downstream tasks.
- [x] we present an end-to-end anomaly segmenation framework, it has both high speed and accuracy, and with no post-processing.
- [ ] our method achieve SOTA in three anomaly detection/segmentation datasets. (#ToDo)

### Anomaly segmentation Demo(SHTech dataset):

<img src="./demo/demo.gif" width="500" />

