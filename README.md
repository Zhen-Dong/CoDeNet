# CoDeNet
CoDeNet: Efficient Deployment of Input-Adaptive Object Detection on Embedded FPGAs. (FPGA 2021 Oral) \
This is the official implementation for CoDeNet, including training/testing/quantization codes and [model zoo]().

## Introduction
CoDeNet is an efficient object detection model on PyTorch, with SOTA performance on Pascal VOC and Microsoft COCO datasets under efficient setting.
It is based on CenterNet with co-designed deformable convolution and an efficient network architecture. It can run 27fps on an Ultra96 (ZU3EG) FPGA with 55.1 AP50 on Pascal VOC.

## Main Results
This is our main results on Pascal VOC dataset, taken from Table 3 in [our paper](https://arxiv.org/pdf/2006.08357.pdf)
|Detector	 |Resolution	|DownSample	 |Weights|Activations|Model Size	|MACs(G)	| Framerate   | AP50 |
|------------|-----------|---------------|-------|-----------|--------------|-------|---------|--------|		
|Tiny-YOLO				 |416x416	|MaxPool 		|32-bit	|32-bit		|60.5 MB 	   |3.49   | NA   | 57.1	|
|CoDeNet1x (config a)	 |256x256	|Stride4		|4-bit	|8-bit		|0.76 MB 	   |0.29   |32.2  | 51.1	|
|CoDeNet1x (config b)	 |256x256	|Stride2+MaxPool|4-bit	|8-bit		|0.76 MB	   |0.29   |26.9  | 55.1	|
|CoDeNet1x (config c)	 |512x512	|Stride4		|4-bit	|8-bit		|0.76 MB	   |1.14   |9.3   | 61.7	|
|CoDeNet1x (config d)	 |512x512	|Stride4		|4-bit	|8-bit		|2.90 MB	   |3.54   |5.2   | 67.1	|
|CoDeNet1x (config e)	 |512x512	|Stride2+MaxPool|4-bit	|8-bit		|2.90 MB	   |3.58   |4.6   | 69.7	|

## Installation
1. First make a directory for CoDeNet, clone this repo and rename it `src`
```
mkdir CoDeNet; cd CoDeNet
git clone https://github.com/Zhen-Dong/CoDeNet.git
mv CoDeNet src
```
2. Build a new virtual environment with python3.6.9 and install the requirements
```
pip install -r requirements.txt
```
3. Build the external library
```
cd lib/models/external
make
```
Note that whenever one moves to a new python environment or a new hardware, the external lib should be rebuilt.

4. Create directories for experiments, download our pretrained models from [google drive](https://drive.google.com/file/d/1kxw2zZmko5MP3RQlUf6kiapHrAKqIykD/view?usp=sharing) and put them under corresponding directories. The directories should look like this.
```
CoDeNet
|-- src
`-- exp
    `-- ctdet
        |-- pascal_shufflenetv2_config_a
        |   `-- model_last.pth
        |-- pascal_shufflenetv2_config_b
        |   `-- model_last.pth
        |-- pascal_shufflenetv2_config_c
        |   `-- model_last.pth
        |-- pascal_shufflenetv2_config_d
        |   `-- model_last.pth
        `-- pascal_shufflenetv2_config_e
            `-- model_last.pth
```
5. Prepare data
 - For COCO data, download the images (train 2017, test 2017, val 2017) and the annotation files (2017 train/val and test image info) from the [MS COCO dataset](http://cocodataset.org/#download).
 - For Pascal data, run the shell script `tools/get_pascal_voc.sh`. This includes downloading the images, downloading the annotations and merging the two annotation files into one json.
 - Put the data directories under `CoDeNet/data`, and make them look like this.
```
CoDeNet
|-- src
|-- exp
`-- data
   |-- coco
   |   |-- annotations
   |   `-- images
   |       |-- test2017
   |       |-- train2017
   |       `-- val2017
   `-- voc
       |-- annotations
       |-- images
       `-- VOCdevkit
```

## Quick Start
<!-- Note: quantized model weights name problem? -->
For testing, use the pretrained models we provide for 5 configurations.
To test with config a:
```
python test.py ctdet --arch shufflenetv2 --exp_id pascal_shufflenetv2_config_a --dataset pascal --input_res 256 --resume --flip_test --gpu 0
```
To test with config b:
```
python test.py ctdet --arch shufflenetv2 --exp_id pascal_shufflenetv2_config_b --dataset pascal --input_res 256 --resume --flip_test --gpu 0 --maxpool
```
To test with config c:
```
python test.py ctdet --arch shufflenetv2 --exp_id pascal_shufflenetv2_config_c --dataset pascal --input_res 512 --resume --flip_test --gpu 0
```
To test with config d:
```
python test.py ctdet --arch shufflenetv2 --exp_id pascal_shufflenetv2_config_d --dataset pascal --input_res 512 --resume --flip_test --gpu 0 --w2
```
To test with config e:
```
python test.py ctdet --arch shufflenetv2 --exp_id pascal_shufflenetv2_config_a --dataset pascal --input_res 512 --resume --flip_test --gpu 0 --w2 --maxpool
```
For training, we provide training code for ordinary model (`main.py`) and quantized model (`quant_main.py`). For example, to train the ordinary model, run
```
python main.py ctdet --arch shufflenetv2 --exp_id pascal_shufflenetv2_512_1 --dataset pascal --head_conv 64 --input_res 512 --num_epochs 90 --lr_step 50,70 --gpu 0
```
and to train the quantized model, run
```
python quant_main.py ctdet --arch shufflenetv2 --exp_id coco_shufflenetv2_512_1 --head_conv 64 --lr 5e-4 --batch_size 128 --input_res 512 --num_epochs 270 --lr_step 180,210 --gpu 12,13,14,15 --resume
```

## License
CoDeNet is released under the [MIT license](https://github.com/Zhen-Dong/CoDeNet/blob/main/LICENSE).
