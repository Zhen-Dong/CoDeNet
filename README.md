# CoDeNet
CoDeNet: Efficient Deployment of Input-Adaptive Object Detection on Embedded FPGAs. (FPGA 2021 Oral) \
This is the official implementation for CoDeNet, including training/testing/quantization codes and model zoo.

## Introduction
CoDeNet is an efficient object detection model on PyTorch, with SOTA performance on Pascal VOC and Microsoft COCO datasets under efficient settings.
It is based on CenterNet with co-designed deformable convolution and an efficient network architecture. It can run 27fps on an Ultra96 (ZU3EG) FPGA with 55.1 AP50 on Pascal VOC.

## Main Results
These are our main results on Pascal VOC dataset, taken from Table 3 in [our paper](https://arxiv.org/pdf/2006.08357.pdf).
|Detector	 |Resolution |DownSample	 |W/A-bit    |Model Size	|MACs	|Framerate| AP50   |
|------------|-----------|---------------|-----------|--------------|-------|---------|--------|		
|Tiny-YOLO				 |416x416	|MaxPool 		|W32A32		|60.5MB 	   |3.49G   | NA   | 57.1	|
|CoDeNet1x(config a)	 |256x256	|Stride4		|W4A8		|0.76MB 	   |0.29G   |32.2  | 51.1	|
|CoDeNet1x(config b)	 |256x256	|S2+MaxPool     |W4A8		|0.76MB	       |0.29G   |26.9  | 55.1	|
|CoDeNet1x(config c)	 |512x512	|Stride4		|W4A8		|0.76MB	       |1.14G   |9.3   | 61.7	|
|CoDeNet1x(config d)	 |512x512	|Stride4		|W4A8		|2.90MB	       |3.54G   |5.2   | 67.1	|
|CoDeNet1x(config e)	 |512x512	|S2+MaxPool     |W4A8		|2.90MB	       |3.58G   |4.6   | 69.7	|

These are our main results on Microsoft COCO dataset, taken from Table 4 in [our paper](https://arxiv.org/pdf/2006.08357.pdf).
|Detector	 |Weights    |Activations    |Model Size	|MACs	| AP   | AP50 | AP75 | APs | APm | APl
|------------|-----------|---------------|-------|-----------|--------------|-------|---------|-----------|--------|-------|
|CoDeNet1x   |32bit	     |32bit		|6.07 MB 	   |1.24G   |22.2   | 38.3	| 22.4  | 5.6 | 22.3 | 38.0 |
|CoDeNet1x   |4bit	     |8bit		|0.76 MB	   |1.24G   |18.8   | 33.9	| 18.7  | 4.6 | 19.2 | 32.2 |
|CoDeNet2x   |32bit	     |32bit		|23.4 MB	   |4.41G   |26.1   | 43.3	| 26.8  | 7.0 | 27.9 | 43.5 |
|CoDeNet2x	 |4bit	     |8bit		|2.93 MB	   |4.41G   |21.0   | 36.7	| 21.0  | 5.8 | 22.5 | 35.7 |


## Installation
1. Make a directory for CoDeNet, clone this repo and rename it as `src`.
```
mkdir CoDeNet
cd CoDeNet
git clone https://github.com/Zhen-Dong/CoDeNet.git
mv CoDeNet src
```
2. Build a new virtual environment with python3.6 and install the requirements.
```
pip install -r requirements.txt
```
3. Build the external library.
```
cd lib/models/external
make
```
Note that whenever one moves to a new python environment or a new machine, the external lib should be rebuilt.

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
5. Prepare data:
 - For COCO data, download the images (train 2017, test 2017, val 2017) and the annotation files (2017 train/val and test image info) from the [MS COCO dataset](http://cocodataset.org/#download).
 - For Pascal data, run the shell script `tools/get_pascal_voc.sh`. This includes downloading the images, downloading the annotations and merging the two annotation files into one json file.
 - Put the data directories under `CoDeNet/data`, and the structure should look like this.
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

## Usage Guide
<!-- Note: quantized model weights name problem? -->
For testing, use the pretrained models we provide for 5 configurations.
To test with config a:
```
python test.py ctdet --arch shufflenetv2 --exp_id pascal_shufflenetv2_config_a --dataset pascal --input_res 256 --resume --flip_test --gpu 0 --resume-quantize
```
To test with config b:
```
python test.py ctdet --arch shufflenetv2 --exp_id pascal_shufflenetv2_config_b --dataset pascal --input_res 256 --resume --flip_test --gpu 0 --maxpool --resume-quantize
```
To test with config c:
```
python test.py ctdet --arch shufflenetv2 --exp_id pascal_shufflenetv2_config_c --dataset pascal --input_res 512 --resume --flip_test --gpu 0 --resume-quantize
```
To test with config d:
```
python test.py ctdet --arch shufflenetv2 --exp_id pascal_shufflenetv2_config_d --dataset pascal --input_res 512 --resume --flip_test --gpu 0 --w2 --resume-quantize
```
To test with config e:
```
python test.py ctdet --arch shufflenetv2 --exp_id pascal_shufflenetv2_config_a --dataset pascal --input_res 512 --resume --flip_test --gpu 0 --w2 --maxpool --resume-quantize
```
For training, we provide training/fine-tuning code for ordinary model (`main.py`) and quantized model (`quant_main.py`). For example, to train the ordinary model, run
```
python main.py ctdet --arch shufflenetv2 --exp_id pascal_shufflenetv2_512_1 --dataset pascal --head_conv 64 --input_res 512 --num_epochs 90 --lr_step 50,70 --gpu 0
```
and to train the quantized model from an ordinary pretrained model, run
```
python quant_main.py ctdet --arch shufflenetv2 --exp_id pascal_shufflenetv2_512_1 --dataset pascal --head_conv 64 --input_res 512 --num_epochs 180 --lr_step 50,70 --gpu 0 --resume --resume-quantize --wt-percentile
```

## License
CoDeNet is released under the [MIT license](https://github.com/Zhen-Dong/CoDeNet/blob/main/LICENSE).
