# KOALAnet
**This is the official repository of "[KOALAnet: Blind Super-Resolution using Kernel-Oriented Adaptive Local Adjustment](https://arxiv.org/abs/2012.08103)", CVPR 2021.**

We provide the training and test code along with the trained weights and the test dataset. If you find this repository useful, please consider citing our [paper](https://arxiv.org/abs/2012.08103).

### Reference   
> Soo Ye Kim*, Hyeonjun Sim*, and Munchurl Kim, "KOALAnet: Blind Super-Resolution using Kernel-Oriented Adaptive Local Adjustment", CVPR, 2021. (* *equal contribution*)
> 
**BibTeX**
```bibtex
@InProceedings{Kim_2021_CVPR,
    author    = {Kim, Soo Ye and Sim, Hyeonjun and Kim, Munchurl},
    title     = {KOALAnet: Blind Super-Resolution Using Kernel-Oriented Adaptive Local Adjustment},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {10611-10620}
}
```
### Requirements
Our code is implemented using TensorFlow and was tested under the following setting:
* Python 3.6
* TensorFlow 1.13
* CUDA 10.0
* cuDNN7.4.1
* NVIDIA TITAN RTX
* Windows 10

## Test Code
1. Download the necessary files and place them in **<source_path>**:
    * Source code (main.py, koalanet.py, ops.py and utils.py)
    * Download the test dataset from [here](https://www.dropbox.com/sh/zkwia1ndleokeex/AAClDJY5sUDVWRLgSfi1sL3ka?dl=0).
    * Download the trained weights from [here](https://www.dropbox.com/sh/m0e2wezc2nv3z22/AAAaA-b1BGohioe4_EHzE_oIa?dl=0).
2. Set arguments defined in main.py and run main
    * Set ```--phase 'test'``` and provide the input and label paths to ```--test_data_path``` and ```--test_label_path``` and checkpoint path to ```--test_ckpt_path```.
    * Example: ```python main.py --phase 'test' --test_data_path './testset/Set5/LR/X4/imgs' --test_label_path './testset/Set5/HR' --test_ckpt_path './pretrained_ckpt'```
3. Result images will be saved in **<source_path>/results/imgs_test**.

### Notes
* Set ```--factor``` to ```2``` or ```4``` depending on your desired upscaling factor. 
* If you're using the provided testset, 6 datasets each with 2 scaling factors can be used. To try these out, set ```--test_data_path``` to ```'./testset/[dataset]/LR/X[factor]/imgs'``` and ```--test_label_path``` to ```'./testset/[dataset]/HR'```, where:
    * ```[dataset]: Set5, Set14, BSD100, Urban100, Manga109 or DIV2K```
    * ```[factor]: 2 or 4```
* If you want to test our model on your own data, set ```--test_data_path``` and ```--test_label_path``` to your desired path.
* If no ground truth HR images are available, set ```--eval False``` (defaults to True) to only save images without computing PSNR.
* If you're getting memory issues due to large input image sizes and limited memory, try setting ```--test_patch 2, 2 or 4, 4 ...``` (defaults to ```1, 1```). This option divides the input into an nxn grid, performs SR on each patch and stitches them back into a full image. Inference time measurements would be inaccurate in this case.
* When testing with your trained version, set ```--test_ckpt_path``` accordingly, to where you've stored the weights.

## Training Code
1. Download the necessary files and place them in **<source_path>**:
    * Source code (main.py, koalanet.py, ops.py and utils.py)
    * Download a training dataset (e.g. [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)).
2. 3-stage training:
    * Pretrain the downsampling network with ```python main.py --phase 'train' --training_stage 1```
    * Pretrain the upsampling network with ```python main.py --phase 'train' --training_stage 2```
    * Joint training of both networks with ```python main.py --phase 'train' --training_stage 3```
    * Set ```--training_data_path``` and ```--training_label_path``` to the directory containing training and validation data (e.g. ```python main.py --phase 'train' --training_stage 3 --training_data_path './dataset/DIV2K/train/DIV2K_train_HR' --validation_data_path './dataset/DIV2K/val/DIV2K_valid_HR'```)
3. Checkpoints will be saved in **<source_path>/ckpt**.
4. Monitoring training:
    * Intermediate results will be available in **<source_path>/results/imgs_train**.
    * A text log file and TensorBoard logs will be saved in **<source_path>/logs**.

### Notes
* Set ```--factor``` to ```2``` or ```4``` depending on your desired upscaling factor. 
* If ```--tensorboard True``` (defaults to True), tensorboard logs will be saved.
* Model settings (gaussian kernel size, local filter size in the downsampling and upsampling networks, etc) and hyperparameters (number of epochs, batch size, patch size, learning rate, etc) are defined as arguments. Default values are what we used for the paper. Please refer to main.py for details.

## Test Dataset
In blind SR, not a lot of benchmark datasets are available yet. We release the [random anisotropic Gaussian testset](https://www.dropbox.com/sh/zkwia1ndleokeex/AAClDJY5sUDVWRLgSfi1sL3ka?dl=0) we used in our paper, consisting of six datasets (Set5, Set14, BSD100, Urban100, Manga109 and DIV2K) and two scale factors (2 and 4). We hope that the community will use them for future research in SR.  

**Disclaimer:** The degradation kernels folder contains images of degradation kernels used for generating the corresponding LR image, *scaled and upsampled for better visualization*. They should only be used as visual reference.

## Contact
Please contact us via any of the following emails: sooyekim@kaist.ac.kr, flhy5836@kaist.ac.kr or leave a note in the issues tab.


