# DOC
DOC: Text Recognition via Dual Adaptation and Clustering

## Environment
 `cuda==10.1, python==3.6.8`.
 
 ## Install
 `pip install torch==1.2.0 pillow==6.2.1 torchvision==0.4.0 lmdb nltk natsort`
 
 ## Datasets
 The prepared synthetic text dataset and real scene text datasets can be download from [here](https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0)[1]
 1. Synthetic text dataset
 * [MJSynth(MJ)](https://www.robots.ox.ac.uk/~vgg/data/text/)
 * [SynthText(ST)](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)
 2. Real scene text dataset
 * The union of the training datasets [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html), [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset), [IC13](https://rrc.cvc.uab.es/?ch=2), [IC15](https://rrc.cvc.uab.es/?ch=4)
 * Benchmark evaluation scene text datasets:  [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html), [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset), [IC03](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions), [IC13](https://rrc.cvc.uab.es/?ch=2), [IC15](https://rrc.cvc.uab.es/?ch=4), [SVTP](https://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf), [CUTE](http://cs-chan.com/downloads_CUTE80_dataset.html)
 3. Handwritten text dataset can be download from [here](https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW)[2]
 * [IAM](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
 ## Pretrained model
 You can download the pretrained models from [here](https://www.dropbox.com/sh/4a9vrtnshozu929/AAAZucKLtEAUDuOufIRDVPOTa?dl=0)[1]
 ## Training and evaluation
 * Training
```
 CUDA_VISIBLE_DEVICES=1 python train.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--src_train_data ./data/data_lmdb_release/training/ \
--tar_train_data ./data/IAM/ --tar_select_data IAM --tar_batch_ratio 1 --valid_data ./data/IAM/ \
--continue_model ./data/TPS-ResNet-BiLSTM-Attn.pth \
--batch_size 192 --lr 1 \
--experiment_name _adv_global_local_synth2iam_pc_0.1 --pc 0.1
```
* Evaluation
```
 CUDA_VISIBLE_DEVICES=0 python test.py   --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn   \
 --eval_data ./data/IAM \
 --saved_model ./data/TPS-ResNet-BiLSTM-Attn.pth 
 ```
 ## Reference
 [1] Baek, Jeonghun, et al. "What is wrong with scene text recognition model comparisons? dataset and model analysis." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.\
 [2] Zhang, Yaping, et al. "Robust text image recognition via adversarial sequence-to-sequence domain adaptation." IEEE Transactions on Image Processing 30 (2021): 3922-3933.
 ## Acknowledgement
 This implementation is based on [Seq2SeqAdapt](https://github.com/AprilYapingZhang/Seq2SeqAdapt) and [pytorch-center-loss](https://github.com/KaiyangZhou/pytorch-center-loss)
