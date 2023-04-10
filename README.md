
# Bidirectional Transformer Network with Logits Compensation for Unbiased Scene Graph Generation



# Requirements:

```
Python <= 3.8
PyTorch >= 1.2 (Mine 1.7.1 (CUDA 11.3))
torchvision >= 0.4 (Mine 0.8.2 (CUDA 11.3))
cocoapi
yacs
matplotlib
GCC >= 4.9
OpenCV
```

# Step-by-step installation  



```
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name scene_graph_benchmark_new
conda activate scene_graph_benchmark_new

# this installs the right pip and dependencies for the fresh python
conda install ipython
conda install scipy
conda install h5py

# scene_graph_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python overrides

# pytorch and torchvision
https://download.pytorch.org/whl/torch_stable.html


# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex

# WARNING if you use older Versions of Pytorch (anything below 1.7), you will need a hard reset,
# as the newer version of apex does require newer pytorch versions. Ignore the hard reset otherwise.
git reset --hard 3fe10b5597ba14a748ebb271a6ab97c09c5701ac

python setup.py install --cuda_ext --cpp_ext


# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
# !!!!!!!!!!!!!!!!!!!!!!!!!!! very important !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
python setup.py build develop


```

# DATASET

The following is adapted from [Danfei Xu](https://github.com/danfeiX/scene-graph-TF-release/blob/master/data_tools/README.md) and [neural-motifs](https://github.com/rowanz/neural-motifs).

Note that our codebase intends to support attribute-head too, so our `VG-SGG.h5` and `VG-SGG-dicts.json` are different with their original versions in [Danfei Xu](https://github.com/danfeiX/scene-graph-TF-release/blob/master/data_tools/README.md) and [neural-motifs](https://github.com/rowanz/neural-motifs). We add attribute information and rename them to be `VG-SGG-with-attri.h5` and `VG-SGG-dicts-with-attri.json`. The code we use to generate them is located at `datasets/vg/generate_attribute_labels.py`.  Although, we encourage later researchers to explore the value of  attribute features, in our paper "Unbiased Scene Graph Generation from  Biased Training", we follow the conventional setting to turn off the  attribute head in both detector pretraining part and relationship  prediction part for fair comparison, so does the default setting of this  codebase.

### Download:

1. Download the VG images [part1 (9 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2 (5 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract these images to the file `datasets/vg/VG_100K`. If you want to use other directory, please link it in `DATASETS['VG_stanford_filtered']['img_dir']` of `maskrcnn_benchmark/config/paths_catelog.py`.
2. Download the [scene graphs](https://1drv.ms/u/s!AmRLLNf6bzcir8xf9oC3eNWlVMTRDw?e=63t7Ed) and extract them to `datasets/vg/VG-SGG-with-attri.h5`, or you can edit the path in `DATASETS['VG_stanford_filtered_with_attribute']['roidb_file']` of `maskrcnn_benchmark/config/paths_catalog.py`.



# Faster R-CNN pre-training

The following command can be used to train your own Faster R-CNN model:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 10001 --nproc_per_node=4 tools/detector_pretrain_net.py --config-file "configs/e2e_relation_detector_X_101_32_8_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 4 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.STEPS "(30000, 45000)" SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 MODEL.RELATION_ON False OUTPUT_DIR /data/xuhongbo/checkpoints/pretrained_faster_rcnn SOLVER.PRE_VAL False
```

where `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` represent the id of GPUs and number of GPUs you use, `--config-file` means the config we use, where you can change other parameters. `SOLVER.IMS_PER_BATCH` and `TEST.IMS_PER_BATCH` are the training and testing batch size respectively, `DTYPE "float16"` enables Automatic Mixed Precision supported by [APEX](https://github.com/NVIDIA/apex), `SOLVER.MAX_ITER` is the maximum iteration, `SOLVER.STEPS` is the steps where we decay the learning rate, `SOLVER.VAL_PERIOD` and `SOLVER.CHECKPOINT_PERIOD` are the periods of conducting val and saving checkpoint, `MODEL.RELATION_ON`  means turning on the relationship head or not (since this is the  pretraining phase for Faster R-CNN only, we turn off the relationship  head),  `OUTPUT_DIR` is the output directory to save checkpoints and log (considering `/home/username/checkpoints/pretrained_faster_rcnn`), `SOLVER.PRE_VAL` means whether we conduct validation before training or not.



# Protocols

There are **three standard protocols**: (1)  Predicate Classification (PredCls): taking ground truth bounding boxes  and labels as inputs, (2) Scene Graph Classification (SGCls) : using  ground truth bounding boxes without labels, (3) Scene Graph Detection  (SGDet): detecting SGs from scratch. We use two switches `MODEL.ROI_RELATION_HEAD.USE_GT_BOX` and `MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL` to select the protocols.

For **Predicate Classification (PredCls)**, we need to set:

```
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True
```

For **Scene Graph Classification (SGCls)**:

```
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```

For **Scene Graph Detection (SGDet)**:

```
MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```



# Predefined Models

We abstract various SGG models to be different `relation-head predictors` in the file `roi_heads/relation_head/roi_relation_predictors.py`,  which are independent of the Faster R-CNN backbone and relation-head  feature extractor. To select our predefined models, you can use `MODEL.ROI_RELATION_HEAD.PREDICTOR`.

For [Neural-MOTIFS](https://arxiv.org/abs/1711.06640) Model:

```
MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor
```

For [Iterative-Message-Passing(IMP)](https://arxiv.org/abs/1701.02426) Model (Note that SOLVER.BASE_LR should be changed to 0.001 in SGCls, or the model won't converge):

```
MODEL.ROI_RELATION_HEAD.PREDICTOR IMPPredictor
```

For [VCTree](https://arxiv.org/abs/1812.01880) Model:

```
MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor
```

For our predefined Signal-Transformer Model (Note that  Transformer Model needs to change SOLVER.BASE_LR to 0.001,  SOLVER.SCHEDULE.TYPE to WarmupMultiStepLR, SOLVER.MAX_ITER to 16000,  SOLVER.IMS_PER_BATCH to 16, SOLVER.STEPS to (10000, 16000).)

```
MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor
```

For our BTrans Model:

```
MODEL.ROI_RELATION_HEAD.PREDICTOR BiTransformerPredictor
```

The default settings are under `configs/e2e_relation_X_101_32_8_FPN_1x.yaml` and `maskrcnn_benchmark/config/defaults.py`. The priority is `command > yaml > defaults.py`





# Examples of the Training Command

Training Example 1 : (PreCls, Motif Model, LC)

```
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.SOFTENING_FACTOR 10.0 SOLVER.SCALING_FACTOR 1.0 SOLVER.CHECKPOINT_PERIOD 4000 GLOVE_DIR /data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_BCL/glove MODEL.PRETRAINED_DETECTOR_CKPT /data/xuhongbo/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_BCL/output/predcls-motifs-LC
```

Training Example 2 : (SGCls, VCTree Model, LC)

```
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --master_port 10059 --nproc_per_node=1 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.SOFTENING_FACTOR 10.0 SOLVER.SCALING_FACTOR 1.0 SOLVER.CHECKPOINT_PERIOD 4000 GLOVE_DIR /data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_BCL/glove MODEL.PRETRAINED_DETECTOR_CKPT /data/xuhongbo/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_BCL/output/sgcls-vctree-LC
```

Training Example 2 : (SGDet, BTrans Model, LC)

```
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --master_port 10051 --nproc_per_node=1 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR BiTransformerPredictor  SOLVER.SCHEDULE.TYPE  WarmupMultiStepLR SOLVER.BASE_LR  0.001 SOLVER.STEPS "(10000, 16000)" SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 16000 SOLVER.VAL_PERIOD 2000 SOLVER.SOFTENING_FACTOR 10.0 SOLVER.SCALING_FACTOR 1.0 SOLVER.CHECKPOINT_PERIOD 4000 GLOVE_DIR /data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_BCL/glove MODEL.PRETRAINED_DETECTOR_CKPT /data/xuhongbo/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_BCL/output/sgdet-BTrans-LC
```





# Visualization

The following file can be used for model visualization:

`"./tools/visual_SGDET.py"`

# Acknowledge
This work is developed based on the SGG benchmark https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch


