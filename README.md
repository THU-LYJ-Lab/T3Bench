<img src="https://t3bench.com/static/images/logobig.png" align="center">

# T<sup>3</sup>Bench: Benchmarking Current Progress in Text-to-3D Generation

![](fig/A_cactus_with_pink_flowers.gif)

**T<sup>3</sup>Bench** is the first comprehensive text-to-3D benchmark containing diverse text prompts of three increasing complexity levels that are specially designed for 3D generation. To assess both the subjective quality and the text alignment, we propose two automatic metrics based on multi-view images produced by the 3D contents. The *quality* metric combines multi-view text-image scores and regional convolution to detect quality and view inconsistency. The *alignment* metric uses multi-view captioning and Large Language Model (LLM) evaluation to measure text-3D consistency. Both metrics closely correlate with different dimensions of human judgments, providing a paradigm for efficiently evaluating text-to-3D models.

<img src="https://t3bench.com/static/images/pipeline_v2.png">



## Evaluate on T<sup>3</sup>Bench

### Environment Setup

We adopt the implementation of <a href="https://github.com/threestudio-project/threestudio">ThreeStudio</a> to test the current text-to-3D methods. Please first follow the instructions of ThreeStudio to setup the generation environment.

Then install the following packages used for evaluation:

```shell
pip install -r requirements.txt
```

Note that we use a slightly modified version of ThreeStudio to ensure efficient generation.



### Evaluation

##### Run Text-to-3D and Extract Mesh

```shell
# YOUR_GROUP: Choose the prompt set to test, including [single, surr, multi]
# YOUR_METHOD: We now support latentnerf, magic3d, fantasia3d, dreamfusion, sjc, and prolificdreamer.
python run_t3.py --group YOUR_GROUP --gpu YOUR_GPU --method YOUR_METHOD
python run_mesh.py --group YOUR_GROUP --gpu YOUR_GPU --method YOUR_METHOD
```



##### Quality Evaluation

```shell
python run_eval_quality.py --group YOUR_GROUP --gpu YOUR_GPU --method YOUR_METHOD
```



##### Alignment Evaluation

```shell
# First get the 3D prompt of the text-to-3D result
python run_caption.py --group YOUR_GROUP --gpu YOUR_GPU --method YOUR_METHOD
# then run the LLM Evaluation
python run_eval_alignment.py --group YOUR_GROUP --gpu YOUR_GPU --method YOUR_METHOD
```



### Citation

```
TBD
```



### Acknowledgement

This project could not be possible without the open-source works from <a href="https://github.com/threestudio-project/threestudio">ThreeStudio</a>, <a href="https://github.com/crockwell/Cap3D">Cap3D</a>, <a href="https://github.com/ashawkey/stable-dreamfusion">Stable-DreamFusion</a>, <a href="https://github.com/THUDM/ImageReward">ImageReward</a>, <a href="https://github.com/salesforce/LAVIS">LAVIS</a>. We sincerely thank them all.
