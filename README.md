# Learning to Generate Diverse Pedestrian Movements from Web Videos with Noisy Labels

This is the official implementation of the ICLR 2025 paper, "Learning to Generate Diverse Pedestrian Movements from Web Videos with Noisy Labels", including the preprocessing of the CityWalkers dataset and the PedGen model code release.

<a href="https://arxiv.org/abs/2410.07500"><img src="https://img.shields.io/badge/arXiv-Paper-red"></a> 
<a href="https://genforce.github.io/PedGen"><img src="https://img.shields.io/badge/Project-Page-yellow"></a>

[Zhizheng Liu](https://scholar.google.com/citations?user=Asc7j9oAAAAJ&hl=en), [Joe Lin](https://github.com/joe-lin-tech), [Wayne Wu](https://wywu.github.io/), [Bolei Zhou](https://boleizhou.github.io/)
 <br>
     University of California, Los Angeles
 <br>
 ![Teaser](/docs/assets/teaser.jpg)

## Installation and Demo
Setup the repo:
```
git clone --recursive git@github.com:genforce/PedGen.git
cd PedGen
conda env create -f env.yaml -n pedgen 
conda activate pedgen
pip install -e .
```

Download checkpoints at this [link](https://drive.google.com/drive/folders/1JZAqARnNjt2H6vpGh2PJe09hYIC6S3Es?usp=sharing) in `ckpts` folder:
- `pedgen_no_context.ckpt`, PedGen model without context factors. 
- `pedgen_with_context.ckpt`, PedGen model with all context factors (scene, human, goal).


Download SMPL body models (SMPL_MALE.pkl, SMPL_FEMALE.pkl, SMPL_NEUTRAL.pkl) at this [link](https://smpl.is.tue.mpg.de/index.html) in `smpl` folder.

Run Demo:
```
python scripts/demo.py
```
Feel free to try different context factors to generate diverse movements.

 
## Preprocessing CityWalkers

Please check [preprocess.md](preprocess/preprocess.md) for details.

## Training & Evaluation
We use [lightning-cli](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) to do train/eval our model. 
To train on CityWalkers and reproduce our results run:
```
python scripts/main.py fit -c cfgs/pedgen_with_context.yaml --data.data_root $DATA_ROOT
```
, where $DATA_ROOT is the root of the preprocessed CityWalkers dataset. Additional information about CARLA evaluation can be found [here](pedgen/eval/carla_eval.md).

## Acknowledgements
We would like to thank the following projects for inspiring our work and open-sourcing their implementations: [WHAM](https://github.com/yohanshin/WHAM), [SLAHMR](https://github.com/vye16/slahmr), [MDM](https://github.com/GuyTevet/motion-diffusion-model), [HumanMAC](https://github.com/LinghaoChan/HumanMAC), [TRUMANS](https://github.com/jnnan/trumans_utils), [ZoeDepth](https://github.com/isl-org/ZoeDepth), [SegFormer](https://github.com/NVlabs/SegFormer), [SLOPER4D](https://github.com/climbingdaily/SLOPER4D).

## Contact

For any questions or discussions, please contact [Zhizheng Liu](zhizheng@cs.ucla.edu).

## Reference

If our work is helpful to your research, please cite the following:

```bibtex
@inproceedings{liu2025learning,
  title={Learning to Generate Diverse Pedestrian Movements from Web Videos with Noisy Labels},
  author={Liu, Zhizheng and Lin, Joe and Wu, Wayne and Zhou, Bolei},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```