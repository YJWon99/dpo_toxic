# AI707 Final Project
## Mechanistically Understanding DPO: Toxicity

This repository provides the models, data, and experiments used in [A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity](https://arxiv.org/abs/2401.01967).

## Models, Data

You can download the models and datasets used in our paper [here](https://drive.google.com/drive/folders/1baArqcjIc2Q4OllLVUz1hp3p3XxmdteK?usp=drive_link).

Save the checkpoints under `./checkpoints` and unzip the data files under `./data`.

## Experiments

All of our experiments can be found under `./toxicity`.
To run interventions, see `./toxicity/eval_interventions/run_evaluations.py`.

To re-create any of our figures, see `./toxicity/eval_interventions/figures`.

## Training DPO

To train your own dpo model:
```bash
cd toxicity/train_dpo
CUDA_VISIBLE_DEVICES=0 python train.py exp_name="[name of your experiment]"
```
*Note: Training SFT and DPO models takes less than 10 minutes on a single A100 GPU.*

## Sampling Responses
See `./toxicity/train_dpo/sample_ratio.py` for details for the hyper-parameter setup.
To sample from the DPO/SFT trained models, and the Pi_l distribution:
```bash
cd toxicity/train_dpo
CUDA_VISIBLE_DEVICES=0 python sample_ratio.py > PATH_TO_PRINT_RESULTS.log # 
```
See `./sample_from_pil.log` for response samples.


## How to Cite

If you find our work relevant, please cite as following:

```
@inproceedings{lee2024mechanistic,
  title={A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity},
  author={Lee, Andrew and Bai, Xiaoyan and Pres, Itamar and Wattenberg, Martin and Kummerfeld, Jonathan K and Mihalcea, Rada},
  booktitle={International Conference on Machine Learning},
  pages={26361--26378},
  year={2024},
  organization={PMLR}
}
```
