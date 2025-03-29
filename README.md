# diffusion-based-motion-style-transfer

If our project is helpful for your research, please consider citing :
``` 
@inproceedings{hu2024diffusion,
  title={Diffusion-based Human Motion Style Transfer with Semantic Guidance},
  author={Hu, Lei and Zhang, Zihao and Ye, Yongjing and Xu, Yiwen and Xia, Shihong},
  booktitle={Computer Graphics Forum},
  volume={43},
  number={8},
  pages={e15169},
  year={2024},
  organization={Wiley Online Library}
}
```

## To-Do

- [x] Release the style transfer demo for Xia dataset.
- [x] Release the style finetuning code.
- [ ] Release the data process code.
- [ ] Release the diffusion-based T2M prior pre-training code.
- [ ] Release the motion semantic discriminator pre-training code.

## Environment
```bash
pip install -r requirements.txt
```

## Preparation
### Step 1: Download the pre-trained model
Please download the pretrained [motion prior (Text-to-Motion Prior)](https://drive.google.com/file/d/17jR4MPNjJezjXlUtpvhcjcJ9i7evZHiB/view?usp=sharing) and unzip in the root directory. 

### Step 2: Download the pre-processed Xia dataset
Please download the pre-processed [Xia dataset](https://drive.google.com/file/d/1cUQJdno5JlW98z5QLdcmE4JQv8GMU-Gt/view?usp=sharing) 

The raw motion bvh files (was retargeted to the format of the SMPL) can be download in this [url](https://drive.google.com/file/d/1fRvlSX9A1Srvx4TZMgCL_8laHll2EPa_/view?usp=sharing).

We will relese the data process script soon.

### Step 3: Download the SMPL files
Plase download the [smpl body models](https://drive.google.com/file/d/12-dmRfFvhq0QQv5kWNwqQ01pP3K1Ggvi/view?usp=sharing)



## Finetine the diffusion model for style transfer

### run the finetune_style_diffusion.py for few-shot style transfer
```bash
python -m train.finetune_style_diffusion --overwrite \
                --save_dir save_stylexia/inpainting_style_model \
                --dataset stylexia_posrot \
                --resume_checkpoint save_stylexia/inpainting_style_model/model_pretrained.pt \
                --style_example 286depressed_running.npy 

```
You can replace the style example motion by specifying --style_example. The fine-tuning probably lasts a few tens of seconds.


## run the demo
```bash
python -m sample.demo_style_transfer --model_path save_stylexia/inpainting_style_model/286depressed_running/model000000032.pt \
                                     --input_content 005childlike_normal walking.npy \ 
                                     --output_dir ./output_demo
```
You can revise the --input_content for changing the content motion.

## Acknowledgement
We want to thank the following contributors that our code is based on:

[guided-diffusion](https://github.com/openai/guided-diffusion), [motion-diffusion-model](https://github.com/GuyTevet/motion-diffusion-model/), [MotionCLIP](https://github.com/GuyTevet/MotionCLIP), [text-to-motion](https://github.com/EricGuo5513/text-to-motion)