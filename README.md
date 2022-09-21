<div align="center">

<h1>Text2Light: Zero-Shot Text-Driven HDR Panorama Generation</h1>

<div>
    <a href='https://frozenburning.github.io/' target='_blank'>Zhaoxi Chen</a>&emsp;
    <a href='https://wanggcong.github.io/' target='_blank'>Guangcong Wang</a>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu</a>
</div>
<div>
    S-Lab, Nanyang Technological University
</div>

<strong><a href='https://sa2022.siggraph.org/' target='_blank'>TOG 2022 (Proc. SIGGRAPH Asia)</a></strong>

### [Project Page](https://frozenburning.github.io/projects/text2light) | [Video](https://youtu.be/XDx6tOHigPE) | [Paper](https://arxiv.org/abs/2209.09898)

<tr>
    <img src="https://github.com/FrozenBurning/FrozenBurning.github.io/blob/master/projects/text2light/img/teaser.gif" width="100%"/>
</tr>

</div>

## Updates

[09/2022] Paper uploaded to arXiv. [![arXiv](https://img.shields.io/badge/arXiv-2209.09898-b31b1b.svg)](https://arxiv.org/abs/2209.09898)

[09/2022] Model weights released. [![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=yellow)](https://drive.google.com/drive/folders/1X8wUNjYYQ8l3YG1_Fwb0CqFvKvKOQ3Go?usp=sharing)

[09/2022] Code released.

## Installation
We highly recommend using [Anaconda](https://www.anaconda.com/) to manage your python environment. You can setup the required environment by the following command:
```bash
conda env create -f environment.yml
conda activate text2light
```

## Text-driven HDRI Generation

You may do the following steps to generate HDR panoramas from free-form texts with our models.
### Download Pretrained Models
Please download our checkpoints from [Google Drive](https://drive.google.com/drive/folders/1X8wUNjYYQ8l3YG1_Fwb0CqFvKvKOQ3Go?usp=sharing) to run the following inference scripts. We recommend to save the downloaded models in the `./logs` folder.

### All-in-one Inference Script
All inference codes are in [text2light.py](text2light.py), you can learn to use it by:
```bash
python text2light.py -h
```

Here are some examples, the output will be saved in `./generated_panorama`:
- Generate a HDR panorama from a single sentence:
    ```bash
    python text2light.py -rg logs/global_sampler_clip -rl logs/local_sampler_outdoor --outdir ./generated_panorama --text "YOUR SCENE DESCRIPTION" --clip clip_emb.npy --sritmo ./logs/sritmo.pth --sr_factor 4
    ```

- Generate HDR panoramas from a list of texts:
    ```bash
    # assume your texts is stored in alt.txt
    python text2light.py -rg logs/global_sampler_clip -rl logs/local_sampler_outdoor --outdir ./generated_panorama --text ./alt.txt --clip clip_emb.npy --sritmo ./logs/sritmo.pth --sr_factor 4
    ```

- Generate low-resolution (512x1024) LDR panoramas only:
    ```bash
    # assume your texts is stored in alt.txt
    python text2light.py -rg logs/global_sampler_clip -rl logs/local_sampler_outdoor --outdir ./generated_panorama --text ./alt.txt --clip clip_emb.npy
    ```

## Rendering

Our generated HDR panoramas can be directly used in any modern graphics pipeline as the environment texture and light source. Here we take [Blender](https://www.blender.org/) as an example.

### From GUI
Open Blender -> Select `Shading` Panel -> Select `Shader Type` as `World` -> Add an `Environment Texture` node -> Browse and select our generated panoramas -> Render

You can also refer to this [tutorial](https://www.youtube.com/watch?v=gC4Uqr4E78U).

### From Command line
For the ease of batch processing, e.g. rendering with multiple HDRIs, we offer scripts in command line for rendering your 3D assets.

1. Download the Linux version of Blender from [Blender Download Page](https://www.blender.org/download/).
2. Unpack it and check the usage of Blender:
    ```bash
    # assume your downloaded version is 3.1.2
    tar -xzvf blender-3.1.2-linux-x64.tar.xz
    cd blender-3.1.2-linux-x64
    ./blender --help
    ```
3. Add an alias to your .bashrc or .zshrc:
    ```bash
    # PATH_TO_DOWNLOADED_BLENDER indicates the parent directory where you save the downloaded blender
    alias blender="/PATH_TO_DOWNLOADED_BLENDER/blender-3.1.2-linux-x64/blender"
    ```
4. Back to the codebase of Text2Light, and run the following commands for different rendering setup:
    - Render four shader balls given all HDRIs stored at `PATH_TO_HDRI`
    ```bash
    blender --background --python rendering_shader_ball.py -- ./rendered_balls 100 1000 PATH_TO_HDRI
    ```


## Training
Our training is stage-wise with multiple steps. The details are listed as follows.

### Data Preparation
Assume all your HDRIs for training are stored at `PATH_TO_HDR_DATA`, please run [process_hdri.py](./process_hdri.py) to process the data:
```bash
python process_hdri.py --src PATH_TO_HDR_DATA
```
The processed data will be saved to `./data` by default and organized as follows: 
```
├── ...
└── Text2Light/
    ├── data/
        ├── train/
            ├── calib_hdr
            ├── ldr
            └── raw_hdr
        ├── val/
            ├── calib_hdr
            ├── ldr
            └── raw_hdr
        └── meta/
```

### Stage I - Text-driven LDR Panorama Generation

The training stage1 is launched by [train_stage1.py](train_stage1.py), you can check the usage by:
```bash
python train_stage1.py -h
```

1) Train the global codebook
    ```bash
    python train_stage1.py --base configs/global_codebook.yaml -t True --gpu 0,1,2,3,4,5,6,7
    ```
2) Train the local codebook
    ```bash
    python train_stage1.py --base configs/local_codebook.yaml -t True --gpu 0,1,2,3,4,5,6,7
    ```
3) Train the text-conditioned global sampler. Please specify the path to global codebook in the config YAML.
    ```bash
    python train_stage1.py --base configs/global_sampler_clip.yaml -t True --gpu 0,1,2,3,4,5,6,7 
    ```
4) Train the structure-aware local sampler. Please specify the path to global and local codebooks in the config YAML, respectively.
    ```bash
    python train_stage1.py --base configs/local_sampler_spe.yaml -t True --gpu 0,1,2,3,4,5,6,7 
    ```

### Stage II - Super-resolution Inverse Tonemapping

The training stage2 is launched by [train_stage2.py](train_stage2.py), you can check the usage by:
```bash
python train_stage2.py -h
```

The default setting can be trained on a single A100 GPU without DDP:
```bash
# assume you use the default --dst_dir in process_hdri.py, thus the hdr dataset would be stored in ./data
python train_stage2.py --dir ./data --save_dir ./output/bs32_7e-5 --workers 16 --val_ep 5 --gpu 0
```

To enable distributed training, for example, over 8 GPUs:
```bash
python train_stage2.py --dir ./data --save_dir ./output/bs32_7e-5 --workers 8 --val_ep 5 --ddp
```


## Acknowledgements
This work is supported by the National Research Foundation, Singapore under its AI Singapore Programme, NTU NAP, MOE AcRF Tier 2 (T2EP20221-0033), and under the RIE2020 Industry Alignment Fund - Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s).

Text2Light is implemented on top of the [VQGAN](https://github.com/CompVis/taming-transformers) codebase. We also thanks [CLIP](https://github.com/openai/CLIP) and [LIIF](https://github.com/yinboc/liif) for their released models and codes. Thanks this [repo](https://github.com/yuki-koyama/blender-cli-rendering) for its amazing command line rendering toolbox.