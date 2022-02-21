#### First install this `mmdetection` as `dev` mode following `docs/INSTALL.md`.

#### Important files: mmdet/apis/train.py, mmdet/apis/inference.py, tools/attack.py, tools/parsing.py

#### Before trying to run the attack, you need to download the pretrained weights according to the mmdetection model zoo and set the checkpoint path in `mmdet/apis/train.py`. Line 601, 651, 653 in `mmdet/apis/train.py` are set according to the number of black models to be evaluated with.
#### Further more, you need to set the `max_batch` in `mmdet/apis/train.py` to control the number of images to be attacked along with the `gpus` and `imgs_per_gpu` arguments from command line.
#### The path configuration in function `generate_data` in `tools/attack.py` must be consistent with that in `yolov3/test.py`. The script will load the saved images and labels to do the evaluation.


### Commands
```
cd tools
python attack.py --target_attack --generate_data --clear_output --seed 2 --gpus 1 --imgs_per_gpu 2 --work_dirs $dir$
```
The arguments `--generate_data` and `--clear_output` are used only when you try to evaluate the attack on yolov3. Otherwise, you won't need them, but remember to check Line 601, 651, 653 in `mmdet/apis/train.py` accordingly.

