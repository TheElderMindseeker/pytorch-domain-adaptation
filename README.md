# Domain Adaptation from Synthetic to Real Data

This repository is based on [Pytorch Domain Adaptation algorithms](https://github.com/jvanvugt/pytorch-domain-adaptation). Here we implement the models for domain adapting from synthetic data (GTA frames) to real data (UCF Crime frames).

## How to run the code

First, you need to install all dependencies using `pipenv`:

```bash
pipenv --python 3.8
pipenv sync
```

Then, put the dataset with GTA frames into folder named `data` at the root of the repository. The folder with UCF-Crime frames should be put under name `t_data` next to the GTA frames dataset. You can find both folders with proper names on the [Google Drive](https://drive.google.com/file/d/0B0ou2bBNVnnSTTZaeEx2OXA4NUU/view?usp=sharing). Just unzip the folder into the repository root and you will be set.

You need to train the source model to perform the domain adaptation. To do that, run:

```bash
pipenv run python train_source.py
```

You can view available options for the script using

```bash
pipenv run python train_source.py --help
```

Three types of models are available for training: `gta`, `gta-res`, and `gta-vgg`. The models will be put into the folder `trained_models` under the corresponding names.

After training the source models you can domain adapt them using one of three methods contained in the corresponding scripts: `adda.py`, `revgrad.py`, and `wdgrl.py`. You can use scripts' `--help` option to view the available options. For example, you can run:

```bash
pipenv run python adda.py --model gta-res
```

to domain adapt ResNet-based model from GTA to UCF-Crime.

Finally, you can test the trained models on target data to measure their performance:

```bash
pipenv run python test_model.py --model gta-res-adda
```

Use `--help` option to view the script's availalbe options.

### Bonus: Making Live Analysis Videos

You can use script named `presentation_video.py` to make videos with pseudo-live analysis. Choose a video with possible crime in it, name it `sample_video.avi` (or change the name in the script to what you need), put it into the repository root folder, then run:

```bash
pipenv run python presentation_video.py
```

The script outputs `output_video.mp4` into the same folder. By default, the script uses ResNet-based network unadapted, but you can change it if you need. Some sample videos can be found on the [Google Drive](https://drive.google.com/file/d/1FRbMxr1kwZbc5E846OTxGjNfSBJglGKC/view?usp=sharing).
