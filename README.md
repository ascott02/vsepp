# VSEPP Web Service

This is a fork of the [vsepp](https://github.com/fartashf/vsepp) project. Their original README.me is below. This project extends that code, and adds a memory resident webservice using [web.py](https://webpy.org).

Depends on Python 2.7, PyTorch==1.4.0, and tensorflow-gpu==1.15.0 for Cuda 10 .

Get the code and install dependencies:

    git clone https://github.com/ascott02/vsepp.git 
    cd vsepp
    virtualenv --python=python2.7 ../venv
    source ../venv/bin/activate
    pip install -r requirements_build.txt # installs cython and numpy
    pip install -r requirements.txt       # installs the rest

Download the punkt dictionary

    source ../code/venv/bin/activate
    python -m nltk.downloader punkt

Get the data and models 

    wget -v http://www.cs.toronto.edu/~faghri/vsepp/vocab.tar 
    wget -v http://www.cs.toronto.edu/~faghri/vsepp/data.tar 
    wget -v http://www.cs.toronto.edu/~faghri/vsepp/runs.tar

Extract those to somewhere you have a lot of space

    mkdir /mnt/BIGSPACE/vsepp
    cd /mnt/BIGSPACE/vsepp
    tar -xvf vocab.tar
    tar -xvf data.tar
    tar -xvf runs.tar

Symlink from your vsepp checkout dir to your unpacked dirs

    cd ~/code/vsepp # or wherever you had checked it out
    ln -s /mnt/BIGSPACE/vsepp/vocab 
    ln -s /mnt/BIGSPACE/vsepp/data
    ln -s /mnt/BIGSPACE/vsepp/runs

Run the test_eval.py script to test it out
  
    cd ~/code/vsepp # or wherever you had checked it out
    source ../venv/bin/activate
    python test_eval.py

Run the webservice. *Note: Ask Andrew for the config.py not in repo*

    cd ~/code/vsepp # or wherever you had checked it out
    source ../venv/bin/activate
    cd webservice
    python main.py [port]   # port is optional, defaults to 8080 

Original README.md follows

# Improving Visual-Semantic Embeddings with Hard Negatives

Code for the image-caption retrieval methods from
**[VSE++: Improving Visual-Semantic Embeddings with Hard Negatives](https://arxiv.org/abs/1707.05612)**
*, F. Faghri, D. J. Fleet, J. R. Kiros, S. Fidler, Proceedings of the British Machine Vision Conference (BMVC),  2018. (BMVC Spotlight)*

## Dependencies
We recommended to use Anaconda for the following packages.

* Python 2.7 (Checkout branch `python3`)
* [PyTorch](http://pytorch.org/) (>0.2) (Checkout branch `pytorch4.1`)
* [NumPy](http://www.numpy.org/) (>1.12.1)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)
* [pycocotools](https://github.com/cocodataset/cocoapi)
* [torchvision]()
* [matplotlib]()


* Punkt Sentence Tokenizer:
```python
import nltk
nltk.download()
> d punkt
```

## Download data

Download the dataset files and pre-trained models. We use splits produced by [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/deepimagesent/). The precomputed image features are from [here](https://github.com/ryankiros/visual-semantic-embedding/) and [here](https://github.com/ivendrov/order-embedding). To use full image encoders, download the images from their original sources [here](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html), [here](http://shannon.cs.illinois.edu/DenotationGraph/) and [here](http://mscoco.org/).

```bash
wget http://www.cs.toronto.edu/~faghri/vsepp/vocab.tar
wget http://www.cs.toronto.edu/~faghri/vsepp/data.tar
wget http://www.cs.toronto.edu/~faghri/vsepp/runs.tar
```

We refer to the path of extracted files for `data.tar` as `$DATA_PATH` and 
files for `models.tar` as `$RUN_PATH`. Extract `vocab.tar` to `./vocab` 
directory.

## Evaluate pre-trained models

```python
from vocab import Vocabulary
import evaluation
evaluation.evalrank("$RUN_PATH/coco_vse++/model_best.pth.tar", data_path="$DATA_PATH", split="test")'
```

To do cross-validation on MSCOCO, pass `fold5=True` with a model trained using 
`--data_name coco`.

## Training new models
Run `train.py`:

```bash
python train.py --data_path "$DATA_PATH" --data_name coco_precomp --logger_name 
runs/coco_vse++ --max_violation
```

Arguments used to train pre-trained models:

| Method    | Arguments |
| :-------: | :-------: |
| VSE0      | `--no_imgnorm` |
| VSE++     | `--max_violation` |
| Order0    | `--measure order --use_abs --margin .05 --learning_rate .001` |
| Order++   | `--measure order --max_violation` |


## Reference

If you found this code useful, please cite the following paper:

    @article{faghri2018vse++,
      title={VSE++: Improving Visual-Semantic Embeddings with Hard Negatives},
      author={Faghri, Fartash and Fleet, David J and Kiros, Jamie Ryan and Fidler, Sanja},
      booktitle = {Proceedings of the British Machine Vision Conference ({BMVC})},
      url = {https://github.com/fartashf/vsepp},
      year={2018}
    }

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
