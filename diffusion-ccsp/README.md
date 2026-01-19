## Setting Up

* Clone this repo

  ```shell
  git clone https://github.com/zt-yang/diffusion-ccsp.git --recurse-submodules
  ```

* Set up Jacinle following the instructions [here](https://github.com/vacancy/Jacinle). If the directory you installed Jacinle does not share the same parent folder as this repository, put `export PYTHONPATH=/your/path/to/Jacinle:$PYTHONPATH` in `diffusion-ccsp/setup.sh`.

    ```shell
    cd ..
    git clone https://github.com/vacancy/Jacinle --recursive
    ## echo "export PYTHONPATH=/your/path/to/Jacinle:$PYTHONPATH" >> diffusion-ccsp/setup.sh  ## optional
    ```

* Set up dependencies.

    ```shell
    cd diffusion-ccsp
    conda create --name diffusion-ccsp python=3.9
    conda activate diffusion-ccsp
    pip install -r requirements.txt
    ```

* Source environment variables before running codes (includes `conda activate diffusion-ccsp`).

    ```shell
    source setup.sh
    ```

* Compile IK for Franka Panda if want to collect and test robot planning.

    ```shell
    (cd pybullet_engine/ikfast/franka_panda; python setup.py)
    ```

## Download data and pre-trained models

By default, download for task `RandomSplitQualitativeWorld`. Download into `data/`, `logs/`, and `wandb/` folder

```shell
python download_data_checkpoints.py
```

## Solving CCSP

```shell
python solve_csp.py
```


## Training

```shell
python train_ddpm.py -timesteps 1000 -EBM 'ULA' -input_mode qualitative
```


