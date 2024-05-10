# plant-traits-kaggle

Helping ecologist generalize models for predicting plant traits to better understand the health of ecosystems.

Code for this Kaggle competition: https://www.kaggle.com/competitions/planttraits2024/overview

For additional notes see this doc: https://docs.google.com/document/d/1YLDUVcI2sjkkCSk9zewKPOpY5vFpeMFmkDEsXBXCNCU/edit



### Quickstart guide
To get up and running with the code, follow these steps. Make sure you have Docker installed.

1. `bin/build.sh`

2. Download `planttraits2024.zip` from the Kaggle comp linked above. Unzip it in `data/raw`

3. `bin/preprocess_data.sh` to prepare the data and precompute embeddings

4. For W&B logging, contact Nathan to be added to https://wandb.ai/nathan-mandi/PlantTraits2024. Then, create a `.env` file with your `WANDB_API_KEY`

You are set up! You should now be able to run `bin/train.sh` and other scripts to train models, interact with the code in the repo, etc.

### Testing

Most of the modules have tests. Run them with `bin/test.sh` and pass in any args you would pass to `pytest`.
Ex: `bin/test.sh -s tests/data/datasets/test_baseline_dataset.py`


## potluck template readme
This repo uses Potluck, a machine learning repo template from Kung Fu AI. Here are instructions for use.

### Requirements

- [Docker][docker-url]
- [Docker Compose][docker-compose-url]
- [NVIDIA Docker Container Runtime][nvidia-url]

### Quick Start

- Run `bin/build.sh`
- Run `bin/preprocess_data.sh`


### Training

To train the model, all you need to do is run this command:

```sh
bin/train.sh
```

(Note: Please include further instructions if GPU is required!)

### Testing

Once the Docker image is built we can run the project's unit tests to verify everything is
working. The below command will start a Docker container and execute all unit tests using the
[pytest framework](https://docs.pytest.org/en/latest/).

```sh
bin/test.sh
```

If you want to run a test on a specific file or directory (rather than running all the tests in
the tests/ directory), you can use the `-k` flag and list the file path afterwards.

For example, if we specifically wanted to run a test called "test_api", and its file path is as
"tests/test_api.py", we can run:

```shell script
bin/test.sh -k test_api.py
```

### Summary of Commands

The `bin/` directory contains basic shell bin that allow us access to common commands on most
environments. We're not guaranteed much functionality on any generic machine, so keeping these
basic is important.

The most commonly used bin are:

- `bin/build.sh` - build docker container(s) defined in `Dockerfile` and `compose.yaml`
- `bin/test.sh` - run unit tests defined in `tests/`
- `bin/notebook.sh` - instantiate a new jupyter notebook server
- `bin/shell.sh` - instantiate a new bash terminal inside the container
- `bin/train.sh` - train a model

Additional bin:

- `bin/lint.sh` - check code formatting for the project
- `bin/setup_environment.sh` - sets any build arguments or settings for all containers brought up
  with docker compose
- `bin/up.sh` - bring up all containers defined in `compose.yaml`
- `bin/down.sh` - stops all containers defined in `compose.yaml` and removes associated
  volumes, networks and images

## Data Directory

Data organization philosophy from [cookiecutter data
science](https://github.com/drivendata/cookiecutter-data-science)

```
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
```
