# GABDataset Experiments on Gender Bias in Vision-Language Models

In this repository, we provide the code and dataset (GABDataset) used to examine gender bias in Vision-Language Models (VLMs) through various experiments described in the main paper. The repository is organized into three phases: **phaze1**, **phaze2**, and **phaze3**, with corresponding directories for each experiment.

## Repository Structure

### 1. **BiasExperiment Directory**

This directory contains the notebooks for measuring subject binding bias as detailed in the paper. For each phase (phaze1, phaze2, phaze3), you will find:

- **Experiment Notebook:** Used to conduct the experiment.
- **Results Notebook:** Used to aggregate and plot the results.

### 2. **TextEncoderBiasExperiment Directory**

This directory focuses on measuring text encoder bias. It includes:

- **Experiment Notebook:** Used to perform the experiment.
- **Results Notebook:** Used to aggregate the results.

### 3. **TextToImageRetrievalExperiment Directory**

In this directory, we assess the model's capability for text-to-image retrieval. It includes:

- **Experiment Notebook:** Used to execute the experiment.
- **Results Notebook:** Used to aggregate and plot the results.

### 4. **ActivityRetrievalExperiment Directory**

This directory measures the model’s ability to differentiate and understand activities in a scene between similar actions. For each phase, the directory contains:

- **Experiment Notebook:** Used to conduct the experiment.
- **Results Notebook:** Used to aggregate and plot the results.

## Phases

The repository is structured into three phases:
- **phaze1**
- **phaze2**
- **phaze3**

Each phase represents a distinct set of images used in the experiments.

## How to Use

1. Clone this repository to your local machine.
2. Download and unzip the dataset by running the following commands:
    ```bash
    cd GABInsight
    gdown https://drive.google.com/uc?id=13qeOuszF52b8F7Bkvxg5GEEHEl_7obzM
    unzip phazes.zip
    ```
3. Navigate to the directory of the experiment you wish to explore.
4. Open the corresponding experiment notebook to run the experiment.
5. Use the results notebook to aggregate and visualize the results.

For detailed information on each experiment, refer to the main paper associated with this repository.
