# Cross-modal Few-shot Learning Implementation

<div align="center">
    <img src="https://github.com/776styjsu/CrossModal-LLM-Enhanced-Learning/assets/50536905/c85b7838-a230-4643-9ab2-f6aa0faafddb">
</div>



## Overview
This repository provides a naive implementation of the cross-modal few-shot learning approach presented in this paper https://arxiv.org/pdf/2301.06267.pdf. The primary goal of this paper is to improve few-shot learning. In this implementation, we demonstrated that the technique also enhances the classifier's accuracy in a non few-shot settings while harnessing the significant boost of inference speed compare to methods that don't leverage the power of pre-trained embedding models such as CLIP.

## Key Features
Multimodal Foundation: Utilizes models like CLIP that map different modalities to the same representation space. Like the paper, this implementation supports the embedding of three modalities: visual, textual, and audio.
Cross-modal Adaptation: A simple approach that learns from the embedded examples across different modalities using shallow classifiers.
Enhanced Classifier: Our experiments (though many aspects of the design should be made more rigorous and improved) showed superiority in accuracy using an image-text classifier compare to a fresh ResNet50 (image only) and a further-trained top-ranked classifier by https://www.kaggle.com/vlomme (audio only) on a curated subset of Cornell Birdcall Identification dataset. 

## Future Work
LLM Prompting: We have considered using LLM prompting techniques to generate textual data that aligns with how CLIP are trained for further experiments to see whether this can be an effective model-improving strategy.
Segmentation Techniques: Segmentation techniques like SAM have also been thought of to improve the quality and robustness of image data but haven't been implemented yet.

## Getting Started

### Setting up Dependencies

To set up the dependencies for the project, please follow the instructions below:

1. Open your command prompt or terminal.

2. Navigate to the project directory.

3. Run the command `conda env create -f environment.yml`. This will create a new conda environment with the name specified in the `environment.yml` file, and install all the necessary dependencies.

4. Once the installation is complete, activate the environment by running the command `conda activate <environment-name>`. Replace `<environment-name>` with the name of the environment created in step 3.

5. You're now ready to use the project with all the required dependencies installed.

**Note:** Make sure you have [conda](https://docs.conda.io/en/latest/miniconda.html) installed on your system before following the above steps.
