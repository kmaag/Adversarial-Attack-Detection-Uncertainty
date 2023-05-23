## Uncertainty-based Detection of Adversarial Attacks in Semantic Segmentation

For a detailed description, please refer to https://arxiv.org/abs/2305.12825.

### Packages and their versions:
Code tested with ```Python 3.9.12``` and ```pip 22.3.1```.
Install Python packages via
```sh
pip install -r requirements.txt
```

### Preparation:
The re-implementations of the fast gradient sign method (FGSM), the stationary segmentation mask method (SSMM) and the dynamic nearest neighbor method (DNNM) can be found in `attacks.py` which we run for two models provided in https://github.com/LikeLy-Journey/SegmenTron. For running the patch attack, we consider the original framework https://github.com/retis-ai/SemSegAdvPatch/.

For the evaluation, edit all necessary paths stored in `global_defs.py`. The outputs will be saved in `./outputs` by default. 
Select the tasks to be executed by setting the corresponding boolean variable (True/False). These functions are CPU based and parts are parallized over the number of input images, adjust `NUM_CORES` to make use of this. 

### Run the code:
```python
python main_functions.py
```

## Author:
Kira Maag, kira.maag@rub.de

