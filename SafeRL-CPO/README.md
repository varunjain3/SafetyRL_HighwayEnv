# This CPO implementation on Highway-env is built on the intial work of [https://github.com/dobro12/CPO]. 

## How to run the experiments!
- Firstly please make sure that the environment is made using the `environment.yml` file.
- ``` conda env create -f environment.yml ```
- Then use the CPO conda environment that will have all the required functionalities installed.
- Check the ```./highwayenv/highway_env/envs/highway_env.py``` path for seeing the custom implementation of Cost function to incorporate safety.
## Training
- ```python train.py```
## Testing
- ```python train.py test```
- Please note that this would require a display device. This wouldn't work on remote servers without display.