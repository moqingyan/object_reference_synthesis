# object reference synthesis
Generating Programmatic Referring Expressions via Program Synthesis

### Install
#### Setup the environment
In this project, we used pytorch-geometric and the dependancies of it.

Method 1. Import the environment to your anaconda (Recommended)
In this directory, there is a torch_geometric.yml file. You can import the environment to your anaconda to install all the dependancies. 
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

Method 2. Use the dockerfile 
In this directory, there is a Dockerfile. You can build an docker image here and run the result inside. 
```
docker build .
docker run -it <DOCKER IMAGE NAME>
```
then in the container, you need to activate the conda environment.
```
conda init bash 
exec bash 
conda activate pytorch_geometric
```

Method 3. build from scratch
You can install pytorch geometric according to the documentation here
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html


### Run the scripts
In this directory, there is a run.py which invokes different scripts to execute different tasks. You can tune the parameters in the run.py to change the settings. Further, there are more parameters you can change in the src/cmd_args, and src_prob/cmd_args.