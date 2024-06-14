# How to use the cluster

## Login
Activate the University of Osnabrueck's VPN and make sure no other VPNs are active, then run:
```bash
ssh <username>@hpc3.rz.uos.de
```
If no error occurs, this will prompt you to enter your password. You are not going to see the input, so just enter it and hit `ENTER`.

## Interactive session

To allocate a compute node and get a shell prompt run:

```bash
srun --pty /bin/bash
```

To end the session type
```bash
exit
```
and hit `ENTER`.


## Slurm

For detailed information on a slurm command you can run
```bash
man <command_name>
```

### Submit a job

To submit a job using `slurm`, create a script specifying the job. Here's an example. Replace `job_name`, `HH:MM:SS`, `XXgb`, `X`, `partition_name`, `/path/to/miniconda3`, `environment_name`, `/path/to/project/directory`, `script.py`, and the arguments with the appropriate values for your job.
```sh
#!/bin/bash
#SBATCH -J job_name
#SBATCH --time=HH:MM:SS
#SBATCH --mem=XXgb
#SBATCH --gpus=X
#SBATCH --ntasks=X
#SBATCH --cpus-per-task=X
#SBATCH --partition=partition_name
#SBATCH --gpu-bind=single:1

source $HOME/.bashrc

# Activate the emergab environment using conda
spack load miniconda3
conda activate <env_name>

# Navigate to the emergent-abstractions directory
cd "$HOME/emergent-abstractions/"

# Run the Python script using srun 
srun python -m egg.nest.nest_local --game train --sweep grid_search/parameters.json --n_workers=25 --root_dir "grid_search/02-04b" --name "emergent-abstractions"

# Make sure the system has time to save all the models and stuff
srun sleep 10
```
Alternatively, if you encounter errors regarding the activation of your environment, provide the env path. 
```sh
#!/bin/bash

#SBATCH -J standard-grid-search
#SBATCH --time=35:00:00
#SBATCH --mem=400gb
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=gpu
#SBATCH --gpu-bind=single:1


# Source the .bashrc file
source $HOME/.bashrc

# Activate the emergab environment using conda
spack load miniconda3
conda activate <env_name>

# Navigate to the emergent-abstractions directory
cd "$HOME/emergent-abstractions/"

# Add the path to YOUR environment before command line to run
srun /home/student/r/rverdugo/miniconda3/envs/<env_name>/bin/python -m egg.nest.nest_local --game train --sweep grid_search/parameters.json --n_workers=29 --root_dir "grid_search/03-04b" --name "emergent-abstractions"

# Make sure the system has time to save all the models and stuff
srun sleep 10
```

With this script, to schedule your job, run:
```bash
sbatch <script_file_path>
```
For example:
```bash
sbatch gridsearch.sh
```

This will return a reference to your job:

```bash
Submitted batch job 6012345
```

The number at the end (6012345) is the `JOBID` assigned by Slurm.

### View job queue
```bash
squeue
```

This command will display the list of queued and running jobs. The output of squeue includes the following columns:

- JOBID: The unique identifier assigned to each job.
- PARTITION: The partition or queue to which the job is assigned.
- NAME: The name of the job.
- USER: The username of the user who submitted the job.
- ST: The state of the job. Common states include:

  - PD: Pending (waiting for resources)
  - R: Running
  - CG: Completing
  - CD: Completed
- TIME: The time the job has been running or waiting in the queue.
- NODES: The number of nodes allocated to the job.
- NODELIST(REASON): The list of nodes assigned to the job or the reason why the job is waiting (e.g., Resources, Priority).

#### filter squeue output
The queue can be pretty long. To get a more concise overview over your jobs you can filter the output of squeue based on
- job states: use the -t or --states option followed by the desired state(s)
- user name: use the -u or --user option followed by your username

Here's a few examples

##### view only your jobs
```bash
squeue -u <your_username>
```

##### view only your running jobs
```bash
squeue -u <your_username> -t R
```

##### view only your pending and running jobs
```bash
squeue -u your_username -t PD,R
```

By default, `squeue` provides a snapshot of the job queue at the moment you run the command. To get real-time updates, you can use the `-i` or `--iterate` option followed by the update interval in seconds.

For example, to update the queue information every 5 seconds
When the job runs, a file named `slurm-<jobid>.out` will be created in the directory you started the job from. This file contains the job's output that you would usually see printed to the terminal. You can look at the content by running:

```bash
squeue -u your_username -i 5
```

This command will continuously update the output every 5 seconds until you stop it with `Ctrl+C`.

```bash
cat slurm-<jobid>.out
```
Replace `<jobid>` with your actual `JOBID` assigned by Slurm.

### Cancel a job
```bash
scancel <jobid>
```

### Transfer a folder from the cluster to your local machine
 
```bash
scp -r <username>@hpc3.rz.uos.de:/home/student/r/<username>/some_path/some_folder <local_target_folder>
```

Example (copy to current folder on local machine):

Navigate to the folder in your local machine you wish to store the file in. Right click and open a terminal within that folder.
```bash
scp -r user123@hpc3.rz.uos.de:/home/student/r/123/some_path/some_folder .
```

## Dealing with conda environments

### List existing environments

This command will list all the discoverable Conda environments.

```bash
conda info --envs
```

If the desired environment is not listed, you may need to modify the PATH to include the Conda installation directory. Run the following command, replacing `/path/to/miniconda3` with the actual path to your Miniconda installation

### Set correct environment path
```bash
export PATH="/path/to/miniconda3/bin:$PATH"
```

### Veryfiy environment
You can veryfiy that the correct conda environment is running by running:

```bash
which conda
```

It should display the path to the Conda executable in your Miniconda installation directory.

### Source the Conda initialization script 

Source the Conda initialization script by running
```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
```
Replace `/path/to/miniconda3` with the actual path to your Miniconda installation obtained by `conda info --envs`. 

### Activate conda environment

```bash
conda activate <environment_name>
```

Verify that the environment is activated by checking the command prompt. It should display the name of the active environment in parentheses, like this:

```bash
(environment_name) user@cluster:~$
```

## Issues and Troubleshooting

### Conda command not found

If you encounter the error message `conda: command not found`, it means that the Conda executable is not in your system's `PATH`. To resolve this issue:

1. Locate the Conda installation directory on your cluster. It is usually in your home directory or a shared directory.
2. Add the Conda installation directory to your `PATH` by running
   ```bash
   export PATH="/path/to/miniconda3/bin:$PATH"
   ```
3. Verify that the Conda command is now accessible
    ```bash
    which conda
    ```

### Conda environment not found

If you receive the error message "Could not find conda environment: environment_name", it indicates that the specified Conda environment does not exist or is not discoverable. To resolve this issue:
1. Check the list of available Conda environments
   ```bash
   conda info --envs
   ```
2. If the desired environment is not listed, make sure you have activated the correct Conda installation and sourced the initialization script
    ```bash
    export PATH="/path/to/miniconda3/bin:$PATH"
    source /path/to/miniconda3/etc/profile.d/conda.sh
    ```


### Module not found error
If you encounter the error message "ModuleNotFoundError: No module named 'your_module'," it means that the specified Python module is not installed in the active Conda environment. To resolve this issue:
1. activate your conda environment
2. install the missing module (using conda or pip)
    ```bash
    conda install <module_name>
    ```
    or
    ```bash
    pip install <module_name>
    ```
3. verify that the module is installed by running:
    ```bash
    python -c "import <module_name>"
    ```
    Replace `<module_name>` by your concrete module's name.
    If this passes without an error, the module is correctly installed.

### Example: egg not found
You might also see the following error:
```bash
Error while finding module specification for 'egg.nest.nest_local' (ModuleNotFoundError: No module named 'egg')
srun: error: hpc3-41: task 0: Exited with exit code 1
```

1. activate `emergab` environment:
   ```bash
   conda activate emergab
   ```
2. start a interactive python shell
   ```bash
   python
   ```
3. find the location of the module
   ```python
   import egg
   print(egg.__file__)
   ```
   this should output something like this:
   ```bash
   /home/student/<letter>/<username>/miniconda3/envs/emergab/lib/python3.9/site-packages/egg/__init__.py
   ```
   exit the interactive pyton shell:

   ```python
   exit()
   ```
4. The directory containing the `egg` module is the path up to the egg folder. In this example, it would be:
   ```bash
   /home/student/<letter>/<username>/miniconda3/envs/emergab/lib/python3.9/site-packages/
   ```
   Use this path to set the `PYTHONPATH` environment variable in the script that is used for the job:
   ```sh
   export PYTHONPATH="/home/student/r/rverdugo/miniconda3/envs/emergab/lib/python3.9/site-packages:$PYTHONPATH"
   ```
5. To use the absolute path for the egg module in your script, replace the srun command with
   ```sh
   srun python /home/student/<letter>/<username>/miniconda3/envs/emergab/lib/python3.9/site-packages/egg/nest/nest_local.py <rest of the command>
   ```

### installation of egg via pip fails
```bash
(emergab2) [rverdugo@mgmt01 emergent-abstractions]$ pip install git+https://github.com/facebookresearch/EGG.git
Collecting git+https://github.com/facebookresearch/EGG.git
  Cloning https://github.com/facebookresearch/EGG.git to /tmp/pip-req-build-rg__c10k
  Running command git clone --filter=blob:none --quiet https://github.com/facebookresearch/EGG.git /tmp/pip-req-build-rg__c10k
  /usr/libexec/git-core/git-remote-https: symbol lookup error: /lib64/libk5crypto.so.3: undefined symbol: EVP_KDF_ctrl, version OPENSSL_1_1_1b
  error: subprocess-exited-with-error

  × git clone --filter=blob:none --quiet https://github.com/facebookresearch/EGG.git /tmp/pip-req-build-rg__c10k did not run successfully.
  │ exit code: 128
  ╰─> See above for output.

  note: This error originates from a subprocess, and is likely not a problem with pip.
error: subprocess-exited-with-error

× git clone --filter=blob:none --quiet https://github.com/facebookresearch/EGG.git /tmp/pip-req-build-rg__c10k did not run successfully.
│ exit code: 128
╰─> See above for output.

note: This error originates from a subprocess, and is likely not a problem with pip.
```

Download the [egg repo from github](https://github.com/facebookresearch/EGG)  and navigate to the EGG folder. Activate the desired (conda) environment you would like to install egg to, then run:

```bash
pip install .
```

To check that `egg` was installed correctly as a module, run:

```
python -c "import egg"
```

If this executes without an error, it means the `egg` was installed succesfully.

### Issue with loading the PyTorch C
```bash
Traceback (most recent call last):
  File "/home/student/r/rverdugo/miniconda3/envs/emergab/lib/python3.9/site-packages/egg/nest/nest_local.py", line 13, in <module>
    from egg.nest.wrappers import ConcurrentWrapper
  File "/home/student/r/rverdugo/miniconda3/envs/emergab/lib/python3.9/site-packages/egg/nest/wrappers.py", line 10, in <module>
    import torch
  File "/home/student/r/rverdugo/miniconda3/envs/emergab/lib/python3.9/site-packages/torch/__init__.py", line 451, in <module>
    raise ImportError(textwrap.dedent('''
ImportError: Failed to load PyTorch C extensions:
    It appears that PyTorch has loaded the `torch/_C` folder
    of the PyTorch repository rather than the C extensions which
    are expected in the `torch._C` namespace. This can occur when
    using the `install` workflow. e.g.
        $ python setup.py install && python -c "import torch"

    This error can generally be solved using the `develop` workflow
        $ python setup.py develop && python -c "import torch"  # This should succeed
    or by running Python from a different directory.
srun: error: hpc3-4: task 0: Exited with exit code 1
```
This error indicates a problem with the torch installation. To fix it, we will uninstall torch and re-install it using the `conda` package manager instead of pip.

1. uninstalling
```bash
conda activate emergab
pip uninstall torch
```
2. reinstall
```bash
conda install pytorch torchvision -c pytorch
```
3. verify correct installation
```
python -c "import torch; print(torch.__version__)"
```
This should print the version of pytorch, for example:
```bash
2.2.2
```

