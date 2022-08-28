
# Steps on how to connect to Cori on NERSC and submit jobs (CPU).
* The first thing you need to do is go to your terminal and login to cori via ssh, by typing the following.

        ssh yourusername@cori.nersc.gov
        password: PASSWORD + OTP

<p align="center">
<img src="https://github.com/bashirmohd/my-daphne/blob/main/Documentation_scripts_master/running_jobs_nersc/figures/login.png" width="90%" height="90%" title="login">
<p>

<p align="center">
<img src="https://github.com/bashirmohd/my-daphne/blob/main/Documentation_scripts_master/running_jobs_nersc/figures/password.png" width="90%" height="90%" title="password">
<p>

* To obtain your remote directory path, type the following on Cori.

        pwd

<p align="center">
<img src="https://github.com/bashirmohd/my-daphne/blob/main/Documentation_scripts_master/running_jobs_nersc/figures/path.png" width="90%" height="90%" title="path">
<p>
        
* To Transfer a 'folder/file' from your local directory(Your Computer) to your remote directory on Cori, run the following:

        scp -r foldername  mohammed@cori.nersc.gov://global/homes/m/mohammed/projects
        
* To check the available modules on cori type the following: and copy the modules required to run your script.

        module avail
 
* NERSC uses Slurm for cluster and resource management as well as job scheduling. Slurm is responsible for allocating resources to users, providing a framework for starting, executing and monitoring work on allocated resources and scheduling work for future execution. For more details about Slurm please click [here](https://slurm.schedmd.com/).

* sbatch is used to submit a job script for later execution. To create a batch script called "test.sh" using the vim editor, type in the following on your Cori node: 

        vim test.sh
* Copy the follwoing below and paste in the test.sh file. ( for qos, I selected premium, for time I selected 1hr, for nodes I selected 1 node, for constraint I selected haswell. I want to be alerted when my job Begins, Ends or Fail). Please note I have used the module load command to load the modules required to run my python script. The last line will run my python scprit (test.py). I have added an Optional line which will save all errors in (myfile.err) and all non-errors will be save in (myfile.out)
     

        #!/bin/bash
        #SBATCH --qos=premium
        #SBATCH -t 01:00:00
        #SBATCH --nodes=1
        #SBATCH --exclusive
        #SBATCH --constraint=haswell
        #SBATCH --mail-type=begin,end,fail
        #SBATCH --mail-user=bmohammed@lbl.gov
        module load python/3.7-anaconda-2019.10
        module load tensorflow/intel-1.15.0-py37
        python test.py 1>> myfile.out 2>> myfile.err
      
     For more details on other examples of job scripts please click [here](https://docs.nersc.gov/jobs/examples/).
      
       For Vim: Press i to insert, Press :wq  or  :x to Save and close the file.

     For more details on some useful vim command  click [here](https://docs.nersc.gov/jobs/examples/) 


 * To submit a job, run the following:
 
        sbatch test.sh
    When you submit the job, Slurm responds with the job's ID, which will be used to identify this job in reports from Slurm.
    
   To see the jobs in the queue associated to a user, run:
        
        squeue | grep mohammed
    
<p align="center">
<img src="https://github.com/bashirmohd/my-daphne/blob/main/Documentation_scripts_master/running_jobs_nersc/figures/sbatch%20.png" width="90%" height="90%" title="sbatch">
<p>
        
<p align="center">
<img src="https://github.com/bashirmohd/my-daphne/blob/main/Documentation_scripts_master/running_jobs_nersc/figures/squeue_job_number.png" width="90%" height="90%" title="squeue">
<p>
 
 
  * To check the status of jobs in the cue using your job ID, run:     
   
        squeue -j 30331136
 
<p align="center">
<img src="https://github.com/bashirmohd/my-daphne/blob/main/Documentation_scripts_master/running_jobs_nersc/figures/salloc.png" width="90%" height="90%" title="salloc">
<p>
   
   
  * sacct is used to report job or job step accounting information about active or completed jobs. To see more details, like CPU allocated and when the jobs starts running or ends, simple run:     
 
        sacct --allocations
   
  * Notice the State has changed from PENDING to RUNNING. This means your job is running.
  
<p align="center">
<img src="https://github.com/bashirmohd/my-daphne/blob/main/Documentation_scripts_master/running_jobs_nersc/figures/salloc_running.png" width="90%" height="90%" title="job_running">
<p>
    

# Steps on how to connect to Cori on NERSC and submit jobs (GPU).

  * Firstly, run the following by loading esslurm, else you will get an error:     
   
        module load esslurm
        
<p align="center">
<img src="https://github.com/bashirmohd/my-daphne/blob/main/Documentation_scripts_master/running_jobs_nersc/figures/module_load_gpu.png" width="90%" height="90%" title="esslurm">
<p>
        
  * Then login to your GPU node by typing the following:   
        
        salloc -C gpu -N 1 -t 01:00:00 -c 10 --gres=gpu:1

<p align="center">
<img src="https://github.com/bashirmohd/my-daphne/blob/main/Documentation_scripts_master/running_jobs_nersc/figures/gpu_login.png" width="90%" height="90%" title="gpu_login">
<p>
    
 
  * To check your GPU specification:   
        
        srun nvidia-smi
        
 <p align="center">
<img src="https://github.com/bashirmohd/my-daphne/blob/main/Documentation_scripts_master/running_jobs_nersc/figures/check_gpu_specs.png" width="90%" height="90%" title="gpu_specs">
<p>
