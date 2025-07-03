# Checkpoint/Restore with DMTCP
DMTCP (Distributed MultiThreaded Checkpointing) is a tool to transparently checkpoint and restore single host or distributed computation in user-space. The checkpoint/restore mechanism is implemented on a library lever (semi-transparent), meaning one can only dump applications known to run successfully with the DMTCP libraries, but the latter does not provide proxies for all kernel APIs. Additionally, application performance may be affected due to the overhead introduced by the checkpoint and restore mechanism.


## Installation
Follow the steps below to install and use DMTCP:

### 1. Clone and Build DMTCP

Clone the DMTCP repository from [https://github.com/dmtcp/dmtcp](https://github.com/dmtcp/dmtcp) and build it:

```bash
git clone https://github.com/dmtcp/dmtcp.git
cd dmtcp
./configure
make
# No need to run 'make install'
```
### 2. Clone this repository 
Clone this repository:
```bash
git clone https://github.com/fablnt/DMTCP-checkpoint.git
```
### 3. Script setup 
Open the file ```checkpoint.sh``` and set the variable ```DMTCP_EXEC``` to the absolute path of the dmtcp executables, e.g.
```bash
DMTCP_EXEC="/leonardo/home/userexternal/$USER/dmtcp/bin/
```

It may be necessary to make ```checkpoint.sh``` executable with 
 ```bash
chmod +x /path_to_file/checkpoint.sh
```

## Usage 
The `checkpoint.sh` script handles the execution of checkpoint and restore operations with DMTCP in a simplified manner, printing output both in the I/O stream and in log files. 


Notice that, for each program launched with the `checkpoint.sh` script, a separated coordinator process will be started and attached to the program. 

### Starting a program
To run a program

```
./checkpoint.sh start <additional arguments> fileName.py <python arguments>
```
where the additional arguments can be:
- ```-id``` : tag to identify different process executing the same python file.
- ```-i```: specify time in seconds after which the program will checkpoint its state (default is 20s).


> [!NOTE]
> The coordinator port is set automatically.

When a program is launched with ```start``` for the first time, a directory named ```id_scriptName_pythonArguments``` will be created containing the following files:
- ```application.log```: contains the program output.
- ```execution.log```: contains various information about the program and its execution.
- ```coordinator.log```: contains the coordaintor output.
- ```dmtcp.config```: contains the DMTCP configuration set.

If a program is launched with ```start```, but a directory with that name already exists, the program will not start. (We assume that a checkpoint already exists, so restart should be used instead).


### Resuming a program
To resume a checkpointed program

```
./checkpoint.sh restart -id <id> -i <time> fileName.py <python arguments>
```
The resumed program will checkpoint its state after ```<time>``` seconds, as it does in the start command.


```-id``` must be included if previously specified in the start command. Additionally, the ```-i``` flags can be used equivalently as in the start command.
Please note that the python arguments must be the same used in the start command used to run the program for the first time. 

>[!IMPORTANT]
> The program execution is interrupted after a checkpoint is being performed. This option can be disabled by removing the flag ```--kill-after-ckpt``` in the ```checkpoint.sh``` at lines
>[65](https://github.com/fablnt/DMTCP-checkpoint/blob/fd760b676d6f5ab8be89e17e41604920424e7aaf/checkpoint.sh#L65) and [130](https://github.com/fablnt/DMTCP-checkpoint/blob/fd760b676d6f5ab8be89e17e41604920424e7aaf/checkpoint.sh#L130)


## Project structure
This repository contains:

- ```checkpoint.sh```: the checkpoint/restore script.
- ```src```: contains the files used to run the tests, with additional debugging information.
- ```results```: contains the results of the previous tests performed on the Leonardo cluster of CINECA. Additional information can be found in [results/README.md](https://github.com/fablnt/DMTCP-checkpoint/blob/master/results/README.md)


## Limitations
The DMTCP tool presents some limitations:

1) The tool does not checkpoint applications and libraries that work with GPUs (e.g. torch).
2) Some libraries conflicts with dmtcp, resulting in a stalling in the program without reporting any particular error.
4) The DMTCP developers suggest to use the [MANA](https://mana-doc.readthedocs.io/en/latest/) plugin to handle MPI workloads, but we did not manage to install it on Leonardo. 
