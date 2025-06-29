# Checkpoint/Restore with DMTCP
DMTCP (Distributed MultiThreaded Checkpointing) is a tool to transparently checkpoint and restore single host or distributed computation in user-space. The checkpoint/restore mechanism is implemented on a library lever (semi-transparent), meaning one can only dump applications known to run successfully with the DMTCP libraries, but the latter does not provide proxies for all kernel APIs. Additionally, application performances may be degraded due to the overhead introduced by the checkpoint/restore mechanism.


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
### 2. Clone this repository (to do)

### 3. setup (to do)
 

## Usage (adapt to dmtcp)
The `checkpoint.sh` script handles the execution of checkpoint and restore operations with CRIU in a simplified manner. Before executing any of the below commands, source the `bashrc` file with

```
source bashrc
```

### Starting a program
To run a program

```
start <additional arguments> fileName.py <python arguments>
```
where the additional arguments can be:
- ```-id``` : tag to identify different process executing the same python file.
- ```-time```: specify time in seconds after which the program will stop and checkpoint its state.
- ```-periodic```: specify time in seconds after which the program will checkpoint its state peridiocally, continuing its execution.

If not present, the output directory will be created, containing for each process an output_fileName_id.log file that logs the output stream.

If not present, the checkpoints directory will be created, containing for each process a checkpoint_fileName_id directory containing the checkpoint snapshots.



### Checkpointing a program
To checkpoint a program 

```
stop -id <id> fileName.py 
```
```-id``` must be included if previously specified in the start command.


### Resuming a program
To resume a checkpointed program

```
resume -id <id> fileName.py 
```
```-id``` must be included if previously specified in the start command. Additionally, the ```-periodic``` and the ```-time``` flags can be used equivalently as in the start command.




## Project structure

## Limitations
