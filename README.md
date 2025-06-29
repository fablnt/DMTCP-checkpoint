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
### 2. Clone this repository (to do)
Clone this repository:
```bash
git clone https://github.com/fablnt/DMTCP-checkpoint.git
```
### 3. setup (to do)
 Absolute path to dmtcp

## Usage (adapt to dmtcp)
The `checkpoint.sh` script handles the execution of checkpoint and restore operations with DMTCP in a simplified manner. Before executing any of the below commands, source the `bashrc` file with

```
source bashrc
```
Notice that, for each program launched with the `checkpoint.sh` script, a separated coordinator process will be started and attached to the program. 

### Starting a program
To run a program

```
start <additional arguments> fileName.py <python arguments>
```
where the additional arguments can be:
- ```-id``` : tag to identify different process executing the same python file.
- ```-i```: specify time in seconds after which the program will checkpoint its state.

After checkpointing, the program will continue running and checkpointing again after the time specified with ```-i```.

When a program is launched with ```start``` for the first time, a directory will be created containing the following files:
- application.log
- execution.log
- coordinator.log

If a program is launched with ```start```, but a directory with that name already exists, the program will not start. 


### Resuming a program
To resume a checkpointed program

```
resume -id <id> fileName.py 
```
```-id``` must be included if previously specified in the start command. Additionally, the ```-i``` flags can be used equivalently as in the start command.




## Project structure


## Limitations
