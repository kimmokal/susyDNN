# susyDNN

Repository for training signal/background DNN classifier. Required software: Python 2.7 and ROOT 6.XX.

First clone the repository.
```
$ git clone https://github.com/kimmokal/susyDNN
$ cd susyDNN/
```

Create a virtual environment in which to install the required Python packages.
```
$ virtualenv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Workflow

### 1. Preselection

First create a symbolic link in the working directory to the directory that contains the friend trees:
```
$ ln -s /path/to/the/friend/trees friend_trees
```
The event preselection is done using `skim_friend_trees.sh`, which can be made executable with `chmod`.
```
$ chmod +x skim_friend_trees.sh
```
Then run the bash script. **(Note: The working directory is hardcoded in the script, so you will need to manually modify the correct path in the file.)** It utilizes the _rooteventselector_ feature of ROOT 6.XX for quick and efficient event preselection and skimming of the friend trees. The preselection cuts and the selected input features for the DNN training are determined in the sript.
```
$ ./skim_friend_trees.sh
```
After the script has finished running, the skimmed friend trees can be found in the _preprocessedData/skimmed/_ directory.
