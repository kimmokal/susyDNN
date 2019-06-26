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

### 1. Symbolic link to the friend trees

First create a symbolic link in the working directory to the directory that contains the friend trees:
```
$ ln -s /path/to/the/friend/trees friend_trees
```
Here is the (collapsible) list of the friend trees I currently have in my _friend_trees_ directory:
<details><summary>FRIEND TREES</summary>
<p>
  
```
evVarFriend_T1tttt_MiniAOD_15_10.root
evVarFriend_T1tttt_MiniAOD_19_01.root
evVarFriend_T1tttt_MiniAOD_19_08.root
evVarFriend_T1tttt_MiniAOD_19_10.root
evVarFriend_T1tttt_MiniAOD_22_01.root
evVarFriend_T1tttt_MiniAOD_22_08.root
evVarFriend_DYJetsToLL_M50_HT400to600.root
evVarFriend_DYJetsToLL_M50_HT600to800.root
evVarFriend_DYJetsToLL_M50_HT800to1200.root
evVarFriend_DYJetsToLL_M50_HT1200to2500.root
evVarFriend_DYJetsToLL_M50_HT2500toInf.root
evVarFriend_QCD_HT500to700.root
evVarFriend_QCD_HT700to1000.root
evVarFriend_QCD_HT1000to1500.root
evVarFriend_QCD_HT1500to2000.root
evVarFriend_QCD_HT2000toInf.root
evVarFriend_TBar_tch_powheg.root
evVarFriend_TBar_tWch_ext.root
evVarFriend_T_tch_powheg.root
evVarFriend_TTJets_DiLepton.root
evVarFriend_TTJets_LO_HT600to800_ext.root
evVarFriend_TTJets_LO_HT800to1200_ext.root
evVarFriend_TTJets_LO_HT1200to2500_ext.root
evVarFriend_TTJets_LO_HT2500toInf_ext.root
evVarFriend_TTJets_SingleLeptonFromTbar.root
evVarFriend_TTJets_SingleLeptonFromT.root
evVarFriend_TToLeptons_sch_amcatnlo.root
evVarFriend_T_tWch_ext.root
evVarFriend_WJetsToLNu_HT400to600.root
evVarFriend_WJetsToLNu_HT600to800.root
evVarFriend_WJetsToLNu_HT800to1200.root
evVarFriend_WJetsToLNu_HT1200to2500.root
evVarFriend_WJetsToLNu_HT2500toInf.root
```

**Note:** I renamed _evVarFriend_T1tttt_MiniAOD_19_01_v2.root_ to _evVarFriend_T1tttt_MiniAOD_19_01.root_ and removed _FRIEND_TOTAL_SIGNAL.root_

</p>
</details>

### 2. Preselection

The event preselection is done using `skim_friend_trees.sh`, which can be made executable with `chmod`.
```
$ chmod +x skim_friend_trees.sh
```
Then run the bash script. (**Note:** The working directory is hardcoded in the script, so you will need to manually modify the correct path in the file.) It utilizes the _rooteventselector_ feature of ROOT 6.XX for quick and efficient event preselection and skimming of the friend trees. The preselection cuts and the selected input features for the DNN training are determined in the sript.
```
$ ./skim_friend_trees.sh
```
After the script has finished running, the skimmed friend trees can be found in the _preprocessedData/skimmed/_ directory.


### 3. Preprocessing

The preprocessing script normalizes the chosen input features (subtracts the mean and divides by the standard deviation) and creates train and test sets for each mass point in the _preprocessedData/_ directory. The script also saves the StandardScaler to a file as it is needed if the same normalization is to be performed later. For safety reasons, the full list of the training set's variables and the normalized variables are also saved to file.

**Note:** the preprocessing script only considers the background samples listed in _background_file_list.txt_ and you will need to manually fix the file paths in the text file.

**Second note:** the script also adds _sampleName_ column to the train/test sets. It is done by mapping each sample name to an integer. This is encoding/decoding method is implemented very clumsily in the _utilities.py_ file. Make sure it is updated to encompass all the signal/background samples you use.
```
$ python preprocess_massPoints.py
```
Each train/test set only contains one signal sample for a specific mGo/mLSP mass point.

### 4. Neural network training

Train a DNN for each mass point. The trained models are saved to the _models/_ directory. The script assumes by default that a GPU is used for the training. If only CPU is available, some changes to the script may need to be done.
```
$ python DNN_train_massPoints.py
```

### 5. Plot the results

You can now plot the ROC curve and the DNN output distribution for each mass point. This script will save them to the _plots/_ directory in .png and .pdf format.
```
$ python roc_output_plotter.py
```
**Note:** this script is suboptimal, as it needs to be executed once for each mass point (i.e. six times). You need to change the mass point every time inside the script. It is currently done on the line 31.

### 6. Write the DNN output to the original friend trees

Now it is possible to add the DNN predictions to the original friend trees. The following script is far from optimal and has not been validated properly. You will need to create an output directory and change the path in the script. Once again the mass point needs to be chosen and it can be changed inside the script on line 29.
```
$ python write_to_tree.py
```
It would make sense to include the input and output directories as command line options in the future. Also the script would definitely benefit from parallelization. Now by default it processes all the friend trees contained in the input directory and it can be very slow.
