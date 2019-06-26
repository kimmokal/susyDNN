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

_Note: I renamed_ evVarFriend_T1tttt_MiniAOD_19_01_v2.root _to_ evVarFriend_T1tttt_MiniAOD_19_01.root _and removed_ FRIEND_TOTAL_SIGNAL.root

</p>
</details>

### 2. Preselection

The event preselection is done using `skim_friend_trees.sh`, which can be made executable with `chmod`.
```
$ chmod +x skim_friend_trees.sh
```
Then run the bash script. **(Note: The working directory is hardcoded in the script, so you will need to manually modify the correct path in the file.)** It utilizes the _rooteventselector_ feature of ROOT 6.XX for quick and efficient event preselection and skimming of the friend trees. The preselection cuts and the selected input features for the DNN training are determined in the sript.
```
$ ./skim_friend_trees.sh
```
After the script has finished running, the skimmed friend trees can be found in the _preprocessedData/skimmed/_ directory.
