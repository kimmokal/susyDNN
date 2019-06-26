#!/bin/bash

# Path to the working directory
WORK_DIR=/work/kimmokal/susyDNN

# Paths to the output directory and the directory containing the friend trees
DATA_DIR=$WORK_DIR/friend_trees
OUT_DIR=$WORK_DIR/preprocessedData/skimmed

# Determine the preselection cuts
CUTS="(HT>=500)&&(LT>250)&&(nLep==1)&&(Lep_pt>25)&&(nVeto==0)&&(nJets30Clean>=5)&&(Jet2_pt>80)&&(nBJet>=1)&&(dPhi>0.5)"

# Choose the wanted input features for the DNN training
INPUT_FEATURES="LT,HT,Jet1_pt,Jet2_pt,MET,nJets30Clean,nBCleaned_TOTAL,nTop_Total_Combined,nResolvedTop,dPhi"

# Loop over all the friend trees in DATA_DIR
for FILE in $DATA_DIR/*
do
    # Remove the directory path and file extension from the file name
    FNAME=${FILE%.*}
    FNAME=${FNAME#$DATA_DIR/}

    # Add suffix to the output file name
    OUT_NAME="${FNAME}_skimmed.root"

    # Skim the ROOT tree and only include the features necessary for the DNN training
    rooteventselector -s $CUTS -e "*" -i "$INPUT_FEATURES" $FILE:sf/t $OUT_DIR/$OUT_NAME

    echo "Processed: $FNAME"
done

echo "All done!"
