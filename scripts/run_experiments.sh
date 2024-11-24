#!/bin/bash

#############################
# RUN ALL TASK COMBINATIONS #
#############################

# For each model...
for model in citenet deepwalk gcn
do

    # For each dataset...
    for dataset in citeseer cora pubmed
    do

        # Run task
        python -m main run-task $model $dataset

    done

done