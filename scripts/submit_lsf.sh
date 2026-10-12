#!/bin/bash
# Example LSF job script (cluster-specific: adapt the queue name and resources to your site).
# Submit with:  bsub < scripts/submit_lsf.sh      (from the repository root)
#BSUB -J temporal_encoding
#BSUB -q cceb_normal                       # replace with your queue
#BSUB -n 1
#BSUB -R "rusage[mem=2000]"                # MB per core
#BSUB -o output.temporal_encoding.%J.log
#BSUB -e error.temporal_encoding.%J.err

python run_experiments.py --experiments MAR_noncausal --n-runs 30
