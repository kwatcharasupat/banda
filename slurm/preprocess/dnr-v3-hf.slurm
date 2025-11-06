#!/bin/bash
#SBATCH --array=5
#SBATCH -Jdnr-download
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=24 --mem-per-cpu=4G
#SBATCH --time=16:00:00
#SBATCH --output=./slurm-out/Report-%A/%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kwatchar3@gatech.edu
#SBATCH --signal=SIGTERM@120

cd $SLURM_SUBMIT_DIR

module load mamba
mamba activate banda

# --- Language and Split Definitions ---
# Index 0-6: multi, eng, deu, fao, fra, spa, cmn (Train/Val/Test/Metadata)
# Index 7-12: eus, jpn, inc, dra, bnt, yor (Test/Metadata)

LANG_CODES=(
    "multi" "eng" "deu" "fao" "fra" "spa" "cmn"  "eus" "jpn" "inc" "dra" "bnt" "yor"          
)

# Use the SLURM_ARRAY_TASK_ID (range 0-12)
TASK_ID=$SLURM_ARRAY_TASK_ID

# Check if TASK_ID is within the bounds of the array
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge ${#LANG_CODES[@]} ]; then
    echo "ERROR: Invalid TASK_ID ($TASK_ID). The array index is out of bounds [0-${#LANG_CODES[@]}]."
    exit 1
fi

LANG_CODE=${LANG_CODES[$TASK_ID]}

echo "Current SLURM Array Task ID: $TASK_ID"
echo "Current language code: $LANG_CODE"
echo "--------------------------------------"

hf upload-large-folder "kwatcharasupat/dnr-v3-$LANG_CODE" $LANG_CODE --repo-type=dataset --no-bars --num-workers=32 

echo "Task $TASK_ID completed for language $LANG_CODE."