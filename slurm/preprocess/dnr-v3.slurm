#!/bin/bash
#SBATCH --array=1-12
#SBATCH -Jdnr-download
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=8 --mem-per-cpu=16G
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

# Conditional Split Definition
if [ "$TASK_ID" -le 6 ]; then
    # Train set languages (indices 0-6: multi to cmn)
    # This group requires all four splits.
    # SPLITS=("audio-train" "audio-val" "audio-test" "metadata")
    SPLITS=("audio-metadata")
    echo "This is a TRAIN language. Splits: ${SPLITS[@]}"
elif [ "$TASK_ID" -ge 7 ] && [ "$TASK_ID" -le 12 ]; then
    # Test-only languages (indices 7-12: eus to yor)
    # This group only requires test and metadata.
    # SPLITS=("audio-test" "metadata")
    SPLITS=("audio-metadata")
    echo "This is a TEST-ONLY language. Splits: ${SPLITS[@]}"
else
    # This block should not be reached due to the initial bounds check and --array=0-12
    echo "FATAL ERROR in split logic for TASK_ID ($TASK_ID). Exiting."
    exit 1
fi

# --- Download Loop with Curl Continuation ---
for SPLIT in "${SPLITS[@]}"; do
    
    URL="https://nflx-divideandremasterv3-public-dmz-prod-us-west-2.s3.us-west-2.amazonaws.com/dnr-v3-compressed/${LANG_CODE}/dnr-v3-${LANG_CODE}-${SPLIT}.tar.gz"
    OUTPUT_FILE="${LANG_CODE}-${SPLIT}.tar.gz"
    
    echo "Downloading split: $SPLIT from URL: $URL"
    
    # curl command with continuation (-C -), silent mode (-s), showing progress (-#)
    curl -C - -s -# "$URL" -o "$OUTPUT_FILE"
    
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded and saved: $OUTPUT_FILE"
    else
        echo "ERROR: Curl failed to download $URL for $LANG_CODE/$SPLIT. Status code: $?"
        # Optionally add a command to clean up the partial file if download failed
        # rm -f "$OUTPUT_FILE"
    fi
    echo "---"
done

echo "Task $TASK_ID completed for language $LANG_CODE."