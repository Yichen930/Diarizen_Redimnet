#!/bin/bash
#PBS -N diarization_redimnet_full
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -l walltime=04:00:00
#PBS -q normal
#PBS -P Personal
#PBS -j oe

cd $PBS_O_WORKDIR

source /home/users/ntu/ygong006/miniconda3/etc/profile.d/conda.sh

conda activate diarizen

cd /home/users/ntu/ygong006/Diarizen-redimnet/

# Configuration
WAV_DIR="/home/users/ntu/ygong006/Diarizen-redimnet/data/third_dihard_challenge_eval/data/wav"
REF_RTTM="/home/users/ntu/ygong006/Diarizen-redimnet/all_reference_combined.rttm"
OUTPUT_DIR="./redimnet_output_combined"
COMBINED_RTTM="$OUTPUT_DIR/all_redimnet_combined.rttm"
RESULTS_DIR="./redimnet_evaluation_combined"

# ReDimNet configuration
MODEL_ID="BUT-FIT/diarizen-wavlm-base-s80-md"
REDIMNET_MODEL="b1"
BATCH_SIZE=8

mkdir -p "$OUTPUT_DIR"
mkdir -p "$RESULTS_DIR"

echo "Starting ReDimNet batch processing at: $(date)"
echo "WAV files: $WAV_DIR"
echo "Reference RTTM: $REF_RTTM"
echo "Output directory: $OUTPUT_DIR"
echo "ReDimNet model: $REDIMNET_MODEL"

# Step 1: Process all WAV files with ReDimNet
echo "=== Step 1: Processing WAV files with ReDimNet ==="

count=0
total=$(ls "$WAV_DIR"/*.wav 2>/dev/null | wc -l)

if [ $total -eq 0 ]; then
    echo "Error: No WAV files found in $WAV_DIR"
    exit 1
fi

echo "Found $total WAV files to process"

for wav_file in "$WAV_DIR"/*.wav; do
    filename=$(basename "$wav_file" .wav)
    output_rttm="$OUTPUT_DIR/${filename}.rttm"
    
    count=$((count+1))
    
    if [ ! -f "$output_rttm" ]; then
        echo "[$count/$total] Processing: $filename"
        
        # Run ReDimNet diarization - 移除不支持的 --model-cache-dir 参数
        python run_redimnet_direct_replace_fixed.py \
            -i "$wav_file" \
            -o "$OUTPUT_DIR" \
            --model-id "$MODEL_ID" \
            --batch-size "$BATCH_SIZE" \
            --redimnet-model "$REDIMNET_MODEL" 2>&1 | tee "$OUTPUT_DIR/${filename}.log"
            
        if [ $? -eq 0 ] && [ -f "$output_rttm" ]; then
            echo "  ✅ Success: $output_rttm"
        else
            echo "  ❌ Failed: $filename"
        fi
    else
        echo "[$count/$total] Skipping (already exists): $filename"
    fi
done

# Step 2: Combine all RTTM files
echo ""
echo "=== Step 2: Combining RTTM files ==="

# Remove existing combined file
rm -f "$COMBINED_RTTM"

# Count RTTM files
rttm_count=$(ls "$OUTPUT_DIR"/*.rttm 2>/dev/null | wc -l)
echo "Found $rttm_count RTTM files to combine"

if [ $rttm_count -eq 0 ]; then
    echo "Error: No RTTM files found to combine"
    exit 1
fi

# Combine all RTTM files
for rttm_file in "$OUTPUT_DIR"/*.rttm; do
    cat "$rttm_file" >> "$COMBINED_RTTM"
done

echo "Combined RTTM created: $COMBINED_RTTM"
echo "Total lines in combined RTTM: $(wc -l < "$COMBINED_RTTM")"

# Step 3: Evaluate combined RTTM
echo ""
echo "=== Step 3: Evaluating combined RTTM ==="

if [ ! -f "$REF_RTTM" ]; then
    echo "Error: Reference RTTM not found: $REF_RTTM"
    exit 1
fi

if [ ! -f "md-eval.pl" ]; then
    echo "Error: md-eval.pl not found in current directory"
    exit 1
fi

echo "Running DER evaluation..."
perl md-eval.pl -r "$REF_RTTM" -s "$COMBINED_RTTM" > "$RESULTS_DIR/redimnet_combined.eval" 2>&1

# Step 4: Parse and display results
echo ""
echo "=== Step 4: Results ==="

if [ -s "$RESULTS_DIR/redimnet_combined.eval" ]; then
    # Extract DER from the output
    der_line=$(grep "OVERALL" "$RESULTS_DIR/redimnet_combined.eval" | head -1)
    if [[ "$der_line" =~ ([0-9.]+)\% ]]; then
        der="${BASH_REMATCH[1]}"
        echo "✅ ReDimNet Combined DER: ${der}%"
    else
        echo "❌ Could not parse DER from result"
    fi
    
    # Extract individual components
    missed=$(grep "MISSED SPEECH" "$RESULTS_DIR/redimnet_combined.eval" | grep -oE '[0-9.]+%' | head -1 || echo "N/A")
    false_alarm=$(grep "FALARM SPEECH" "$RESULTS_DIR/redimnet_combined.eval" | grep -oE '[0-9.]+%' | head -1 || echo "N/A")
    speaker_error=$(grep "SPEECH DETECTION" "$RESULTS_DIR/redimnet_combined.eval" | grep -oE '[0-9.]+%' | head -1 || echo "N/A")
    
    echo "   Missed Speech: $missed"
    echo "   False Alarm: $false_alarm"
    echo "   Speaker Error: $speaker_error"
    
    # Save summary
    echo "ReDimNet Evaluation Summary" > "$RESULTS_DIR/summary.txt"
    echo "===========================" >> "$RESULTS_DIR/summary.txt"
    echo "Evaluation Date: $(date)" >> "$RESULTS_DIR/summary.txt"
    echo "ReDimNet Model: $REDIMNET_MODEL" >> "$RESULTS_DIR/summary.txt"
    echo "Total WAV files: $total" >> "$RESULTS_DIR/summary.txt"
    echo "Processed RTTM files: $rttm_count" >> "$RESULTS_DIR/summary.txt"
    echo "" >> "$RESULTS_DIR/summary.txt"
    echo "Combined DER Results:" >> "$RESULTS_DIR/summary.txt"
    echo "  Overall DER: ${der}%" >> "$RESULTS_DIR/summary.txt"
    echo "  Missed Speech: $missed" >> "$RESULTS_DIR/summary.txt"
    echo "  False Alarm: $false_alarm" >> "$RESULTS_DIR/summary.txt"
    echo "  Speaker Error: $speaker_error" >> "$RESULTS_DIR/summary.txt"
    
else
    echo "❌ Evaluation failed - empty result file"
fi

# Step 5: Compare with original ResNet34 results if available
echo ""
echo "=== Step 5: Comparison ==="

# Check if we can find the original ResNet34 DER
ORIGINAL_DER="Unknown"
if [ -f "/home/users/ntu/ygong006/Diarizen-master/evaluation_results/summary.txt" ]; then
    orig_der_line=$(grep "Average DER" "/home/users/ntu/ygong006/Diarizen-master/evaluation_results/summary.txt")
    if [[ "$orig_der_line" =~ ([0-9.]+)% ]]; then
        ORIGINAL_DER="${BASH_REMATCH[1]}%"
    fi
fi

echo "Original ResNet34 DER: $ORIGINAL_DER"
echo "ReDimNet ($REDIMNET_MODEL) DER: ${der}%"

# Add comparison to summary
echo "" >> "$RESULTS_DIR/summary.txt"
echo "Comparison:" >> "$RESULTS_DIR/summary.txt"
echo "  Original ResNet34: $ORIGINAL_DER" >> "$RESULTS_DIR/summary.txt"
echo "  ReDimNet ($REDIMNET_MODEL): ${der}%" >> "$RESULTS_DIR/summary.txt"

if [[ "$ORIGINAL_DER" =~ ([0-9.]+)% ]] && [[ "$der" =~ ^[0-9.]+$ ]]; then
    orig_val=$(echo "$ORIGINAL_DER" | sed 's/%//')
    difference=$(echo "scale=2; $der - $orig_val" | bc -l)
    if (( $(echo "$difference < 0" | bc -l) )); then
        improvement=$(echo "scale=2; $difference * -1" | bc -l)
        echo "✅ Improvement: -$improvement%"
        echo "  Improvement: -$improvement%" >> "$RESULTS_DIR/summary.txt"
    else
        echo "⚠️  Regression: +$difference%"
        echo "  Regression: +$difference%" >> "$RESULTS_DIR/summary.txt"
    fi
fi

echo ""
echo "=== Processing Complete ==="
echo "Results saved to: $RESULTS_DIR/"
echo "Combined RTTM: $COMBINED_RTTM"
echo "Evaluation details: $RESULTS_DIR/redimnet_combined.eval"
echo "Summary: $RESULTS_DIR/summary.txt"
echo "Completed at: $(date)"
