# Run with ResNet34 embeddings (baseline)
python run_diarization.py -i audio.wav -o output/ --batch-size 8

# Run with ReDimNet embeddings
python run_redimnet_direct_replace_fixed.py -i audio.wav -o output_redimnet/ --redimnet-model b1

## Custom Modifications

This repository includes a modified version of `diarizen/pipelines/inference.py` to support ReDimNet embedding integration.

**Changes made:**
- Replaced ResNet34 embedding extraction with ReDimNet
- Added projection layer for dimension matching
- Modified clustering input handling

Audio Input 
    → DiariZen Pipeline 
    → VAD + Segmentation 
    → Embedding Extraction (ReDimNet/ResNet34) 
    → VBx Clustering 
    → RTTM Output
