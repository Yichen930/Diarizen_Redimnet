import argparse
import os
import time
from pathlib import Path
from diarizen.pipelines.inference import DiariZenPipeline

def is_docker_environment():
    """Check if running in a Docker container."""
    return os.path.exists('/.dockerenv') or os.path.exists('/proc/1/cgroup') and 'docker' in open('/proc/1/cgroup').read()

def main():
    """
    Main function to run the DiariZen speaker diarization pipeline.
    Works in both Docker and local conda environments.
    """
    # Detect environment and set appropriate defaults
    is_docker = is_docker_environment()
    
    if is_docker:
        default_output_dir = "/app/output"
        default_model_cache_dir = "/app/models"
        audio_help = "Path to the input audio file inside the container (e.g., /app/audio_in/your_audio.wav)."
        cache_help = "Directory where models are cached. Should not be changed in Docker."
    else:
        default_output_dir = "./output"
        default_model_cache_dir = "~/.cache/huggingface/hub"
        audio_help = "Path to the input audio file (e.g., ./my_audio/file.wav)."
        cache_help = "Directory where models are cached. Defaults to HuggingFace cache directory."

    parser = argparse.ArgumentParser(
        description="Run DiariZen Speaker Diarization (Docker or local conda environment).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--audio-file",
        type=str,
        required=True,
        help=audio_help
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=default_output_dir,
        help="Directory to save the RTTM output file."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="BUT-FIT/diarizen-wavlm-base-s80-md",
        choices=["BUT-FIT/diarizen-wavlm-base-s80-md", "BUT-FIT/diarizen-wavlm-large-s80-md"],
        help="Hugging Face model ID to use."
    )
    parser.add_argument(
        "--model-cache-dir",
        type=str,
        default=default_model_cache_dir,
        help=cache_help
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference. Try smaller values (16, 8, 4, 2, 1) if you encounter CUDA out of memory errors."
    )
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cache = Path(args.model_cache_dir).expanduser()
    
    # Check if models exist in cache
    model_exists = (model_cache / f"models--{args.model_id.replace('/', '--')}" / "snapshots").exists()
    
    if not model_exists:
        if is_docker:
            print(f"ERROR: Models for '{args.model_id}' not found in '{model_cache}'.")
            print("Please ensure the models directory was correctly copied into the Docker image.")
            return
        else:
            print(f"Models for '{args.model_id}' not found in '{model_cache}'.")
            print("Downloading models from HuggingFace...")
            # Ensure cache directory exists
            model_cache.mkdir(parents=True, exist_ok=True)

    print(f"Loading model '{args.model_id}' from cache '{model_cache}'...")
    print(f"Using batch size: {args.batch_size}")
    
    # Start timing for model loading
    model_load_start = time.time()
    
    # Create custom configuration with the specified batch size
    config_parse = {
        "inference": {
            "args": {
                "seg_duration": 16,
                "segmentation_step": 0.1,
                "batch_size": args.batch_size,
                "apply_median_filtering": True
            }
        },
        "clustering": {
            "args": {
                "method": "VBxClustering",
                "min_speakers": 1,
                "max_speakers": 20,
                "ahc_criterion": "distance",
                "ahc_threshold": 0.6,
                "Fa": 0.07,
                "Fb": 0.8,
                "lda_dim": 128,
                "max_iters": 20
            }
        }
    }
    
    print (model_cache)
    print("9090900")
    print (args.model_id)
    # Get the model hub path
    from huggingface_hub import snapshot_download
    diarizen_hub = snapshot_download(
        repo_id=args.model_id,
        cache_dir=str(model_cache),
        local_files_only=True
    )
    
    # Get the embedding model path
    from huggingface_hub import hf_hub_download
    embedding_model = hf_hub_download(
        repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
        filename="pytorch_model.bin",
        cache_dir=str(model_cache),
        local_files_only=model_cache is not None
    )
    
    # Create pipeline with custom configuration
    diar_pipeline = DiariZenPipeline(
        diarizen_hub=Path(diarizen_hub).expanduser().absolute(),
        embedding_model=embedding_model,
        config_parse=config_parse,
        rttm_out_dir=str(output_dir),
        model_id=args.model_id
    )
    
    model_load_time = time.time() - model_load_start
    print(f"Model loading completed in {model_load_time:.2f} seconds")

    sess_name = audio_path.stem
    print(f"Processing '{audio_path}' with session name '{sess_name}'...")
    
    # Start timing for the complete pipeline
    pipeline_start = time.time()
    
    try:
        diar_pipeline(str(audio_path), sess_name=sess_name)
        pipeline_time = time.time() - pipeline_start
        total_time = model_load_time + pipeline_time
        
        print(f"✅ Processing complete. RTTM file saved in '{output_dir}'.")
        print(f"Pipeline execution time: {pipeline_time:.2f} seconds")
        print(f"Total time (including model loading): {total_time:.2f} seconds")
        
        # Update the run log with overall timing information
        log_file = output_dir / f"{sess_name}_run_log.txt"
        if log_file.exists():
            with open(log_file, "a") as f:
                f.write(f"\nOverall Timing:\n")
                f.write(f"  Model loading: {model_load_time:.2f} seconds\n")
                f.write(f"  Pipeline execution: {pipeline_time:.2f} seconds\n")
                f.write(f"  Total time: {total_time:.2f} seconds\n")
        
    except Exception as e:
        pipeline_time = time.time() - pipeline_start
        total_time = model_load_time + pipeline_time
        print(f"❌ Error during processing: {str(e)}")
        print(f"Time elapsed before error: {pipeline_time:.2f} seconds")
        print(f"Total time (including model loading): {total_time:.2f} seconds")
        
        # Save error log
        error_log = output_dir / f"{sess_name}_error_log.txt"
        with open(error_log, "w") as f:
            f.write(f"DiariZen Error Log\n")
            f.write(f"=================\n\n")
            f.write(f"Session: {sess_name}\n")
            f.write(f"Audio file: {audio_path}\n")
            f.write(f"Error: {str(e)}\n")
            f.write(f"Model loading time: {model_load_time:.2f} seconds\n")
            f.write(f"Time elapsed before error: {pipeline_time:.2f} seconds\n")
            f.write(f"Total time: {total_time:.2f} seconds\n")
        
        print(f"Error log saved to: {error_log}")
        raise

if __name__ == "__main__":
    main()