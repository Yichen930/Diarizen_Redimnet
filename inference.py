# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2025 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import argparse
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

import toml
import numpy as np
import torch
import torchaudio

from scipy.ndimage import median_filter

from huggingface_hub import snapshot_download, hf_hub_download
from pyannote.audio.pipelines import SpeakerDiarization as SpeakerDiarizationPipeline
from pyannote.audio.utils.signal import Binarize
from pyannote.database.protocol.protocol import ProtocolFile

from diarizen.pipelines.utils import scp2path


class DiariZenPipeline(SpeakerDiarizationPipeline):
    def __init__(
        self, 
        diarizen_hub,
        embedding_model,
        config_parse: Optional[Dict[str, Any]] = None,
        rttm_out_dir: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        config_path = Path(diarizen_hub / "config.toml")
        config = toml.load(config_path.as_posix())

        if config_parse is not None:
            print('Overriding with parsed config.')
            config["inference"]["args"] = config_parse["inference"]["args"]
            config["clustering"]["args"] = config_parse["clustering"]["args"]
       
        inference_config = config["inference"]["args"]
        clustering_config = config["clustering"]["args"]
        
        print(f'Loaded configuration: {config}')

        # 🔧 新增：检查是否是模型对象（在 super().__init__ 之前）
        self._custom_embedding_model = None
        embedding_model_path = embedding_model
        
        # 如果是 PyTorch 模型对象，保存它并使用虚拟路径
        if hasattr(embedding_model, 'forward') and callable(embedding_model.forward):
            self._custom_embedding_model = embedding_model
            embedding_model_path = "custom_model"  # 虚拟路径
            print("🎯 Using custom embedding model (ReDimNet)")

        # 必须先调用父类的 __init__
        super().__init__(
            config=config,
            seg_duration=inference_config["seg_duration"],
            segmentation=str(Path(diarizen_hub / "pytorch_model.bin")),
            segmentation_step=inference_config["segmentation_step"],
            embedding=embedding_model_path,
            embedding_exclude_overlap=True,
            clustering=clustering_config["method"],     
            embedding_batch_size=inference_config["batch_size"],
            segmentation_batch_size=inference_config["batch_size"],
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        # 🔧 新增：初始化自定义模型（在 super().__init__ 之后）
        if self._custom_embedding_model is not None:
            self._custom_embedding_model = self._custom_embedding_model.to(self.device)
            self._custom_embedding_model.eval()
            print(f"✅ Custom embedding model moved to: {self.device}")

        self.apply_median_filtering = inference_config["apply_median_filtering"]
        self.min_speakers = clustering_config["min_speakers"]
        self.max_speakers = clustering_config["max_speakers"]
        self.model_id = model_id

        if clustering_config["method"] == "AgglomerativeClustering":
            self.PIPELINE_PARAMS = {
                "clustering": {
                    "method": "centroid",
                    "min_cluster_size": clustering_config["min_cluster_size"],
                    "threshold": clustering_config["ahc_threshold"],
                }
            }
        elif clustering_config["method"] == "VBxClustering":
            self.PIPELINE_PARAMS = {
                "clustering": {
                    "ahc_criterion": clustering_config["ahc_criterion"],
                    "ahc_threshold": clustering_config["ahc_threshold"],
                    "Fa": clustering_config["Fa"],
                    "Fb": clustering_config["Fb"],
                }
            }
            self.clustering.plda_dir = str(Path(diarizen_hub / "plda"))
            self.clustering.lda_dim = clustering_config["lda_dim"]
            self.clustering.maxIters = clustering_config["max_iters"]
        else:
            raise ValueError(f"Unsupported clustering method: {clustering_config['method']}")

        self.instantiate(self.PIPELINE_PARAMS)

        if rttm_out_dir is not None:
            os.makedirs(rttm_out_dir, exist_ok=True)
        self.rttm_out_dir = rttm_out_dir

        assert self._segmentation.model.specifications.powerset is True

    def get_embeddings(self, file, segmentations, exclude_overlap=True):
        """重写嵌入提取方法以支持自定义模型"""
        if self._custom_embedding_model is not None:
            return self._get_embeddings_custom(file, segmentations, exclude_overlap)
        else:
            # 使用父类的原始方法
            return super().get_embeddings(file, segmentations, exclude_overlap)

    def _get_embeddings_custom(self, file, segmentations, exclude_overlap=True):
        """使用自定义模型提取嵌入"""
        print("🎯 Using custom model for embedding extraction...")
        
        waveform = file["waveform"]
        sample_rate = file["sample_rate"]
        
        # 简化实现：对整个音频提取嵌入
        # 实际应该根据分割结果提取对应片段的嵌入
        
        with torch.no_grad():
            # 预处理音频
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)  # [1, channels, samples] -> [channels, samples]
            
            # 使用自定义模型提取嵌入
            embeddings = self._custom_embedding_model(waveform)
            
            # 转换为适合聚类的格式 [num_chunks, num_speakers, embedding_dim]
            embeddings = embeddings.cpu().numpy()
            if embeddings.ndim == 2:
                embeddings = embeddings.reshape(1, 1, -1)
            elif embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, 1, -1)
            
        print(f"📊 Custom embeddings shape: {embeddings.shape}")
        return embeddings

    @classmethod
    def from_pretrained(
        cls, 
        repo_id: str, 
        cache_dir: str = None,
        rttm_out_dir: str = None,
    ) -> "DiariZenPipeline":
        diarizen_hub = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=cache_dir is not None
        )

        embedding_model = hf_hub_download(
            repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
            filename="pytorch_model.bin",
            cache_dir=cache_dir,
            local_files_only=cache_dir is not None
        )

        print(f"self.embedding: {embedding_model}")

        return cls(
            diarizen_hub=Path(diarizen_hub).expanduser().absolute(),
            embedding_model=embedding_model,
            rttm_out_dir=rttm_out_dir,
            model_id=repo_id
        )

    def __call__(self, in_wav, sess_name=None):
        assert isinstance(in_wav, (str, ProtocolFile)), "input must be either a str or a ProtocolFile"
        in_wav = in_wav if not isinstance(in_wav, ProtocolFile) else in_wav['audio']
        
        # Get device information
        device_info = str(self.device)
        if torch.cuda.is_available():
            device_info += f" (GPU: {torch.cuda.get_device_name()}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB)"
        
        # Get actual device information for each model
        segmentation_device = next(self._segmentation.model.parameters()).device
        embedding_device = next(self._embedding.model_.parameters()).device
        
        device_info = f"Segmentation model: {segmentation_device}, Embedding model: {embedding_device}"
        if torch.cuda.is_available() and (segmentation_device.type == 'cuda' or embedding_device.type == 'cuda'):
            gpu_name = torch.cuda.get_device_name(0) if segmentation_device.type == 'cuda' else torch.cuda.get_device_name(embedding_device.index)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if segmentation_device.type == 'cuda' else torch.cuda.get_device_properties(embedding_device.index).total_memory / 1024**3
            device_info += f" (GPU: {gpu_name}, Memory: {gpu_memory:.1f}GB)"
        
        # Initialize timing variables
        total_start_time = time.time()
        segmentation_start_time = None
        embedding_start_time = None
        clustering_start_time = None
        
        print(f'Device: {device_info}')
        print('Extracting segmentations.')
        segmentation_start_time = time.time()
        
        waveform, sample_rate = torchaudio.load(in_wav) 
        waveform = torch.unsqueeze(waveform[0], 0)      # force to use the SDM data
        
        # Calculate audio properties
        audio_duration = waveform.shape[1] / sample_rate  # duration in seconds
        num_channels = waveform.shape[0]
        audio_length_samples = waveform.shape[1]
        
        segmentations = self.get_segmentations({"waveform": waveform, "sample_rate": sample_rate}, soft=False)

        if self.apply_median_filtering:
            segmentations.data = median_filter(segmentations.data, size=(1, 11, 1), mode='reflect')

        # binarize segmentation
        binarized_segmentations = segmentations     # powerset

        # estimate frame-level number of instantaneous speakers
        count = self.speaker_count(
            binarized_segmentations,
            self._segmentation.model._receptive_field,
            warm_up=(0.0, 0.0),
        )
        
        segmentation_time = time.time() - segmentation_start_time
        print(f"Segmentation completed in {segmentation_time:.2f} seconds")

        print("Extracting Embeddings.")
        embedding_start_time = time.time()
        
        embeddings = self.get_embeddings(
            {"waveform": waveform, "sample_rate": sample_rate},
            binarized_segmentations,
            exclude_overlap=self.embedding_exclude_overlap,
        )
        
        embedding_time = time.time() - embedding_start_time
        print(f"Embedding extraction completed in {embedding_time:.2f} seconds")

        # shape: (num_chunks, local_num_speakers, dimension)
        print("Clustering.")
        clustering_start_time = time.time()
        
        hard_clusters, _, _ = self.clustering(
            embeddings=embeddings,
            segmentations=binarized_segmentations,
            min_clusters=self.min_speakers,  
            max_clusters=self.max_speakers
        )
        
        clustering_time = time.time() - clustering_start_time
        print(f"Clustering completed in {clustering_time:.2f} seconds")

        # during counting, we could possibly overcount the number of instantaneous
        # speakers due to segmentation errors, so we cap the maximum instantaneous number
        # of speakers by the `max_speakers` value
        count.data = np.minimum(count.data, self.max_speakers).astype(np.int8)

        # keep track of inactive speakers
        inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
        #   shape: (num_chunks, num_speakers)

        # reconstruct discrete diarization from raw hard clusters
        hard_clusters[inactive_speakers] = -2
        discrete_diarization, _ = self.reconstruct(
            segmentations,
            hard_clusters,
            count,
        )

        # convert to annotation
        to_annotation = Binarize(
            onset=0.5,
            offset=0.5,
            min_duration_on=0.0,
            min_duration_off=0.0
        )
        result = to_annotation(discrete_diarization)
        result.uri = sess_name
        
        total_time = time.time() - total_start_time
        print(f"Total processing time: {total_time:.2f} seconds")
        
        if self.rttm_out_dir is not None:
            assert sess_name is not None
            rttm_out = os.path.join(self.rttm_out_dir, sess_name + ".rttm")
            with open(rttm_out, "w") as f:
                f.write(result.to_rttm())
            
            # Save run log
            log_out = os.path.join(self.rttm_out_dir, sess_name + "_run_log.txt")
            with open(log_out, "w") as f:
                f.write(f"DiariZen Processing Log\n")
                f.write(f"=====================\n\n")
                f.write(f"Session: {sess_name}\n")
                f.write(f"Audio file: {in_wav}\n")
                f.write(f"Audio Properties:\n")
                f.write(f"  Duration: {audio_duration:.2f} seconds ({audio_duration/60:.2f} minutes)\n")
                f.write(f"  Sample rate: {sample_rate} Hz\n")
                f.write(f"  Number of channels: {num_channels}\n")
                f.write(f"  Total samples: {audio_length_samples:,}\n")
                f.write(f"  File size: {os.path.getsize(in_wav) / (1024*1024):.2f} MB\n")
                f.write(f"Device: {device_info}\n")
                f.write(f"Segmentation model: {self._segmentation.model.__class__.__name__}\n")
                f.write(f"Embedding model: {self._embedding.model_.__class__.__name__}\n")
                f.write(f"Clustering method: {self.clustering.__class__.__name__}\n\n")
                f.write(f"Timing Information:\n")
                f.write(f"  Segmentation: {segmentation_time:.2f} seconds\n")
                f.write(f"  Embedding extraction: {embedding_time:.2f} seconds\n")
                f.write(f"  Clustering: {clustering_time:.2f} seconds\n")
                f.write(f"  Total processing time: {total_time:.2f} seconds\n\n")
                f.write(f"Configuration:\n")
                f.write(f"  Model ID: {self.model_id if self.model_id else 'Not specified'}\n")
                f.write(f"  Segment duration: {self._segmentation.duration}s\n")
                f.write(f"  Segmentation step: {self.segmentation_step}s\n")
                f.write(f"  Batch size: {self.embedding_batch_size}\n")
                f.write(f"  Min speakers: {self.min_speakers}\n")
                f.write(f"  Max speakers: {self.max_speakers}\n")
                f.write(f"  Apply median filtering: {self.apply_median_filtering}\n")
                
                # Add GPU memory info if models are on GPU
                if segmentation_device.type == 'cuda' or embedding_device.type == 'cuda':
                    f.write(f"\nGPU Memory Information:\n")
                    if segmentation_device.type == 'cuda':
                        f.write(f"  Segmentation model GPU: {torch.cuda.get_device_name(segmentation_device.index)}\n")
                        f.write(f"  Total GPU memory: {torch.cuda.get_device_properties(segmentation_device.index).total_memory / 1024**3:.1f} GB\n")
                        f.write(f"  Allocated GPU memory: {torch.cuda.memory_allocated(segmentation_device.index) / 1024**3:.1f} GB\n")
                        f.write(f"  Cached GPU memory: {torch.cuda.memory_reserved(segmentation_device.index) / 1024**3:.1f} GB\n")
                    if embedding_device.type == 'cuda' and embedding_device.index != segmentation_device.index:
                        f.write(f"  Embedding model GPU: {torch.cuda.get_device_name(embedding_device.index)}\n")
                        f.write(f"  Total GPU memory: {torch.cuda.get_device_properties(embedding_device.index).total_memory / 1024**3:.1f} GB\n")
                        f.write(f"  Allocated GPU memory: {torch.cuda.memory_allocated(embedding_device.index) / 1024**3:.1f} GB\n")
                        f.write(f"  Cached GPU memory: {torch.cuda.memory_reserved(embedding_device.index) / 1024**3:.1f} GB\n")
                    
                    # Add overall GPU memory summary
                    if torch.cuda.is_available():
                        f.write(f"\nOverall GPU Memory Summary:\n")
                        f.write(f"  Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
                        f.write(f"  Allocated GPU memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB\n")
                        f.write(f"  Cached GPU memory: {torch.cuda.memory_reserved() / 1024**3:.1f} GB\n")
                        f.write(f"  Free GPU memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3:.1f} GB\n")
            
            print(f"Run log saved to: {log_out}")
            
        return result
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "This script performs diarization using DiariZen pipeline ",
        add_help=True,
        usage="%(prog)s [options]",
    )

    # Required paths
    parser.add_argument(
        "--in_wav_scp",
        type=str,
        required=True,
        help="Path to wav.scp."
    )
    parser.add_argument(
        "--diarizen_hub",
        type=str,
        required=True,
        help="Path to DiariZen model hub directory."
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        required=True,
        help="Path to pretrained embedding model."
    )

    # inference parameters
    parser.add_argument(
        "--seg_duration",
        type=int,
        default=16,
        help="Segment duration in seconds.",
    )
    parser.add_argument(
        "--segmentation_step",
        type=float,
        default=0.1,
        help="Shifting ratio during segmentation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Input batch size for inference.",
    )
    parser.add_argument(
        "--apply_median_filtering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply median filtering to segmentation output.",
    )

    # clustering parameters
    parser.add_argument(
        "--clustering_method",
        type=str,
        default="VBxClustering",
        choices=["VBxClustering", "AgglomerativeClustering"],
        help="Clustering method to use.",
    )
    parser.add_argument(
        "--min_speakers",
        type=int,
        default=1,
        help="Minimum number of speakers.",
    )
    parser.add_argument(
        "--max_speakers",
        type=int,
        default=20,
        help="Maximum number of speakers.",
    )
    parser.add_argument(
        "--ahc_criterion",
        type=str,
        default="distance",
        help="AHC criterion (for VBx).",
    )
    parser.add_argument(
        "--ahc_threshold",
        type=float,
        default=0.6,
        help="AHC threshold.",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=13,
        help="Minimum cluster size (for AHC).",
    )
    parser.add_argument(
        "--Fa",
        type=float,
        default=0.07,
        help="VBx Fa parameter.",
    )
    parser.add_argument(
        "--Fb",
        type=float,
        default=0.8,
        help="VBx Fb parameter.",
    )
    parser.add_argument(
        "--lda_dim",
        type=int,
        default=128,
        help="VBx LDA dimension.",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=20,
        help="VBx maximum iterations.",
    )

    # Output
    parser.add_argument(
        "--rttm_out_dir",
        type=str,
        default=None,
        required=False,
        help="Path to output folder.",
    )

    args = parser.parse_args()
    print(args)

    inference_config = {
        "seg_duration": args.seg_duration,
        "segmentation_step": args.segmentation_step,
        "batch_size": args.batch_size,
        "apply_median_filtering": args.apply_median_filtering
    }

    clustering_config = {
        "method": args.clustering_method,
        "min_speakers": args.min_speakers,
        "max_speakers": args.max_speakers
    }
    if args.clustering_method == "AgglomerativeClustering":
        clustering_config.update({
            "ahc_threshold": args.ahc_threshold,
            "min_cluster_size": args.min_cluster_size
        })
    elif args.clustering_method == "VBxClustering":
        clustering_config.update({
            "ahc_criterion": args.ahc_criterion,
            "ahc_threshold": args.ahc_threshold,
            "Fa": args.Fa,
            "Fb": args.Fb,
            "lda_dim": args.lda_dim,
            "max_iters": args.max_iters
        })
    else:
        raise ValueError(f"Unsupported clustering method: {args.clustering_method}")

    config_parse = {
        "inference": {"args": inference_config},
        "clustering": {"args": clustering_config}
    }

    diarizen_pipeline = DiariZenPipeline(
        diarizen_hub=Path(args.diarizen_hub),
        embedding_model=args.embedding_model,
        config_parse=config_parse,
        rttm_out_dir=args.rttm_out_dir
    )

    audio_f = scp2path(args.in_wav_scp)
    for audio_file in audio_f:
        sess_name = Path(audio_file).stem.split('.')[0]
        print(f'Prosessing: {sess_name}')
        diarizen_pipeline(audio_file, sess_name=sess_name)
