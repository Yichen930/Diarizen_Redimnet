import argparse
import os
import time
import sys
from pathlib import Path

# 添加 ReDimNet 到路径
sys.path.insert(0, './redimnet-repo')

import torch
import torch.nn as nn
import numpy as np
from diarizen.pipelines.inference import DiariZenPipeline

class ReDimNetWrapper(nn.Module):
    """包装ReDimNet以匹配pyannote嵌入模型的接口"""
    
    def __init__(self, redimnet_model):
        super().__init__()
        self.redimnet_model = redimnet_model
        # 原始模型的输出维度是256，但ReDimNet是192
        # 我们需要添加一个投影层来匹配维度
        self.projection = nn.Linear(192, 256)
        
    def forward(self, waveforms, weights=None):
        """
        适配pyannote的调用接口
        waveforms: 音频波形 [batch, channels, samples] 或 [batch, samples]
        weights: 可选的权重参数（我们忽略它）
        """
        # ReDimNet期望 [batch, samples] 形状
        original_shape = waveforms.shape
        
        if waveforms.dim() == 3:
            # [batch, channels, samples] -> [batch, samples]
            waveforms = waveforms.squeeze(1)
        
        # 使用ReDimNet提取嵌入
        embeddings = self.redimnet_model(waveforms)
        
        # 投影到256维以匹配原始模型
        embeddings = self.projection(embeddings)
        
        return embeddings

def main():
    parser = argparse.ArgumentParser(description="Run DiariZen with ReDimNet - Direct model replacement")
    parser.add_argument("-i", "--audio-file", type=str, required=True)
    parser.add_argument("-o", "--output-dir", type=str, default="./output_redimnet")
    parser.add_argument("--model-id", type=str, default="BUT-FIT/diarizen-wavlm-base-s80-md")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--redimnet-model", type=str, default="b1")
    
    args = parser.parse_args()

    print(f"🚀 Direct ReDimNet model replacement")
    print(f"🔧 Using ReDimNet: {args.redimnet_model}")
    
    # 创建输出目录
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # 加载ReDimNet
    from hubconf import ReDimNet
    redimnet_original = ReDimNet(model_name=args.redimnet_model, train_type="ptn", dataset="vox2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    redimnet_original = redimnet_original.to(device)
    redimnet_original.eval()
    
    # 包装ReDimNet
    redimnet_wrapped = ReDimNetWrapper(redimnet_original).to(device)
    redimnet_wrapped.eval()
    
    # 使用原始DiariZen pipeline
    from huggingface_hub import snapshot_download, hf_hub_download
    
    diarizen_hub = snapshot_download(
        repo_id=args.model_id,
        cache_dir="./models",
        local_files_only=True
    )
    
    # 使用原始嵌入模型路径创建pipeline
    embedding_model_path = hf_hub_download(
        repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
        filename="pytorch_model.bin",
        cache_dir="./models",
        local_files_only=True
    )
    
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
    
    # 创建pipeline
    pipeline = DiariZenPipeline(
        diarizen_hub=Path(diarizen_hub),
        embedding_model=embedding_model_path,
        config_parse=config_parse,
        rttm_out_dir=args.output_dir,
        model_id=args.model_id
    )
    
    # 🔧 直接替换嵌入模型
    pipeline._embedding.model_ = redimnet_wrapped
    print("✅ ReDimNet model replaced successfully!")
    print(f"📊 Original embedding dim: 256, ReDimNet dim: 192 -> 256 (with projection)")
    
    # 运行处理
    pipeline(args.audio_file, sess_name=Path(args.audio_file).stem)
    print(f"✅ Processing complete with ReDimNet!")

if __name__ == "__main__":
    main()
