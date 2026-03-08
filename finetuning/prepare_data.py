# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import torch
import gc
import sys
import os

# Local qwen_tts modülünü ekle
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from qwen_tts import Qwen3TTSTokenizer

# Cihaza göre batch size ayarlama
def get_optimal_batch_size(device):
    if device == "mps":
        return 4  # Apple MPS için çok daha küçük batch
    elif device.startswith("cuda"):
        return 16  # CUDA için orta batch
    else:
        return 8   # CPU için batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    args = parser.parse_args()

    # Cihaza göre batch size belirleme
    BATCH_INFER_NUM = get_optimal_batch_size(args.device)
    print(f"Cihaz: {args.device} | Batch size: {BATCH_INFER_NUM}")

    # MPS için bellek optimizasyonu
    if args.device == "mps":
        torch.mps.set_per_process_memory_fraction(0.8)
        print("MPS bellek fraksiyonu: 0.8")

    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model_path,
        device_map=args.device,
    )

    total_lines = open(args.input_jsonl).readlines()
    total_lines = [json.loads(line.strip()) for line in total_lines]
    print(f"Toplam {len(total_lines)} satır işlenecek")

    final_lines = []
    batch_lines = []
    batch_audios = []
    
    for i, line in enumerate(total_lines):
        batch_lines.append(line)
        batch_audios.append(line['audio'])

        if len(batch_lines) >= BATCH_INFER_NUM:
            print(f"İşleniyor: {i+1-BATCH_INFER_NUM+1}-{i+1} / {len(total_lines)}")
            
            # Bellek temizleme
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
            
            enc_res = tokenizer_12hz.encode(batch_audios)
            for code, line in zip(enc_res.audio_codes, batch_lines):
                line['audio_codes'] = code.cpu().tolist()
                final_lines.append(line)
            
            # Batch temizleme
            batch_lines.clear()
            batch_audios.clear()
            del enc_res
            
            # Bellek temizleme
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

    # Kalan öğeleri işle
    if len(batch_audios) > 0:
        print(f"Kalan işleniyor: {len(batch_audios)} öğe")
        
        # Bellek temizleme
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        
        enc_res = tokenizer_12hz.encode(batch_audios)
        for code, line in zip(enc_res.audio_codes, batch_lines):
            line['audio_codes'] = code.cpu().tolist()
            final_lines.append(line)
        
        # Temizleme
        batch_lines.clear()
        batch_audios.clear()
        del enc_res
        
        # Final bellek temizleme
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    print(f"Kodlama tamamlandı. Toplam {len(final_lines)} öğe işlendi.")
    final_lines = [json.dumps(line, ensure_ascii=False) for line in final_lines]

    print(f"Çıktı dosyasına yazılıyor: {args.output_jsonl}")
    with open(args.output_jsonl, 'w') as f:
        for line in final_lines:
            f.writelines(line + '\n')
            
    print("İşlem tamamlandı!")

if __name__ == "__main__":
    main()
