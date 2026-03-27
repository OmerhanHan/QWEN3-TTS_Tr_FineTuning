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
"""
Qwen3-TTS LoRA Fine-Tuning Script
Mac M4 Pro (MPS) ve düşük VRAM'li cihazlar için optimize edilmiş.

Kullanım:
    python sft_12hz_lora.py \
        --train_jsonl train_with_codes_tr_tr.jsonl \
        --batch_size 1 \
        --num_epochs 3 \
        --lora_rank 16

Full fine-tune vs LoRA karşılaştırması:
    Full: ~1.7B parametre, ~14-16 GB VRAM
    LoRA: ~8.4M parametre (%0.5), ~6-8 GB VRAM
"""
import argparse
import json
import os
import shutil
import gc
import sys

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from huggingface_hub import snapshot_download

# Local qwen_tts modülünü ekle
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

# LoRA / PEFT
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


def get_device_accelerator_config():
    """Cihaza göre Accelerator konfigürasyonu — LoRA için daha agresif gradient accumulation"""
    if torch.cuda.is_available():
        return Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS için LoRA-optimized ayarlar
        return Accelerator(gradient_accumulation_steps=8, mixed_precision="no", log_with="tensorboard")
    else:
        return Accelerator(gradient_accumulation_steps=8, log_with="tensorboard")


def memory_cleanup():
    """Bellek temizliği"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def print_trainable_params(model):
    """Eğitilebilir parametre istatistiklerini göster"""
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    
    print(f"📊 Parametre İstatistikleri:")
    print(f"   Toplam parametreler:     {total:>12,}")
    print(f"   Eğitilebilir parametreler: {trainable:>12,}")
    print(f"   Eğitilebilir oran:       {100 * trainable / total:.2f}%")
    return trainable, total


def get_lora_target_modules():
    """
    LoRA uygulanacak hedef modülleri döndür.
    
    Talker (Qwen3TTSTalkerForConditionalGeneration) içindeki:
    - Attention katmanları: q_proj, k_proj, v_proj, o_proj
    - MLP katmanları: gate_proj, up_proj, down_proj
    """
    return [
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


target_speaker_embedding = None

def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser(description="Qwen3-TTS LoRA Fine-Tuning")
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output_lora")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)   # LoRA için daha yüksek LR
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    
    # LoRA hiperparametreleri
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank (düşük = daha az parametre, yüksek = daha iyi kalite)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (genellikle rank'ın 2 katı)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout oranı")
    parser.add_argument("--merge_and_save", action="store_true",
                        help="Eğitim sonrası LoRA ağırlıklarını modele birleştir ve tam model kaydet")
    
    args = parser.parse_args()

    # Cihaza özel accelerator konfigürasyonu
    accelerator = get_device_accelerator_config()
    
    # MPS için bellek optimizasyonu
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(0.7)
        print("🍎 Apple MPS algılandı — LoRA modu etkin")
        print("   MPS bellek fraksiyonu: 0.7")
    
    memory_cleanup()  # Başlangıç temizliği

    MODEL_PATH = args.init_model_path
    
    # ====== 1. Model Yükleme ======
    print(f"\n🔄 Model yükleniyor: {MODEL_PATH}")
    
    # Cihaza göre dtype belirleme
    if torch.cuda.is_available():
        model_dtype = torch.bfloat16
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        model_dtype = torch.float32  # MPS ile LoRA float32 daha stabil
    else:
        model_dtype = torch.float32
    
    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=model_dtype,
        attn_implementation="eager",  # flash_attention_2 MPS'te çalışmaz
        low_cpu_mem_usage=True,
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)
    
    print(f"✅ Model yüklendi (dtype: {model_dtype})")
    
    # ====== 2. LoRA Konfigürasyonu ======
    print(f"\n🎯 LoRA konfigürasyonu oluşturuluyor...")
    print(f"   Rank: {args.lora_rank}")
    print(f"   Alpha: {args.lora_alpha}")
    print(f"   Dropout: {args.lora_dropout}")
    
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=get_lora_target_modules(),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # Speaker encoder ve embedding'leri dondur — LoRA sadece attention/MLP'ye uygulanır
        modules_to_save=["codec_embedding"],  # codec_embedding güncellenmeli (speaker embedding için)
    )
    
    # ====== 3. LoRA'yı Talker Modeline Uygula ======
    # Önce tüm modeli dondur
    for param in qwen3tts.model.parameters():
        param.requires_grad = False
    
    # Talker'a LoRA uygula
    print(f"\n🔧 LoRA Talker modeline uygulanıyor...")
    qwen3tts.model.talker = get_peft_model(qwen3tts.model.talker, lora_config)
    
    print(f"\n✅ LoRA uygulandı!")
    print_trainable_params(qwen3tts.model.talker)
    
    # ====== 4. Veri Yükleme ======
    print(f"\n📂 Eğitim verisi yükleniyor: {args.train_jsonl}")
    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    print(f"   Toplam örnek: {len(train_data)}")
    
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=dataset.collate_fn
    )

    # ====== 5. Optimizer — sadece LoRA parametreleri ======
    # Sadece requires_grad=True olan parametreleri optimize et
    trainable_params = [p for p in qwen3tts.model.talker.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    
    print(f"\n⚙️  Optimizer: AdamW (lr={args.lr}, weight_decay=0.01)")
    print(f"   Optimize edilen parametre grubu sayısı: {len(trainable_params)}")

    # ====== 6. Accelerator Hazırlığı ======
    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs
    model.train()

    total_steps = len(train_dataloader) * num_epochs
    print(f"\n🚀 Eğitim başlıyor!")
    print(f"   Epoch sayısı: {num_epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Toplam step: {total_steps}")
    print(f"   Gradient accumulation: {accelerator.gradient_accumulation_steps}")
    print(f"   Efektif batch size: {args.batch_size * accelerator.gradient_accumulation_steps}")
    print(f"{'='*60}\n")

    # ====== 7. Eğitim Döngüsü ======
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                # Speaker embedding (dondurulmuş, gradient yok)
                speaker_embedding = model.speaker_encoder(ref_mels.to(model.talker.device).to(model.talker.dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                # LoRA sarmalı üzerinden embedding erişimi
                # PeftModel, base_model üzerinden orijinal katmanlara erişim sağlar
                talker_base = model.talker.base_model.model if hasattr(model.talker, 'base_model') else model.talker
                
                input_text_embedding = talker_base.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = talker_base.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    code_predictor = talker_base.code_predictor if hasattr(talker_base, 'code_predictor') else model.talker.code_predictor
                    codec_i_embedding = code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                # Forward pass — LoRA katmanları otomatik olarak devreye girer
                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[:, 1:][codec_mask[:, 1:]]

                # Sub-talker fine-tune (code_predictor)
                sub_talker_forward = talker_base.forward_sub_talker_finetune if hasattr(talker_base, 'forward_sub_talker_finetune') else model.talker.forward_sub_talker_finetune
                sub_talker_logits, sub_talker_loss = sub_talker_forward(talker_codec_ids, talker_hidden_states)

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)

                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                num_batches += 1

            if step % 10 == 0:
                avg_loss = epoch_loss / max(num_batches, 1)
                accelerator.print(f"📈 Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f}")
            
            # Periyodik bellek temizliği
            if step % 50 == 0:
                memory_cleanup()

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        accelerator.print(f"\n✅ Epoch {epoch} tamamlandı | Ortalama Loss: {avg_epoch_loss:.4f}\n")

        # ====== 8. Checkpoint Kaydetme ======
        if accelerator.is_main_process:
            lora_output_dir = os.path.join(args.output_model_path, f"lora-checkpoint-epoch-{epoch}")
            os.makedirs(lora_output_dir, exist_ok=True)
            
            # LoRA adapter ağırlıklarını kaydet (çok küçük, ~35 MB)
            unwrapped_talker = accelerator.unwrap_model(model).talker
            unwrapped_talker.save_pretrained(lora_output_dir)
            
            # Speaker embedding'i ayrıca kaydet  
            speaker_emb_path = os.path.join(lora_output_dir, "speaker_embedding.pt")
            if target_speaker_embedding is not None:
                torch.save(target_speaker_embedding.cpu(), speaker_emb_path)
            
            # Konfigürasyon bilgilerini kaydet  
            lora_meta = {
                "base_model": MODEL_PATH,
                "speaker_name": args.speaker_name,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "epoch": epoch,
                "avg_loss": avg_epoch_loss,
                "target_modules": get_lora_target_modules(),
            }
            with open(os.path.join(lora_output_dir, "lora_training_meta.json"), 'w') as f:
                json.dump(lora_meta, f, indent=2, ensure_ascii=False)
            
            print(f"💾 LoRA checkpoint kaydedildi: {lora_output_dir}")
            
            # Son epoch'ta ve --merge_and_save flag'i ile tam model oluştur
            if epoch == num_epochs - 1 and args.merge_and_save:
                print(f"\n🔀 LoRA ağırlıkları modele birleştiriliyor...")
                merged_output_dir = os.path.join(args.output_model_path, f"merged-checkpoint-epoch-{epoch}")
                
                # LoRA'yı birleştir
                merged_talker = unwrapped_talker.merge_and_unload()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.talker = merged_talker
                
                # HuggingFace model dosyalarını kopyala
                local_model_path = MODEL_PATH
                if not os.path.isdir(local_model_path):
                    local_model_path = snapshot_download(MODEL_PATH)
                shutil.copytree(local_model_path, merged_output_dir, dirs_exist_ok=True)
                
                # Config güncelle (mevcut sft_12hz.py mantığı)
                input_config_file = os.path.join(local_model_path, "config.json")
                output_config_file = os.path.join(merged_output_dir, "config.json")
                with open(input_config_file, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                config_dict["tts_model_type"] = "custom_voice"
                talker_config = config_dict.get("talker_config", {})
                talker_config["spk_id"] = {
                    args.speaker_name: 3000
                }
                talker_config["spk_is_dialect"] = {
                    args.speaker_name: False
                }
                config_dict["talker_config"] = talker_config
                
                with open(output_config_file, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                
                # Birleştirilmiş ağırlıkları kaydet
                state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}
                
                drop_prefix = "speaker_encoder"
                keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
                for k in keys_to_drop:
                    del state_dict[k]
                
                weight = state_dict['talker.model.codec_embedding.weight']
                state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
                save_path = os.path.join(merged_output_dir, "model.safetensors")
                save_file(state_dict, save_path)
                
                print(f"✅ Birleştirilmiş tam model kaydedildi: {merged_output_dir}")
    
    print(f"\n{'='*60}")
    print(f"🎉 LoRA Fine-tuning tamamlandı!")
    print(f"   LoRA checkpoints: {args.output_model_path}/")
    if args.merge_and_save:
        print(f"   Birleştirilmiş model: {args.output_model_path}/merged-checkpoint-epoch-{num_epochs-1}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
