#!/bin/bash
set -e

echo "🚀 FLEURS verisi ile Qwen3-TTS Fine-tuning başlatılıyor..."

# Parametreler
LANGUAGE_CODE="${1:-hi_in}"  # Varsayılan: Hindi (India)
MAX_SAMPLES="${2:-500}"      # Varsayılan: 500 örnek
BATCH_SIZE="${3:-2}"         # Varsayılan: 2 (GPU belleğine göre ayarlayın)
NUM_EPOCHS="${4:-3}"         # Varsayılan: 3 epoch
LR="${5:-2e-5}"             # Varsayılan öğrenme oranı
USE_LORA="${6:-true}"        # Varsayılan: true (LoRA kullan — Mac M4 Pro için önerilir)
LORA_RANK="${7:-16}"         # LoRA rank
LORA_ALPHA="${8:-32}"        # LoRA alpha

# Script'in bulunduğu dizini referans al (hangi dizinden çağrılırsa çağrılsın çalışır)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Dizinler
WORK_DIR="${SCRIPT_DIR}/fleurs_training_${LANGUAGE_CODE}"
FLEURS_DATA_DIR="${WORK_DIR}/fleurs_data"
OUTPUT_DIR="${WORK_DIR}/model_output"

echo "📋 Parametreler:"
echo "  🌍 Dil: ${LANGUAGE_CODE}"
echo "  📊 Maksimum örnek sayısı: ${MAX_SAMPLES}"
echo "  🔢 Batch size: ${BATCH_SIZE}"
echo "  🔄 Epoch sayısı: ${NUM_EPOCHS}"
echo "  📈 Öğrenme oranı: ${LR}"
echo "  📁 Çalışma dizini: ${WORK_DIR}"
echo "  🔧 LoRA modu: ${USE_LORA}"
if [ "${USE_LORA}" = "true" ]; then
  echo "  📐 LoRA rank: ${LORA_RANK}"
  echo "  📐 LoRA alpha: ${LORA_ALPHA}"
fi

# Çalışma dizini oluştur
mkdir -p "${WORK_DIR}"

# 1. FLEURS verisini dönüştür
echo "📥 1. FLEURS verisi indiriliyor ve dönüştürülüyor..."
python3 "${SCRIPT_DIR}/convert_fleurs_to_qwen_format.py" \
  --language "${LANGUAGE_CODE}" \
  --output_dir "${FLEURS_DATA_DIR}" \
  --max_samples "${MAX_SAMPLES}" \
  --split "train"

# Dönüştürülmüş dosya yollarını al
RAW_JSONL="${FLEURS_DATA_DIR}/train_raw_${LANGUAGE_CODE}.jsonl"
TRAIN_JSONL="${FLEURS_DATA_DIR}/train_with_codes_${LANGUAGE_CODE}.jsonl"

echo "✅ FLEURS verisi başarıyla dönüştürüldü: ${RAW_JSONL}"

# 2. Audio kodlarını çıkar
echo "🎵 2. Audio kodları çıkarılıyor..."
python3 "${SCRIPT_DIR}/prepare_data.py" \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl "${RAW_JSONL}" \
  --output_jsonl "${TRAIN_JSONL}"

echo "✅ Audio kodları başarıyla çıkarıldı: ${TRAIN_JSONL}"

# 3. Fine-tuning başlat
SPEAKER_NAME="fleurs_speaker_${LANGUAGE_CODE}"

if [ "${USE_LORA}" = "true" ]; then
  echo "🎯 3. LoRA Fine-tuning başlatılıyor (M4 Pro optimize)..."
  LORA_LR="${LR:-2e-4}"  # LoRA için daha yüksek LR önerilir
  
  python3 "${SCRIPT_DIR}/sft_12hz_lora.py" \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --output_model_path "${OUTPUT_DIR}_lora" \
    --train_jsonl "${TRAIN_JSONL}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LORA_LR}" \
    --num_epochs "${NUM_EPOCHS}" \
    --speaker_name "${SPEAKER_NAME}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --merge_and_save
  
  echo "✅ LoRA Fine-tuning tamamlandı!"
else
  echo "🎯 3. Full Fine-tuning başlatılıyor..."
  
  python3 "${SCRIPT_DIR}/sft_12hz.py" \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --output_model_path "${OUTPUT_DIR}" \
    --train_jsonl "${TRAIN_JSONL}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --num_epochs "${NUM_EPOCHS}" \
    --speaker_name "${SPEAKER_NAME}"
  
  echo "✅ Full Fine-tuning tamamlandı!"
fi

# 4. Test scripti oluştur
echo "📝 4. Test scripti oluşturuluyor..."
cat > "${WORK_DIR}/test_model.py" << EOF
#!/usr/bin/env python3
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import os

def test_model(checkpoint_path, test_text, output_wav="test_output.wav"):
    print(f"🧪 Model test ediliyor: {checkpoint_path}")
    print(f"📝 Test metni: {test_text}")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Modeli yükle
    tts = Qwen3TTSModel.from_pretrained(
        checkpoint_path,
        device_map=device,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )
    
    # Ses üret
    wavs, sr = tts.generate_custom_voice(
        text=test_text,
        speaker="${SPEAKER_NAME}",
    )
    
    # Kaydet
    sf.write(output_wav, wavs[0], sr)
    print(f"🎵 Test sesi kaydedildi: {output_wav}")
    print(f"📊 Örnekleme oranı: {sr} Hz")
    print(f"⏱️  Süre: {len(wavs[0]) / sr:.2f} saniye")

if __name__ == "__main__":
    import sys
    
    # En son checkpoint'i bul
    checkpoints = [d for d in os.listdir("${OUTPUT_DIR}") if d.startswith("checkpoint-")]
    if not checkpoints:
        print("❌ Checkpoint bulunamadı!")
        sys.exit(1)
    
    # En son epoch'u seç
    latest_checkpoint = sorted(checkpoints)[-1]
    checkpoint_path = os.path.join("${OUTPUT_DIR}", latest_checkpoint)
    
    # Test metni (dile göre)
    test_texts = {
        "hi_in": "नमस्ते, यह एक परीक्षण है।",  # Hindi
        "tr_tr": "Merhaba, bu bir test cümlesidir.",  # Türkçe
        "en_us": "Hello, this is a test sentence.",  # İngilizce
        "de_de": "Hallo, das ist ein Testsatz.",  # Almanca
        "fr_fr": "Bonjour, ceci est une phrase de test.",  # Fransızca
        "es_es": "Hola, esta es una oración de prueba.",  # İspanyolca
    }
    
    test_text = test_texts.get("${LANGUAGE_CODE}", "Hello, this is a test.")
    
    test_model(checkpoint_path, test_text, f"test_${LANGUAGE_CODE}.wav")
EOF

chmod +x "${WORK_DIR}/test_model.py"

echo "🎉 FLEURS Fine-tuning süreci tamamlandı!"
echo ""
echo "📁 Sonuçlar:"
echo "  🎯 Model checkpoints: ${OUTPUT_DIR}/"
echo "  📊 Training verisi: ${TRAIN_JSONL}"
echo "  🎤 Referans ses: ${FLEURS_DATA_DIR}/ref_audio/reference.wav"
echo ""
echo "🧪 Modeli test etmek için:"
echo "  cd ${WORK_DIR}"
echo "  python3 test_model.py"
echo ""
echo "📋 Mevcut checkpoints:"
if [ -d "${OUTPUT_DIR}" ]; then
    ls -la "${OUTPUT_DIR}/" | grep checkpoint || echo "  Henüz checkpoint oluşturulmadı"
fi