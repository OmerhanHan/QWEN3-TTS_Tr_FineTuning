# FLEURS Veri Kümesi ile Qwen3-TTS Fine-tuning Rehberi

Bu rehber Google FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech) veri kümesi kullanarak Qwen3-TTS modelini fine-tune etmenizi sağlar.

## Desteklenen Diller

FLEURS veri kümesi 102+ dili destekler. Popüler dil kodları:

- `hi_in` - Hindi (Hindistan)
- `tr_tr` - Türkçe (Türkiye)  
- `en_us` - İngilizce (ABD)
- `de_de` - Almanca (Almanya)
- `fr_fr` - Fransızca (Fransa)
- `es_es` - İspanyolca (İspanya)
- `it_it` - İtalyanca (İtalya)
- `ja_jp` - Japonca (Japonya)
- `ko_kr` - Korece (Güney Kore)
- `pt_br` - Portekizce (Brezilya)
- `ru_ru` - Rusça (Rusya)
- `zh_cn` - Çince (Çin)

Tüm dil listesi için: https://huggingface.co/datasets/google/fleurs

## Hızlı Başlangıç

### 1. Gerekli Kütüphaneleri Yükleyin

```bash
pip install qwen-tts datasets soundfile librosa tqdm
```

### 2. Tek Komutla Fine-tuning

```bash
# Hindi ile örnek
./run_fleurs_finetuning.sh hi_in 500 2 3 2e-5

# Türkçe ile örnek  
./run_fleurs_finetuning.sh tr_tr 300 4 5 1e-5

# İngilizce ile örnek
./run_fleurs_finetuning.sh en_us 1000 8 3 2e-5
```

Parametreler:
1. `dil_kodu` (örn: hi_in, tr_tr)
2. `max_örnekler` (varsayılan: 500)
3. `batch_size` (varsayılan: 2)
4. `epoch_sayısı` (varsayılan: 3)
5. `öğrenme_oranı` (varsayılan: 2e-5)

### 3. Manuel Adımlar

#### a) FLEURS Verisini Dönüştürme

```bash
python3 convert_fleurs_to_qwen_format.py \
  --language hi_in \
  --output_dir ./fleurs_data \
  --max_samples 500 \
  --split train
```

#### b) Audio Kodlarını Çıkarma

```bash
python3 prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl ./fleurs_data/train_raw_hi_in.jsonl \
  --output_jsonl ./fleurs_data/train_with_codes_hi_in.jsonl
```

#### c) Fine-tuning

```bash
python3 sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path ./output \
  --train_jsonl ./fleurs_data/train_with_codes_hi_in.jsonl \
  --batch_size 2 \
  --lr 2e-5 \
  --num_epochs 3 \
  --speaker_name fleurs_speaker_hi_in
```

## Özelleştirme Seçenekleri

### 1. Daha Fazla Veri Kullanma

```bash
python3 convert_fleurs_to_qwen_format.py \
  --language tr_tr \
  --max_samples 2000 \
  --output_dir ./large_fleurs_data
```

### 2. Belirli Konuşmacı Seçme

```bash
# Önce veriyi inceleyin ve konuşmacı ID'lerini görün
python3 -c "
from datasets import load_dataset
dataset = load_dataset('google/fleurs', 'tr_tr', split='train', streaming=True)
speakers = set()
for i, sample in enumerate(dataset):
    speakers.add(sample.get('speaker_id'))
    if i > 100: break
print('Konuşmacılar:', sorted(speakers))
"

# Belirli konuşmacıyı referans olarak kullanın
python3 convert_fleurs_to_qwen_format.py \
  --language tr_tr \
  --ref_speaker_id 1234 \
  --output_dir ./specific_speaker_data
```

### 3. Validation Verisi ile Test

```bash
python3 convert_fleurs_to_qwen_format.py \
  --language hi_in \
  --split validation \
  --max_samples 100 \
  --output_dir ./fleurs_validation
```

## Model Test Etme

Fine-tuning tamamlandıktan sonra:

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Modeli yükle
device = "cuda:0"
tts = Qwen3TTSModel.from_pretrained(
    "./output/checkpoint-epoch-2",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# Test metinleri (dile göre)
test_texts = {
    "hi_in": "नमस्ते, मैं एक कृत्रिम बुद्धिमत्ता हूँ।",
    "tr_tr": "Merhaba, ben bir yapay zeka asistanıyım.",
    "en_us": "Hello, I am an artificial intelligence assistant.",
}

# Ses üret
language = "hi_in"  # Değiştirin
wavs, sr = tts.generate_custom_voice(
    text=test_texts[language],
    speaker=f"fleurs_speaker_{language}",
)

# Kaydet
sf.write(f"test_output_{language}.wav", wavs[0], sr)
```

## Performans İpuçları

### GPU Belleği Optimizasyonu

```bash
# Düşük bellek kullanımı
./run_fleurs_finetuning.sh hi_in 200 1 3 2e-5

# Yüksek performans (16GB+ GPU)
./run_fleurs_finetuning.sh hi_in 1000 8 5 1e-5
```

### Batch Size Rehberi

- **4GB GPU**: batch_size=1
- **8GB GPU**: batch_size=2-4  
- **16GB GPU**: batch_size=4-8
- **24GB+ GPU**: batch_size=8-16

### Learning Rate Rehberi

- **Küçük veri (<500 örnek)**: lr=1e-5
- **Orta veri (500-1000)**: lr=2e-5
- **Büyük veri (1000+)**: lr=2e-6

## Hata Giderme

### CUDA Bellek Hatası
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### Hub Bağlantı Hatası
```bash
pip install --upgrade huggingface_hub
```

### Audio Dönüştürme Hatası
```bash
pip install --upgrade soundfile librosa
```

## Çıktı Dosya Yapısı

```
fleurs_training_<dil>/
├── fleurs_data/
│   ├── audio/              # Dönüştürülmüş WAV dosyaları
│   ├── ref_audio/          # Referans konuşmacı sesi
│   ├── train_raw_<dil>.jsonl
│   ├── train_with_codes_<dil>.jsonl
│   └── conversion_stats.json
├── model_output/
│   ├── checkpoint-epoch-0/
│   ├── checkpoint-epoch-1/
│   └── ...
└── test_model.py           # Otomatik test scripti
```

## İleri Seviye Kullanım

### Çoklu Dil Fine-tuning

```bash
# Birden fazla dili sırayla train edin
for lang in hi_in tr_tr en_us; do
    ./run_fleurs_finetuning.sh $lang 300 2 3 2e-5
done
```

### Checkpoint'ten Devam Etme

```bash
python3 sft_12hz.py \
  --init_model_path ./output/checkpoint-epoch-2 \
  --output_model_path ./continued_output \
  --train_jsonl ./new_train_data.jsonl \
  --batch_size 2 \
  --lr 1e-6 \
  --num_epochs 5 \
  --speaker_name continued_speaker
```