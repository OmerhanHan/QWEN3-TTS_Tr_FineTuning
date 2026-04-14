<img width="1024" height="1024" alt="image" src="<img width="1376" height="768" alt="17761729225c15" src="https://github.com/user-attachments/assets/1d2e9542-1639-4994-b7ae-1836cc07f451" />
" />


# 🇹🇷 Qwen3-TTS Türkçe Fine-Tuning & Ses Klonlama

Qwen3-TTS modelini **Türkçe veri setiyle fine-tune** edip, **hedef bir sesi klonlamak** için hazırlanmış Google Colab notebook'u.

> ⚠️ **A100 GPU gereklidir.** Colab Pro kullanmanız önerilir.

---

## 📋 İçindekiler

- [Gereksinimler](#gereksinimler)
- [Veri Seti Yapısı](#veri-seti-yapısı)
- [Hızlı Başlangıç](#hızlı-başlangıç)
- [Notebook Adımları](#notebook-adımları)
- [Ayarlar (Hyperparameters)](#ayarlar-hyperparameters)
- [Checkpoint Kullanımı](#checkpoint-kullanımı)
- [Sorun Giderme](#sorun-giderme)

---

## Gereksinimler

| Gereksinim | Detay |
|---|---|
| **GPU** | NVIDIA A100 (Colab Pro önerilir) |
| **Python** | 3.10+ |
| **Kütüphaneler** | `qwen-tts`, `accelerate`, `safetensors`, `librosa`, `soundfile`, `flash-attn` |
| **Depolama** | Google Drive (veri seti + çıktılar için) |

---

## Veri Seti Yapısı

Google Drive'da aşağıdaki düzende bir klasör hazırlayın:

```
turkish_tts_dataset/
├── metadata_cleaned.csv    # id|text|normalized_text formatında
└── wavs/
    ├── 0001.wav
    ├── 0002.wav
    └── ...
```

### CSV Formatı

```
id|text|normalized_text
0001|Merhaba dünya!|merhaba dünya
0002|Nasılsınız?|nasılsınız
```

### Referans Ses (Voice Cloning)

- Klonlamak istediğiniz sesi `.wav` formatında Drive'a yükleyin
- **24 kHz, mono** olmalı
- CSV'de bu dosya için ayrı bir satır gerekmez
- Notebook'taki `REF_AUDIO_PATH` değişkenini bu dosyanın yoluna ayarlayın

---

## Hızlı Başlangıç

1. **Notebook'u açın:**
   [`Qwen3TTS_Turkish_Finetune.ipynb`](Qwen3TTS_Turkish_Finetune.ipynb) dosyasını Google Colab'da açın

2. **Runtime ayarı:**
   `Runtime → Change runtime type → A100 GPU` seçin

3. **Veri setinizi Drive'a yükleyin** (yukarıdaki yapıya uygun şekilde)

4. **Referans ses dosyasını** Drive'a yükleyin

5. **Notebook'taki yolları güncelleyin:**
   ```python
   DATASET_DIR = "/content/drive/MyDrive/turkish_tts_dataset"
   REF_AUDIO_PATH = "/content/drive/MyDrive/ceren.wav"
   ```

6. **Hücreleri sırasıyla çalıştırın** ▶️

---

## Notebook Adımları

| # | Adım | Açıklama | Tahmini Süre |
|---|---|---|---|
| 1️⃣ | **Kurulum** | Kütüphaneler + Qwen3-TTS repo klonlama | ~3 dk |
| — | **Colab Yaması** | Accelerate/TensorBoard uyumluluk düzeltmesi | < 1 dk |
| 2️⃣ | **Drive & Veri Seti** | Google Drive bağlama, dosya kontrolü | < 1 dk |
| 3️⃣ | **Ayarlar** | Hyperparameter yapılandırma | < 1 dk |
| 4️⃣ | **CSV → JSONL** | Veri setini model formatına dönüştürme | ~1 dk |
| 5️⃣ | **Ses Kodları** | Wav → audio code token çıkarma | ~10-15 dk (2K örnek) |
| 6️⃣ | **Fine-Tuning** | Model eğitimi | ~1-2 saat (2K örnek, 3 epoch) |
| 7️⃣ | **Test** | Türkçe ses üretme ve dinleme | ~2 dk |
| 8️⃣ | **Kaydetme** | Checkpoint + test seslerini Drive'a yedekleme | ~5 dk |

---

## Ayarlar (Hyperparameters)

Notebook'un **3. hücresinden** tüm ayarları değiştirebilirsiniz:

```python
MAX_SAMPLES   = 2000     # Kullanılacak örnek sayısı (tam set ~98K)
BATCH_SIZE    = 2        # GPU belleğine göre ayarlayın
LEARNING_RATE = 2e-5     # Öğrenme hızı
NUM_EPOCHS    = 3        # Epoch sayısı
SPEAKER_NAME  = "ceren"  # Konuşmacı kimliği
```

### Öneriler

| Senaryo | MAX_SAMPLES | NUM_EPOCHS | Tahmini Süre |
|---|---|---|---|
| Hızlı test | 500 | 2 | ~30 dk |
| Standart | 2000 | 3 | ~1-2 saat |
| Yüksek kalite | 5000+ | 5 | ~4-6 saat |

---

## Checkpoint Kullanımı

Eğitim tamamlandıktan sonra checkpoint'ları yükleyip ses üretebilirsiniz:

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained(
    "checkpoint-epoch-2",       # checkpoint yolu
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = tts.generate_custom_voice(
    text="Merhaba, ben Ceren. Bugün hava çok güzel.",
    speaker="ceren",
)
sf.write("output.wav", wavs[0], sr)
```

---

## Sorun Giderme

### Accelerate / TensorBoard Hatası
Notebook'taki **Colab Yaması** hücresini çalıştırın. Bu, `sft_12hz.py` dosyasındaki `log_with="tensorboard"` parametresini kaldırır.

### GPU Bellek Yetersiz (OOM)
- `BATCH_SIZE` değerini **1**'e düşürün
- `MAX_SAMPLES` değerini azaltın

### `metadata_cleaned.csv` Bulunamadı
- Drive'daki yolun doğru olduğundan emin olun
- Dosya adının tam olarak `metadata_cleaned.csv` olduğunu kontrol edin

### Referans ses bulunamadı
- `REF_AUDIO_PATH` yolunun doğru olduğundan emin olun
- Dosyanın **24 kHz, mono WAV** formatında olduğunu kontrol edin

---

## Kullanılan Modeller

| Model | HuggingFace ID |
|---|---|
| Tokenizer | `Qwen/Qwen3-TTS-Tokenizer-12Hz` |
| Base Model | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |

---

## Lisans

Bu proje [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) resmi reposunu temel alır. Lisans bilgisi için ana repo'ya bakın.
