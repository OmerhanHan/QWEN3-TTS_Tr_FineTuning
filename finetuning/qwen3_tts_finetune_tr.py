#!/usr/bin/env python3
"""
Qwen3-TTS Fine-Tuning Script for Turkish (tr_tr)
MSP GPU üzerinde çalıştırmak için tasarlanmıştır.

Kullanım:
    python qwen3_tts_finetune_tr.py [--step STEP_NUMBER]

Adımlar:
    0 -> Tüm adımları sırayla çalıştır (varsayılan)
    1 -> Sadece ortam kurulumu
    2 -> Sadece FLEURS veri dönüştürme
    3 -> Sadece audio kod çıkarma
    4 -> Sadece fine-tuning
    5 -> Sadece test/inference
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def detect_device():
    """Kullanılabilir cihazı otomatik tespit et"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except ImportError:
        return "cpu"


# ========== KONFİGÜRASYON ==========
LANGUAGE_CODE = "tr_tr"
MAX_SAMPLES = 800
BATCH_SIZE = 1 if detect_device() == "mps" else 2  # MPS için 1, diğerleri için 2
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
DEVICE = detect_device()  # Otomatik cihaz tespiti
SPEAKER_NAME = f"fleurs_speaker_{LANGUAGE_CODE}"

# LoRA konfigürasyonu — Mac M4 Pro için varsayılan olarak açık
USE_LORA = True if detect_device() == "mps" else False
LORA_RANK = 16
LORA_ALPHA = 32
LORA_LR = 2e-4  # LoRA için daha yüksek öğrenme oranı

TOKENIZER_MODEL = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

# Dizin yapısı
SCRIPT_DIR = Path(__file__).parent.resolve()
WORK_DIR = SCRIPT_DIR / f"fleurs_training_{LANGUAGE_CODE}"
FLEURS_DATA_DIR = WORK_DIR / "fleurs_data"
OUTPUT_DIR = WORK_DIR / "model_output"

RAW_JSONL = FLEURS_DATA_DIR / f"train_raw_{LANGUAGE_CODE}.jsonl"
TRAIN_JSONL = FLEURS_DATA_DIR / f"train_with_codes_{LANGUAGE_CODE}.jsonl"
# =====================================


def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def print_config():
    """Konfigürasyonu göster"""
    print_header("📋 KONFİGÜRASYON")
    print(f"  🌍 Dil            : {LANGUAGE_CODE}")
    print(f"  📊 Örnek sayısı   : {MAX_SAMPLES}")
    print(f"  🔢 Batch size     : {BATCH_SIZE}")
    print(f"  🔄 Epoch sayısı   : {NUM_EPOCHS}")
    print(f"  📈 Öğrenme oranı  : {LORA_LR if USE_LORA else LEARNING_RATE}")
    print(f"  🖥️  Cihaz          : {DEVICE}")
    print(f"  🎤 Speaker adı    : {SPEAKER_NAME}")
    print(f"  📁 Çalışma dizini : {WORK_DIR}")
    print(f"  🔧 LoRA modu      : {'✅ Aktif' if USE_LORA else '❌ Kapalı'}")
    if USE_LORA:
        print(f"  📐 LoRA rank      : {LORA_RANK}")
        print(f"  📐 LoRA alpha     : {LORA_ALPHA}")
    print()


def check_accelerator():
    """Mevcut hızlandırıcıyı (GPU/MPS/CPU) kontrol et"""
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"  ✅ CUDA GPU Bulundu: {gpu_name}")
            print(f"  💾 GPU Belleği: {gpu_mem:.1f} GB")
            
            # Batch size önerisi
            if gpu_mem < 6:
                suggested_bs = 1
            elif gpu_mem < 12:
                suggested_bs = 2
            elif gpu_mem < 20:
                suggested_bs = 4
            else:
                suggested_bs = 8
            
            if BATCH_SIZE > suggested_bs:
                print(f"  ⚠️  Uyarı: batch_size={BATCH_SIZE} bu GPU için yüksek olabilir. Önerilen: {suggested_bs}")
            return True
            
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                import psutil
                total_mem = psutil.virtual_memory().total / (1024**3)
            except ImportError:
                total_mem = 16  # Varsayılan değer
            
            print(f"  ✅ Apple MPS Bulundu (Metal Performance Shaders)")
            print(f"  🍎 Apple Silicon çip tespit edildi")
            print(f"  💾 Sistem Belleği: {total_mem:.1f} GB")
            
            # MPS için batch size önerisi (unified memory kullanır)
            if total_mem < 16:
                suggested_bs = 1
            elif total_mem < 32:
                suggested_bs = 2
            else:
                suggested_bs = 4
            
            if BATCH_SIZE > suggested_bs:
                print(f"  ⚠️  Uyarı: batch_size={BATCH_SIZE} bu sistem için yüksek olabilir. Önerilen: {suggested_bs}")
            return True
            
        else:
            print(f"  ⚠️  Sadece CPU mevcut - eğitim yavaş olacak")
            print(f"  💡 Öneriler:")
            if platform.system() == "Darwin":  # macOS
                print(f"     - PyTorch'un MPS desteği olan sürümünü kurun")
                print(f"     - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
            else:
                print(f"     - CUDA destekli bir GPU kullanın")
                print(f"     - Bulut GPU servisleri (Colab, AWS, Azure) deneyin")
            return False
            
    except ImportError:
        print("  ❌ PyTorch yüklü değil!")
        return False


def step1_setup():
    """Ortam kurulumu ve bağımlılıkları yükle"""
    print_header("🔧 ADIM 1: Ortam Kurulumu")
    
    # Hızlandırıcı kontrolü
    print("Hızlandırıcı durumu kontrol ediliyor...")
    if not check_accelerator():
        print("\n⚠️  Uyarı: GPU/MPS bulunamadı!")
        if DEVICE == "cpu":
            print("CPU ile devam edecek - eğitim çok yavaş olabilir.")
            response = input("Devam etmek istiyor musunuz? (y/N): ")
            if response.lower() not in ['y', 'yes', 'evet']:
                sys.exit(1)
        else:
            print(f"Makine öğrenmesi hızlandırması için GPU/MPS önerilir.")
            sys.exit(1)
    
    # Gerekli paketleri kontrol et
    print("\nGerekli paketler kontrol ediliyor...")
    required_packages = {
        "qwen_tts": "qwen-tts",
        "datasets": "datasets",
        "soundfile": "soundfile",
        "librosa": "librosa",
        "tqdm": "tqdm",
        "safetensors": "safetensors",
        "accelerate": "accelerate",
        "peft": "peft",
    }
    
    missing = []
    for module, pip_name in required_packages.items():
        try:
            __import__(module)
            print(f"  ✅ {pip_name}")
        except ImportError:
            print(f"  ❌ {pip_name} (yüklü değil)")
            missing.append(pip_name)
    
    if missing:
        print(f"\n📦 Eksik paketler yükleniyor: {', '.join(missing)}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q"
        ] + missing)
        print("✅ Tüm paketler yüklendi!")
    else:
        print("\n✅ Tüm paketler zaten yüklü!")
    
    # qwen-tts'i source'dan yükle (güncel olması için)
    qwen_tts_root = SCRIPT_DIR.parent
    pyproject = qwen_tts_root / "pyproject.toml"
    if pyproject.exists():
        print(f"\n📦 qwen-tts kaynak koddan yükleniyor: {qwen_tts_root}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "-e", str(qwen_tts_root)
        ])
        print("✅ qwen-tts güncel versiyonla yüklendi!")
    
    print("\n✅ Ortam kurulumu tamamlandı!")


def step2_convert_data():
    """FLEURS verisini Qwen format'a dönüştür"""
    print_header(f"📥 ADIM 2: FLEURS Verisi Dönüştürme ({MAX_SAMPLES} örnek)")
    
    # Eğer zaten yeterli veri varsa atla
    if RAW_JSONL.exists():
        line_count = sum(1 for _ in open(RAW_JSONL))
        if line_count >= MAX_SAMPLES:
            print(f"✅ Veri zaten mevcut: {RAW_JSONL} ({line_count} örnek)")
            print("Devam ediliyor...")
            return
        else:
            print(f"⚠️  Mevcut veri yetersiz ({line_count} örnek), yeniden dönüştürülüyor...")
    
    # Çalışma dizinlerini oluştur
    os.makedirs(FLEURS_DATA_DIR, exist_ok=True)
    
    # convert_fleurs_to_qwen_format.py çalıştır
    convert_script = SCRIPT_DIR / "convert_fleurs_to_qwen_format.py"
    cmd = [
        sys.executable, str(convert_script),
        "--language", LANGUAGE_CODE,
        "--output_dir", str(FLEURS_DATA_DIR),
        "--max_samples", str(MAX_SAMPLES),
        "--split", "train"
    ]
    
    print(f"Komut: {' '.join(cmd)}\n")
    start_time = time.time()
    subprocess.check_call(cmd)
    elapsed = time.time() - start_time
    
    # Doğrulama
    if RAW_JSONL.exists():
        line_count = sum(1 for _ in open(RAW_JSONL))
        print(f"\n✅ Dönüştürme tamamlandı! ({elapsed:.1f}s)")
        print(f"📊 Toplam örnek: {line_count}")
        print(f"📁 Dosya: {RAW_JSONL}")
    else:
        print(f"\n❌ Hata: {RAW_JSONL} oluşturulamadı!")
        sys.exit(1)


def step3_extract_codes():
    """Audio kodlarını çıkar"""
    print_header("🎵 ADIM 3: Audio Kodları Çıkarma")
    
    if not RAW_JSONL.exists():
        print(f"❌ Girdi dosyası bulunamadı: {RAW_JSONL}")
        print("Önce Adım 2'yi çalıştırın!")
        sys.exit(1)
    
    # Eğer zaten çıkarılmışsa atla
    if TRAIN_JSONL.exists():
        raw_count = sum(1 for _ in open(RAW_JSONL))
        train_count = sum(1 for _ in open(TRAIN_JSONL))
        if train_count >= raw_count:
            print(f"✅ Audio kodları zaten çıkarılmış: {TRAIN_JSONL} ({train_count} örnek)")
            print("Devam ediliyor...")
            return
        else:
            print(f"⚠️  Eksik audio kodları ({train_count}/{raw_count}), yeniden çıkarılıyor...")
    
    prepare_script = SCRIPT_DIR / "prepare_data.py"
    cmd = [
        sys.executable, str(prepare_script),
        "--device", DEVICE,
        "--tokenizer_model_path", TOKENIZER_MODEL,
        "--input_jsonl", str(RAW_JSONL),
        "--output_jsonl", str(TRAIN_JSONL)
    ]
    
    print(f"Komut: {' '.join(cmd)}\n")
    print("⏳ Bu işlem birkaç dakika sürebilir...\n")
    start_time = time.time()
    subprocess.check_call(cmd)
    elapsed = time.time() - start_time
    
    # Doğrulama
    if TRAIN_JSONL.exists():
        line_count = sum(1 for _ in open(TRAIN_JSONL))
        print(f"\n✅ Audio kodları çıkarıldı! ({elapsed:.1f}s)")
        print(f"📊 Toplam örnek: {line_count}")
        print(f"📁 Dosya: {TRAIN_JSONL}")
        
        # İlk satırı göster (kontrol amaçlı)
        with open(TRAIN_JSONL) as f:
            first = json.loads(f.readline())
            codes_len = len(first.get('audio_codes', []))
            print(f"🔢 İlk örnek audio_codes uzunluğu: {codes_len}")
    else:
        print(f"\n❌ Hata: {TRAIN_JSONL} oluşturulamadı!")
        sys.exit(1)


def step4_finetune():
    """Fine-tuning başlat (LoRA veya Full)"""
    mode_str = "LoRA" if USE_LORA else "Full"
    print_header(f"🎯 ADIM 4: {mode_str} Fine-Tuning")
    
    if not TRAIN_JSONL.exists():
        print(f"❌ Eğitim dosyası bulunamadı: {TRAIN_JSONL}")
        print("Önce Adım 2 ve 3'ü çalıştırın!")
        sys.exit(1)
    
    line_count = sum(1 for _ in open(TRAIN_JSONL))
    total_steps = (line_count // BATCH_SIZE) * NUM_EPOCHS
    lr = LORA_LR if USE_LORA else LEARNING_RATE
    
    print(f"📊 Eğitim verisi: {line_count} örnek")
    print(f"🔄 Toplam tahmini adım: ~{total_steps}")
    print(f"🔧 Mod: {mode_str}")
    
    if USE_LORA:
        output_dir = WORK_DIR / "model_output_lora"
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Çıktı dizini: {output_dir}\n")
        
        sft_script = SCRIPT_DIR / "sft_12hz_lora.py"
        cmd = [
            sys.executable, str(sft_script),
            "--init_model_path", BASE_MODEL,
            "--output_model_path", str(output_dir),
            "--train_jsonl", str(TRAIN_JSONL),
            "--batch_size", str(BATCH_SIZE),
            "--lr", str(lr),
            "--num_epochs", str(NUM_EPOCHS),
            "--speaker_name", SPEAKER_NAME,
            "--lora_rank", str(LORA_RANK),
            "--lora_alpha", str(LORA_ALPHA),
            "--merge_and_save",
        ]
    else:
        output_dir = OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Çıktı dizini: {output_dir}\n")
        
        sft_script = SCRIPT_DIR / "sft_12hz.py"
        cmd = [
            sys.executable, str(sft_script),
            "--init_model_path", BASE_MODEL,
            "--output_model_path", str(output_dir),
            "--train_jsonl", str(TRAIN_JSONL),
            "--batch_size", str(BATCH_SIZE),
            "--lr", str(lr),
            "--num_epochs", str(NUM_EPOCHS),
            "--speaker_name", SPEAKER_NAME,
        ]
    
    print(f"Komut: {' '.join(cmd)}\n")
    if USE_LORA:
        print("⏳ LoRA Fine-tuning başladı! M4 Pro'da ~15dk-1saat sürebilir...\n")
    else:
        print("⏳ Full Fine-tuning başladı! GPU'ya bağlı olarak 30dk-2saat sürebilir...\n")
    start_time = time.time()
    subprocess.check_call(cmd)
    elapsed = time.time() - start_time
    
    # Checkpoint kontrolü
    if output_dir.exists():
        checkpoints = sorted([d for d in os.listdir(output_dir) if d.startswith(("checkpoint-", "lora-checkpoint-", "merged-checkpoint-"))])
        if checkpoints:
            print(f"\n✅ {mode_str} Fine-tuning tamamlandı! ({elapsed / 60:.1f} dakika)")
            print(f"📁 Checkpoints:")
            for cp in checkpoints:
                print(f"    📦 {output_dir / cp}")
        else:
            print(f"\n⚠️  Fine-tuning tamamlandı ama checkpoint bulunamadı.")
    else:
        print(f"\n❌ Hata: Çıktı dizini oluşturulamadı!")
        sys.exit(1)


def step5_test():
    """Eğitilen modeli test et"""
    print_header("🧪 ADIM 5: Model Test")
    
    if not OUTPUT_DIR.exists():
        print(f"❌ Model çıktı dizini bulunamadı: {OUTPUT_DIR}")
        print("Önce Adım 4'ü çalıştırın!")
        sys.exit(1)
    
    # En son checkpoint'i bul
    checkpoints = sorted([d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")])
    if not checkpoints:
        print("❌ Checkpoint bulunamadı!")
        sys.exit(1)
    
    latest_cp = OUTPUT_DIR / checkpoints[-1]
    print(f"📦 Checkpoint: {latest_cp}")
    
    import torch
    import soundfile as sf
    
    print("🔄 Model yükleniyor...")
    from qwen_tts import Qwen3TTSModel
    
    tts = Qwen3TTSModel.from_pretrained(
        str(latest_cp),
        device_map=DEVICE,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )
    
    test_text = "Merhaba, ben bir yapay zeka asistanıyım. Bugün size nasıl yardımcı olabilirim?"
    print(f"📝 Test metni: {test_text}")
    print("🔄 Ses üretiliyor...")
    
    wavs, sr = tts.generate_custom_voice(
        text=test_text,
        speaker=SPEAKER_NAME,
    )
    
    output_wav = WORK_DIR / f"test_output_{LANGUAGE_CODE}.wav"
    sf.write(str(output_wav), wavs[0], sr)
    
    duration = len(wavs[0]) / sr
    print(f"\n✅ Test başarılı!")
    print(f"🎵 Çıktı dosyası: {output_wav}")
    print(f"📊 Sample rate: {sr} Hz")
    print(f"⏱️  Süre: {duration:.2f} saniye")


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS Türkçe Fine-Tuning (MSP GPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Adımlar:
  0  Tüm adımları sırayla çalıştır (varsayılan)
  1  Ortam kurulumu
  2  FLEURS veri dönüştürme (800 örnek)
  3  Audio kod çıkarma (GPU gerekli)
  4  Fine-tuning (GPU gerekli)
  5  Test/Inference (GPU gerekli)
        """
    )
    parser.add_argument("--step", type=int, default=0, choices=[0,1,2,3,4,5],
                        help="Çalıştırılacak adım numarası (0=tümü)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size geçersiz kılma")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Epoch sayısı geçersiz kılma")
    parser.add_argument("--lr", type=float, default=None,
                        help="Öğrenme oranı geçersiz kılma")
    parser.add_argument("--samples", type=int, default=None,
                        help="Örnek sayısı geçersiz kılma")
    parser.add_argument("--use_lora", type=str, default=None, choices=["true", "false"],
                        help="LoRA modunu aç/kapat (varsayılan: MPS'de açık)")
    parser.add_argument("--lora_rank", type=int, default=None,
                        help="LoRA rank geçersiz kılma")
    parser.add_argument("--lora_alpha", type=int, default=None,
                        help="LoRA alpha geçersiz kılma")
    
    args = parser.parse_args()
    
    # Geçersiz kılmalar
    global BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, MAX_SAMPLES, USE_LORA, LORA_RANK, LORA_ALPHA
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    if args.epochs is not None:
        NUM_EPOCHS = args.epochs
    if args.lr is not None:
        LEARNING_RATE = args.lr
    if args.samples is not None:
        MAX_SAMPLES = args.samples
    if args.use_lora is not None:
        USE_LORA = args.use_lora == "true"
    if args.lora_rank is not None:
        LORA_RANK = args.lora_rank
    if args.lora_alpha is not None:
        LORA_ALPHA = args.lora_alpha
    
    print_header("🚀 QWEN3-TTS TÜRKÇE FİNE-TUNİNG")
    print_config()
    
    steps = {
        1: ("Ortam Kurulumu", step1_setup),
        2: ("FLEURS Veri Dönüştürme", step2_convert_data),
        3: ("Audio Kod Çıkarma", step3_extract_codes),
        4: ("Fine-Tuning", step4_finetune),
        5: ("Test/Inference", step5_test),
    }
    
    if args.step == 0:
        # Tüm adımları çalıştır
        for step_num, (name, func) in steps.items():
            func()
    else:
        name, func = steps[args.step]
        func()
    
    print_header("🎉 İŞLEM TAMAMLANDI!")
    print(f"📁 Tüm çıktılar: {WORK_DIR}")
    if OUTPUT_DIR.exists():
        checkpoints = sorted([d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")])
        if checkpoints:
            print(f"\n🧪 Modeli test etmek için:")
            print(f"   python {__file__} --step 5")


if __name__ == "__main__":
    main()
