#!/usr/bin/env python3
"""
FLEURS veri kümesini Qwen3-TTS fine-tuning formatına dönüştürür
"""

import json
import os
import argparse
from pathlib import Path
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

# FLEURS dil kodlarını Qwen3-TTS dil isimlerine eşle
FLEURS_TO_QWEN_LANGUAGE = {
    "zh_cn": "Chinese", "en_us": "English", "ja_jp": "Japanese",
    "ko_kr": "Korean", "de_de": "German", "fr_fr": "French",
    "ru_ru": "Russian", "pt_br": "Portuguese", "es_es": "Spanish",
    "it_it": "Italian", "tr_tr": "Turkish",
}

def convert_fleurs_to_qwen_format(
    language_code: str = "hi_in",
    output_dir: str = "./fleurs_data", 
    max_samples: int = 1000,
    split: str = "train",
    ref_speaker_id: str = None
):
    """
    FLEURS veri kümesini Qwen3-TTS formatına dönüştürür
    
    Args:
        language_code: FLEURS dil kodu (örn: "hi_in", "tr_tr", "en_us")
        output_dir: Çıktı dizini
        max_samples: Maksimum örnek sayısı
        split: Veri kümesi bölümü ("train", "validation", "test")
        ref_speaker_id: Referans konuşmacı ID'si (None ise ilk konuşmacıyı kullanır)
    """
    
    # Çıktı dizinlerini oluştur
    output_path = Path(output_dir)
    audio_dir = output_path / "audio"
    ref_audio_dir = output_path / "ref_audio"
    
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(ref_audio_dir, exist_ok=True)
    
    print(f"FLEURS veri kümesi yükleniyor: {language_code} ({split})")
    
    # FLEURS veri kümesini yükle
    dataset = load_dataset("google/fleurs", language_code, split=split, streaming=True) #trust_remote_code=True
    
    # Veriyi topla
    samples = []
    ref_audio_path = None
    speaker_audios = {}  # Her konuşmacının ses örnekleri
    
    print("Veri örnekleri işleniyor...")
    for i, sample in enumerate(tqdm(dataset)):
        if i >= max_samples:
            break
            
        # Ses verisi ve diğer bilgileri al
        audio_data = sample['audio']['array']
        sample_rate = sample['audio']['sampling_rate']
        text = sample['raw_transcription']
        speaker_id = str(sample.get('speaker_id', f'speaker_{i}'))
        
        # Ses dosyası yolunu oluştur
        audio_filename = f"utt_{i:06d}.wav"
        audio_path = audio_dir / audio_filename
        
        # Ses dosyasını kaydet
        sf.write(str(audio_path), audio_data, sample_rate)
        
        # Konuşmacı seslerini topla (referans için)
        if speaker_id not in speaker_audios:
            speaker_audios[speaker_id] = []
        speaker_audios[speaker_id].append(str(audio_path))
        
        # Veri örneğini kaydet
        samples.append({
            'audio_path': str(audio_path),
            'text': text,
            'speaker_id': speaker_id,
            'original_index': i
        })
        
        # İlk örneği referans olarak kullan (eğer belirtilmemişse)
        if ref_audio_path is None and (ref_speaker_id is None or speaker_id == ref_speaker_id):
            ref_audio_path = str(audio_path)
    
    # Referans ses dosyasını seç veya oluştur
    if ref_speaker_id and ref_speaker_id in speaker_audios:
        # Belirtilen konuşmacının ilk ses dosyasını kullan
        ref_audio_path = speaker_audios[ref_speaker_id][0]
    elif not ref_audio_path and speaker_audios:
        # İlk konuşmacının ilk ses dosyasını kullan
        first_speaker = list(speaker_audios.keys())[0]
        ref_audio_path = speaker_audios[first_speaker][0]
    
    # Referans ses dosyasını ref_audio dizinine kopyala
    final_ref_path = ref_audio_dir / "reference.wav"
    if ref_audio_path and os.path.exists(ref_audio_path):
        import shutil
        shutil.copy2(ref_audio_path, str(final_ref_path))
        print(f"Referans ses dosyası: {final_ref_path}")
    
    # JSONL formatına dönüştür
    jsonl_lines = []
    for sample in samples:
        qwen_language = FLEURS_TO_QWEN_LANGUAGE.get(language_code, "Auto")
        jsonl_line = {
            "audio": sample['audio_path'],
            "text": sample['text'],
            "ref_audio": str(final_ref_path),
            "language": qwen_language
        }
        jsonl_lines.append(jsonl_line)
    
    # JSONL dosyasını kaydet
    jsonl_path = output_path / f"train_raw_{language_code}.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for line in jsonl_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    
    print(f"✅ Dönüştürme tamamlandı!")
    print(f"📁 Çıktı dizini: {output_path}")
    print(f"🎵 Ses dosyaları: {len(samples)} adet")
    print(f"📝 JSONL dosyası: {jsonl_path}")
    print(f"🎤 Referans ses: {final_ref_path}")
    print(f"👥 Toplam konuşmacı sayısı: {len(speaker_audios)}")
    
    # İstatistikleri kaydet
    stats = {
        "language_code": language_code,
        "total_samples": len(samples),
        "unique_speakers": len(speaker_audios),
        "ref_audio_path": str(final_ref_path),
        "output_jsonl": str(jsonl_path)
    }
    
    stats_path = output_path / "conversion_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    return str(jsonl_path), str(final_ref_path)

def main():
    parser = argparse.ArgumentParser(description="FLEURS veri kümesini Qwen3-TTS formatına dönüştür")
    parser.add_argument("--language", type=str, default="hi_in", 
                        help="FLEURS dil kodu (örn: hi_in, tr_tr, en_us)")
    parser.add_argument("--output_dir", type=str, default="./fleurs_data",
                        help="Çıktı dizini")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Maksimum örnek sayısı")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "validation", "test"],
                        help="Veri kümesi bölümü")
    parser.add_argument("--ref_speaker_id", type=str, default=None,
                        help="Referans konuşmacı ID'si (opsiyonel)")
    
    args = parser.parse_args()
    
    convert_fleurs_to_qwen_format(
        language_code=args.language,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        split=args.split,
        ref_speaker_id=args.ref_speaker_id
    )

if __name__ == "__main__":
    main()