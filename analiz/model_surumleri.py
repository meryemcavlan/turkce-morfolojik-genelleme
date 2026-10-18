# -*- coding: utf-8 -*-
"""BULGU-034: Ön eğitimli modellerin Hugging Face Hub sürümlerini (commit kimliklerini) sabitler.

Ne yapar?
  1) Üç modelin Hub'daki güncel commit kimliğini (sha) okur.
  2) Kodlayıcıyı tam o commit'ten yükler ve ana kodla AYNI yöntemle SHA-256 parmak izini hesaplar
     (pooler hariç tüm ağırlıklar; ana koddaki encoder_parmak_izi fonksiyonu).
  3) Parmak izinin ilk 12 basamağını tezdeki değerlerle karşılaştırır (Bölüm 3.13):
       BERTurk 40f6e9871672 · mBERT 802b78379678 · XLM-R ce2c4c4789d9
     Eşleşirse, deneyde kullanılan ağırlıkların bu commit'tekiyle birebir aynı olduğu kanıtlanmış olur.
  4) Sonucu model_surumleri.json dosyasına yazar (manifest klasörüne koyun).

Colab'da çalıştırın (GPU gerekmez; ~1,5 GB indirme):
  !pip install -q transformers huggingface_hub torch
  !python model_surumleri.py
"""
import hashlib
import json

import numpy as np
import torch
from huggingface_hub import model_info
from transformers import AutoModel

MODELLER = {
    "BERTurk": ("dbmdz/bert-base-turkish-cased", "40f6e9871672"),
    "mBERT": ("bert-base-multilingual-cased", "802b78379678"),
    "XLM-R": ("xlm-roberta-base", "ce2c4c4789d9"),
}


def encoder_parmak_izi(encoder):
    """Ana kod v5'teki fonksiyonun aynısı."""
    h = hashlib.sha256()
    for name, t in sorted(encoder.state_dict().items()):
        if name.startswith("pooler."):
            continue
        h.update(name.encode())
        x = t.detach().cpu()
        if x.dtype in (torch.bfloat16, torch.float16):
            x = x.float()
        arr = np.ascontiguousarray(x.numpy())
        h.update(arr.data if arr.ndim > 0 else arr.tobytes())
    return h.hexdigest()


def main():
    sonuc = {}
    for ad, (hf_id, tez_onek) in MODELLER.items():
        sha = model_info(hf_id).sha
        enc = AutoModel.from_pretrained(hf_id, revision=sha)
        pi = encoder_parmak_izi(enc)
        esit = pi[:12] == tez_onek
        sonuc[ad] = {"hf_id": hf_id, "revision": sha, "parmak_izi": pi, "tezdeki_onek": tez_onek,
                     "deneydeki_agirliklarla_ayni": esit}
        print(f"{ad:8s} {hf_id:32s} revision={sha}  parmak izi {pi[:12]}  "
              f"{'✓ tezdekiyle aynı' if esit else '✗ FARKLI — deney sonrası Hub deposu değişmiş'}")
        del enc
    with open("model_surumleri.json", "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2)
    print("→ model_surumleri.json")


if __name__ == "__main__":
    main()
