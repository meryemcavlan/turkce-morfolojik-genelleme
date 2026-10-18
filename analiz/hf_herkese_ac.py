# -*- coding: utf-8 -*-
"""Hugging Face depolarını herkese açık yapar (BULGU-011).

  - Sonuç deposu: meryemmm22/final-thesis-results-v5 (repo_type="dataset")
  - Ana bölünmelerde eğitilen 18 modelin ağırlık depoları (repo_type="model"), örneğin
    meryemmm22/berturk-imst-r2.18-ham-split0-seed42-v5

Colab'da:
  !pip install -q huggingface_hub
  from huggingface_hub import login; login()   # "Write" yetkili token
  !python hf_herkese_ac.py            # önce yalnızca listeler
  !python hf_herkese_ac.py --uygula   # gerçekten herkese açar
"""
import argparse

from huggingface_hub import HfApi

KULLANICI = "meryemmm22"
SONUC_DEPOSU = f"{KULLANICI}/final-thesis-results-v5"
MODEL_DEPOLARI = [f"{KULLANICI}/{m}-{v}-ham-split0-seed{t}-v5"
                  for m in ("berturk", "mbert", "xlmr")
                  for v in ("imst-r2.18", "boun-r2.10")
                  for t in (42, 23, 56)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uygula", action="store_true", help="verilmezse yalnızca mevcut durum yazdırılır")
    ap.add_argument("--modeller-gizli-kalsin", action="store_true", help="yalnız sonuç deposunu aç")
    a = ap.parse_args()
    api = HfApi()
    hedefler = [(SONUC_DEPOSU, "dataset")]
    if not a.modeller_gizli_kalsin:
        hedefler += [(r, "model") for r in MODEL_DEPOLARI]
    for repo, tur in hedefler:
        try:
            bilgi = api.repo_info(repo, repo_type=tur)
            durum = "gizli" if bilgi.private else "herkese açık"
        except Exception as e:
            print(f"✗ {repo}: okunamadı ({type(e).__name__})")
            continue
        if a.uygula and bilgi.private:
            api.update_repo_settings(repo_id=repo, repo_type=tur, private=False)
            durum = "gizli → herkese açık"
        print(f"  {tur:7s} {repo}: {durum}")


if __name__ == "__main__":
    main()
