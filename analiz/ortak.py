# -*- coding: utf-8 -*-
"""Tez analiz betiklerinin ortak yardımcıları.

Bu dosya ana deney kodunu (v5) içe aktarmaz; yalnızca şunlara ihtiyaç duyar:
  1) Sonuç klasörü: kosular/<kosul>_seed<t>/em_records_<model>.csv.gz dosyaları
     (Hugging Face: meryemmm22/final-thesis-results-v5, repo_type="dataset").
  2) UD ağaç bankaları, ana kodun klonladığı düzende:
       <ud_kok>/r2.18/UD_Turkish-IMST/tr_imst-ud-{train,dev,test}.conllu
       <ud_kok>/r2.10/UD_Turkish-BOUN/tr_boun-ud-{train,dev,test}.conllu

Kontrollü bölünme, ana koddaki ControlledSplit.fit ile aynı algoritmadır (MD5 sıralaması,
atom korunumu, held-out oranı %15; aday eşiği IMST 20, BOUN 35).
"""
import glob
import gzip
import hashlib
import os
from collections import Counter, defaultdict

import pandas as pd

OZELLIKLER = ["Case", "Number", "Person", "Tense", "Aspect", "Mood", "Voice", "Polarity", "VerbForm"]
BASLIKLAR = ["upos", "case", "number", "person", "tense", "aspect", "mood", "voice", "polarity", "verbform"]
OZ = {h: h.capitalize() if h != "verbform" else "VerbForm" for h in BASLIKLAR[1:]}  # case → Case
MODELLER = ["BERTurk", "mBERT", "XLM-R"]
TOHUMLAR = [42, 23, 56]
KOSULLAR = [("IMST", 0), ("IMST", 1), ("IMST", 2), ("IMST", 4), ("BOUN", 0)]
VERI = {  # veri seti → (UD etiketi, depo, dosya öneki, aday eşiği)
    "IMST": ("r2.18", "UD_Turkish-IMST", "tr_imst", 20),
    "BOUN": ("r2.10", "UD_Turkish-BOUN", "tr_boun", 35),
}
HELD_OUT_ORANI = 0.15


def kosul_adi(ds, sp):
    return f"{ds} {sp}"


# --------------------------------------------------------------------------- UD verisi
def conllu_oku(yol):
    """Sözdizimsel sözcükleri okur; çok sözcüklü belirteç satırları ve boş düğümler atlanır."""
    cumleler, cur = [], None
    with open(yol, encoding="utf-8") as f:
        for satir in f:
            satir = satir.strip()
            if not satir:
                if cur and cur["form"]:
                    cumleler.append(cur)
                cur = None
                continue
            if cur is None:
                cur = {"form": [], "lemma": [], "upos": [], "feats": []}
            if satir.startswith("#"):
                continue
            a = satir.split("\t")
            if len(a) < 6 or "-" in a[0] or "." in a[0]:
                continue
            fd = {} if a[5] in ("_", "") else dict(x.split("=", 1) for x in a[5].split("|") if "=" in x)
            cur["form"].append(a[1]); cur["lemma"].append(a[2]); cur["upos"].append(a[3]); cur["feats"].append(fd)
    if cur and cur["form"]:
        cumleler.append(cur)
    return cumleler


def demet(fd):
    p = [f"{k}={fd[k]}" for k in OZELLIKLER if k in fd]
    return "|".join(sorted(p)) if p else "NONE"


def atomlar(b):
    return [] if b == "NONE" else b.split("|")


def ud_yukle(ud_kok, ds):
    tag, depo, onek, _ = VERI[ds]
    d = os.path.join(ud_kok, tag, depo)
    return {k: conllu_oku(os.path.join(d, f"{onek}-ud-{k}.conllu")) for k in ("train", "dev", "test")}


def kontrollu_bolunme(ud, ds, sp):
    """Ana koddaki ControlledSplit ile aynı: held-out kümesi, adaylar, ham sıklıklar, kontrollü eğitim."""
    esik = VERI[ds][3]
    sik = Counter(demet(fd) for s in ud["train"] + ud["dev"] for fd in s["feats"])
    aday = [b for b, f in sik.items() if f >= esik and b != "NONE"]
    aday.sort(key=lambda b: int(hashlib.md5((b + str(sp)).encode()).hexdigest(), 16))
    atom_demet = defaultdict(set)
    for b in aday:
        for a in atomlar(b):
            atom_demet[a].add(b)
    hedef = max(1, int(len(aday) * HELD_OUT_ORANI))
    ho = set()
    for b in aday:
        if len(ho) >= hedef:
            break
        if all(len(atom_demet[a] - {b} - ho) >= 1 for a in atomlar(b)):
            ho.add(b)
    tut = lambda X: [s for s in X if not any(demet(fd) in ho for fd in s["feats"])]
    return {"held_out": ho, "aday": set(aday), "sik": sik, "egitim": tut(ud["train"]), "dogrulama": tut(ud["dev"])}


def atom_sayimi(cumleler):
    return Counter(a for s in cumleler for fd in s["feats"] for a in atomlar(demet(fd)))


def upos_demet_kumesi(cumleler):
    return {u + "+" + demet(fd) for s in cumleler for u, fd in zip(s["upos"], s["feats"])}


def baglamlar(ud_kok, sonuc_kok=None):
    """Beş koşulun bölünme bağlamları. sonuc_kok verilirse held-out kümeleri veri/veri_ozeti_*.json ile sınanır."""
    import json
    out, ud_cache = {}, {}
    for ds, sp in KOSULLAR:
        ud = ud_cache.setdefault(ds, ud_yukle(ud_kok, ds))
        b = kontrollu_bolunme(ud, ds, sp)
        b["ud"] = ud
        if sonuc_kok:
            j = os.path.join(sonuc_kok, "veri", f"veri_ozeti_{ds.lower()}_split{sp}.json")
            if os.path.exists(j):
                beklenen = set(json.load(open(j, encoding="utf-8"))["held_out_demet"])
                if beklenen != b["held_out"]:
                    raise RuntimeError(f"{ds} {sp}: held-out kümesi veri_ozeti ile uyuşmuyor; UD sürümünü kontrol edin.")
        out[kosul_adi(ds, sp)] = b
    return out


# --------------------------------------------------------------------------- belirteç kayıtları
def kayitlari_yukle(sonuc_kok):
    dosyalar = sorted(glob.glob(os.path.join(sonuc_kok, "kosular", "*", "em_records_*.csv.gz")))
    if not dosyalar:
        raise FileNotFoundError(f"{sonuc_kok}/kosular/*/em_records_*.csv.gz bulunamadı.")
    parcalar = []
    for f in dosyalar:
        with gzip.open(f, "rt", encoding="utf-8") as h:
            parcalar.append(pd.read_csv(h, dtype=str, keep_default_na=False))
    D = pd.concat(parcalar, ignore_index=True)
    for c in ("is_seen", "form_in_train", "is_em"):
        D[c] = D[c] == "True"
    for c in ("seed", "split_seed", "sent_idx", "word_idx"):
        D[c] = D[c].astype(int)
    D["kosul"] = D["dataset"] + " " + D["split_seed"].astype(str)
    D["pl"] = D["pred_upos"] + "+" + D["pred_bundle"]
    return D


def tohum_ort(g, maske, f=lambda z: z["is_em"].mean()):
    """Tohum başına hesaplanan değerin üç tohum ortalaması (%)."""
    x = g[maske]
    return 100 * x.groupby("seed").apply(f).mean()


def kaydet(df, cikti, ad):
    os.makedirs(cikti, exist_ok=True)
    yol = os.path.join(cikti, ad)
    df.to_csv(yol, index=False, encoding="utf-8-sig")
    print(f"  → {yol}")
    return df
