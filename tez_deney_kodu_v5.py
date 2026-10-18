# -*- coding: utf-8 -*-
"""
===============================================================================
 Türkçe Morfolojik Etiketlemede Transformer Tabanlı Modellerin Görülmemiş Özellik
 Demetlerine Genellemesinin Kontrollü Bölünmelerle Ölçülmesi — DENEY KODU v5 
===============================================================================

ÇALIŞMA MODLARI (aşağıdaki CALISMA_MODU değişkeni):
  "deney"            : Tam deney (bölünme + eğitim + değerlendirme + tüm analizler).
  "veri"             : Yalnızca veri düzeyindeki analizler (GPU gerekmez):
                       bölünme, sızıntı/atom denetimi, en-sık-etiket temel modeli,
                       D_A/D_C, BOUN anotasyon tutarlılık denetimi, OOV, E9.
  "reanaliz"         : ÖNCEKİ koşumların em_records_*.csv.gz dosyaları üzerinden
                       yeniden eğitim YAPMADAN E0–E10 analizleri.
  "hub_yeniden_uret" : em_records dosyaları kaybolduysa, Hugging Face Hub'daki
                       eski ince ayarlı modelleri indirip test tahminlerini
                       yeniden üretir, ardından "reanaliz"i çalıştırır.

NOT (Colab/Jupyter): Bu hücreyi çalışma zamanını YENİDEN BAŞLATTIKTAN sonra ilk
hücre olarak çalıştırın; aksi hâlde CUDA önceden başlatılmışsa
CUBLAS_WORKSPACE_CONFIG etkisiz kalabilir (kod bu durumu tespit edip uyarır).
===============================================================================
"""

# =============================================================================
# 0) ORTAM DEĞİŞKENLERİ — torch İÇE AKTARILMADAN ÖNCE
# =============================================================================
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import gc
import io
import re
import json
import math
import time
import glob
import random
import shutil
import hashlib
import logging
import platform
import warnings
import subprocess
import contextlib
import importlib.metadata as _ilm
from dataclasses import dataclass
from functools import partial
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Set, Tuple

_TORCH_ONCEDEN_YUKLU = "torch" in sys.modules

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# torch daha önce (başka bir hücrede) içe aktarılıp CUDA başlatıldıysa uyar.
_CUDA_ONCEDEN_BASLATILMIS = bool(
    _TORCH_ONCEDEN_YUKLU and torch.cuda.is_available() and torch.cuda.is_initialized()
)

import transformers
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    PreTrainedModel,
)

try:  # Transformers 5.x
    from transformers import PreTrainedConfig as _BaseConfig
except ImportError:  # Transformers 4.x
    from transformers import PretrainedConfig as _BaseConfig

try:
    from transformers.utils import ModelOutput
except ImportError:  # pragma: no cover
    from transformers.modeling_outputs import ModelOutput

from sklearn.metrics import accuracy_score, f1_score
from scipy import stats as sps
from scipy.stats import spearmanr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# =============================================================================
# 1) YAPILANDIRMA
# =============================================================================

KOD_SURUMU = "5.0"               # metrik/akış değişirse artırılır → eski koşum özetleri yeniden kullanılmaz

# ---- Çalışma modu -----------------------------------------------------------
CALISMA_MODU = "deney"          # "deney" | "veri" | "reanaliz" | "hub_yeniden_uret"

# ---- Çıktılar ----------------------------------------------------------------
CIKTI_DIZINI = "sonuclar_v5"     # tüm CSV/PNG/JSON/log dosyaları bu klasöre yazılır
CHECKPOINT_KOK = "checkpoints_v5"

USE_DRIVE_CHECKPOINTS = False
DRIVE_CHECKPOINT_ROOT = "/content/drive/MyDrive/checkpoints_v5"
# Sonuç klasörünün her koşum bloğundan sonra kopyalanacağı Drive klasörü.
#   "otomatik": Colab'da Drive bağlıysa (/content/drive/MyDrive) → .../MyDrive/tez_sonuclar_v5; değilse kapalı.
#   None: kapalı; ya da açık bir yol. Yerel klasör boşsa (Colab VM yenilendiyse) sonuçlar önce buradan,
#   yoksa HF sonuç deposundan GERİ YÜKLENİR ve tamamlanan koşumlar atlanır.
DRIVE_YEDEK_DIZINI: Optional[str] = "otomatik"

# ---- Universal Dependencies verisi -----------------------------------------
UD_KOK_DIZINI = "ud_data"        # <UD_KOK_DIZINI>/<etiket>/<depo> biçiminde klonlanır
UD_SURUM_ETIKETI = "r2.18"       # tezde: "UD v2.18 (GitHub etiketi r2.18)"
USE_BOUN = True
# BOUN anotasyon modu:
#   "r2.10"  : BOUN UD v2.10 — train/dev/test aynı anotasyon kurallarını izler
#              (önerilen). IMST yine r2.18'dir.
#   "uyumla" : BOUN r2.18 + yalnız TRAIN'e 5 deterministik dönüşüm;
#              kalan tutarsızlıklar anotasyon denetiminde raporlanır.
#   "ham"    : BOUN r2.18 olduğu gibi.
BOUN_ANOTASYON_MODU = "r2.10"
# True → ayrıca "BOUN_HAM" (r2.18 ham) veri seti de koşulur (karşılaştırma için).
BOUN_HAM_KARSILASTIRMA = False

# ---- Tohumlar -----------------------------------------------------------------
MULTI_SEED_MODE = True
SEEDS = [42, 23, 56]             # eğitim tohumları
DEFAULT_SEED = 42
SPLIT_SEED = 0                   # ANA bölünme tohumu (tezin ana çizelgeleri)
# Bölünme varyansı: ana bölünmeye EK bölünme sayısı (veri seti başına).
# Her ek bölünme = 3 model × len(SEEDS) koşum (IMST'de 3 ek bölünme ≈ 27 koşum ≈ 1,5–2 saat A100).
EK_SPLIT_SAYISI: Dict[str, int] = {"IMST": 3, "BOUN": 0}
# Ek bölünme tohumları aşağıdaki adaylardan SIRAYLA, YALNIZ VERİ ölçütüyle seçilir (model sonucuna
# bakılmaz): eğitim cümlelerinin en az bu oranı korunmalı ve atom korunumu sağlanmalı. Örn. IMST'de
# tohum 3 çok sık bir demeti held-out'a aldığından eğitimin %87'sini siler → otomatik reddedilir.
EK_SPLIT_ADAY_TOHUMLARI: List[int] = list(range(1, 200))
EK_SPLIT_MIN_EGITIM_KORUNUM = 0.50
# İsterseniz tohumları elle verin (verilirse yukarıdaki otomatik seçim kullanılmaz; ölçüt yine denetlenir):
EK_SPLIT_SEEDS: Dict[str, List[int]] = {}

# ---- Kontrollü bölünme ---------------------------------------------------------
HELD_OUT_RATIO = 0.15
DATASET_MIN_BUNDLE_FREQ = {"IMST": 20, "BOUN": 35, "BOUN_HAM": 35}

# ---- Eğitim hiperparametreleri (Çizelge 3.7 ile aynı) ------------------------
EARLY_STOPPING_PATIENCE = 3
# Düzeltilmiş F1(N−). Alternatif: "exact_match" (dev EM, UPOS+demet).
EARLY_STOPPING_METRIC = "avg_f1_nonone"
# Düzeltilmiş F1(N−) hesap biçimi (her iki biçim de her zaman ayrıca kaydedilir):
#   "tum_belirtecler": NONE etiket listesinden çıkarılır, TÜM belirteçler kullanılır (standart; özellik
#                      kaçırma (FN) ve NONE belirtece özellik uydurma (FP) cezalandırılır)
#   "altin_maske"    : yalnız gerçek etiketi NONE olmayan belirteçler + labels=
#                      (uydurulan özellikleri cezalandırmaz)
F1_NONONE_YONTEMI = "tum_belirtecler"
NUM_TRAIN_EPOCHS = 10
COMPLETION_MIN_EPOCHS = EARLY_STOPPING_PATIENCE + 1
MAX_SEQ_LENGTH = 128
DROPOUT_RATE = 0.1
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1               # warmup_steps = ceil(n/16) × epoch × 0,1 (int)
WEIGHT_DECAY = 0.01
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LR_SCHEDULER = "linear"          # doğrusal ısınma + doğrusal azalma
# None → Trainer varsayılanı (torch ≥ 2.8 ile "adamw_torch_fused" = AdamW).
# Kullanılan gerçek ad her koşumun özetine ve manifest'e yazılır.
OPTIMIZER: Optional[str] = None
SUBWORD_STRATEGY = "last"

# ---- İstatistik -----------------------------------------------------------------
ALPHA = 0.05
BOOTSTRAP_N_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
DA_ALPHA = 0.5                   # atom sapması (Bhattacharyya)
DC_ALPHA = 0.1                   # bileşik sapması
SPEARMAN_MIN_TOKENS = 5          # ek (filtreli) Spearman analizi için alt sınır

# ---- Hugging Face Hub --------------------------------------------------------
HF_USERNAME = "meryemmm22"
HF_MODEL_YUKLE = True            # ince ayarlı modelleri Hub'a yükle
# True → yalnız ANA bölünmenin (SPLIT_SEED) modelleri yüklenir; ek bölünmelerde belirteç kayıtları
# yeterlidir (45 model ≈ 34 GB yerine 18 model ≈ 13 GB Hub depolaması).
HF_MODEL_YUKLE_SADECE_ANA_BOLUNME = True
HF_SONUC_YUKLE = True            # sonuç klasörünü (csv, csv.gz, json, png, log) yükle
# True → depolar GİZLİ.
HF_DEPO_GIZLI = False   # sonuç deposu herkese açık (tez Bölüm 3.13)
HF_SONUC_DEPOSU = f"{HF_USERNAME}/final-thesis-results-v5"
HF_ARTIMLI_YEDEK = True          # her (veri seti, bölünme, tohum) bloğundan sonra yükle
CHECKPOINT_SIL = True            # yükleme başarılıysa ya da yükleme kapalıysa yerel checkpoint'i sil

# ---- Diğer seçenekler ------------------------------------------------------------
ATLA_TAMAMLANANLAR = True        # yarıda kalan koşumu sürdür: tamamlanan model koşumlarını atla
DETERMINIZM_TESTI = True         # aynı tohumla iki kısa eğitim (8 adım); kayıp ve parametre özetleri karşılaştırılır
BIRLESIK_ETIKET_MODELI_KOS = False   # isteğe bağlı birleşik-etiket (demet) modeli
E9_TOKENIZASYON_ANALIZI = True   # [E9] tokenizatör istatistikleri (yalnız tokenizatör dosyaları iner)
HIZLI_TEST = False               # True → küçük alt küme, 2 epoch (kodu sınamak içindir; tez için DEĞİL)
HIZLI_TEST_CUMLE = (300, 80, 200)  # HIZLI_TEST'te (train, dev, test) cümle sayısı
HIZLI_TEST_EPOCH = 2
HIZLI_TEST_YUKLEME_KAPALI = True   # HIZLI_TEST çıktıları ayrı klasöre yazılır ve Hub'a/Drive'a YÜKLENMEZ

# ---- Reanaliz / Hub'dan yeniden üretim ayarları ------------------------------------
# Yalnız TEK bir koşum ailesinin (aynı UD sürümü/anotasyon modu) kayıtlarını içeren bir klasör verin;
# farklı ailelerin kayıtları karışırsa kod durur (sessizce yanlış sayı üretmemek için).
REANALIZ_KAYIT_DIZINI = "eski_kayitlar"   # eski em_records_*.csv.gz dosyalarını buraya koyun
REANALIZ_ALT_KLASORLER = False            # True → alt klasörler de aranır (CIKTI_DIZINI hariç)
REANALIZ_ESKI_SONUC_JSON = "all_results.json"   # ESKİ koddaki dosya; E0 karşılaştırması için (yoksa atlanır)
# Eski kayıtların hangi veriyle üretildiği: veri seti → (UD etiketi, anotasyon modu, bölünme tohumu)
REANALIZ_VERI = {"IMST": ("r2.18", "ham", 0), "BOUN": ("r2.10", "ham", 0)}   # v5 kayıtları sürüm bilgisini kendisi taşır
HUB_ESKI_DEPO_SABLONU = "{user}/{short}-{ds}_seed{seed}"   # eski koddaki depo adları

# ---- Modeller ----------------------------------------------------------------------
# Ön eğitimli modellerin Hub commit kimlikleri (BULGU-034). analiz/model_surumleri.py çıktısındaki
# "revision" değerlerini yazın. None → Hub'daki güncel sürüm (deneyler bu ayarla, 13–14 Ekim 2026'da koşuldu).
MODEL_REVIZYONLARI: Dict[str, Optional[str]] = {
    "dbmdz/bert-base-turkish-cased": "b6e1de16c983e0f2c70664591ea3f22810072608",
    "bert-base-multilingual-cased": "3f076fdb1ab68d5b2880cb87a0886f315b8146f8",
    "xlm-roberta-base": "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
}


def _rev(hf_id: str) -> Optional[str]:
    return MODEL_REVIZYONLARI.get(hf_id)


MODELS = [
    {"name": "BERTurk", "hf_id": "dbmdz/bert-base-turkish-cased", "short": "berturk", "color": "#1f4e79", "hatch": ""},
    {"name": "mBERT",   "hf_id": "bert-base-multilingual-cased",  "short": "mbert",   "color": "#d9822b", "hatch": "//"},
    {"name": "XLM-R",   "hf_id": "xlm-roberta-base",              "short": "xlmr",    "color": "#4c9a5f", "hatch": ".."},
]

MORPH_HEADS = ["upos", "case", "number", "person", "tense",
               "aspect", "mood", "voice", "polarity", "verbform"]
FEAT_HEADS = [h for h in MORPH_HEADS if h != "upos"]
FEAT_KEY_MAP = {
    "case": "Case", "number": "Number", "person": "Person",
    "tense": "Tense", "aspect": "Aspect", "mood": "Mood",
    "voice": "Voice", "polarity": "Polarity", "verbform": "VerbForm",
}
FEAT_KEY_INV = {v: k for k, v in FEAT_KEY_MAP.items()}
_HEAD_ORDER = list(MORPH_HEADS)
_NUM_LABELS_ATTRS = [f"num_{h}_labels" for h in _HEAD_ORDER]
_LABEL_NAMES = [f"{h}_labels" for h in _HEAD_ORDER]


def NAME_MAP() -> Dict[str, str]:
    """Kısa ad → görünen ad (MODELS değiştirilebildiği için her çağrıda üretilir)."""
    return {m["short"]: m["name"] for m in MODELS}


# =============================================================================
# 2) GÜNLÜK (LOG), YOL YARDIMCILARI, ORTAM BİLGİSİ
# =============================================================================

logger = logging.getLogger("tez_v5")
logger.setLevel(logging.INFO)
logger.propagate = False


def _log_kur() -> None:
    """Logger'ı (yeniden) yapılandırır; Jupyter'de tekrar çalıştırmada çift kayıt oluşmaz."""
    os.makedirs(CIKTI_DIZINI, exist_ok=True)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        with contextlib.suppress(Exception):
            h.close()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(CIKTI_DIZINI, "deney.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)


def cikti_yolu(*parcalar: str) -> str:
    """CIKTI_DIZINI altında bir yol üretir ve üst klasörü oluşturur."""
    p = os.path.join(CIKTI_DIZINI, *parcalar)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    return p


class _Tee:
    """print çıktısını hem ekrana hem dosyaya yazar (konsol çıktısı tezle birlikte saklanabilsin).
    Diğer tüm öznitelikler (encoding, isatty, fileno ...) asıl akışa yönlendirilir."""

    def __init__(self, a, b):
        self._a, self._b = a, b

    def write(self, s):
        n = self._a.write(s)
        with contextlib.suppress(Exception):
            self._b.write(s)
        return n if n is not None else len(s)

    def flush(self):
        self._a.flush()
        with contextlib.suppress(Exception):
            self._b.flush()

    def __getattr__(self, item):
        return getattr(self._a, item)


@contextlib.contextmanager
def konsolu_dosyaya_kopyala(path: str):
    f = open(path, "a", encoding="utf-8")
    eski = sys.stdout
    sys.stdout = _Tee(eski, f)
    try:
        yield
    finally:
        sys.stdout = eski
        f.close()


def _surum(paket: str) -> Optional[str]:
    try:
        return _ilm.version(paket)
    except Exception:
        return None


def ortam_bilgisi() -> Dict[str, Any]:
    """Yazılım sürümleri ve donanım bilgisi."""
    info: Dict[str, Any] = {
        "zaman": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for p in ["torch", "transformers", "tokenizers", "accelerate", "safetensors",
              "huggingface_hub", "scikit-learn", "scipy", "numpy", "pandas",
              "statsmodels", "matplotlib", "seaborn"]:
        info[p] = _surum(p)
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["torch_cuda_version"] = torch.version.cuda
    if torch.cuda.is_available():
        try:
            info["gpu_adi"] = torch.cuda.get_device_name(0)
            info["gpu_bellek_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
            info["cudnn"] = torch.backends.cudnn.version()
        except Exception as e:  # pragma: no cover
            info["gpu_adi"] = f"okunamadı: {e}"
    info["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    info["cuda_ayar_oncesi_baslatilmis"] = _CUDA_ONCEDEN_BASLATILMIS
    info["colab"] = "google.colab" in sys.modules
    return info


def pip_freeze_yaz(path: str) -> None:
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                             capture_output=True, text=True, timeout=120).stdout
    except Exception as e:  # pragma: no cover
        out = f"# pip freeze alınamadı: {e}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)


def yapilandirma_ozeti() -> Dict[str, Any]:
    """Manifest için büyük harfli yapılandırma değişkenlerinin anlık görüntüsü."""
    g = globals()
    out = {}
    for k, v in g.items():
        if k.isupper() and not k.startswith("_") and isinstance(v, (int, float, str, bool, list, dict, tuple, type(None))):
            try:
                json.dumps(v)
                out[k] = v
            except TypeError:
                out[k] = str(v)
    return out


def json_yaz(obj: Any, path: str) -> None:
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    gecici = f"{path}.tmp{os.getpid()}"
    with open(gecici, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_default)
    os.replace(gecici, path)   # atomik: yarıda kesilirse eksik dosya kalmaz


def _nan_to_none(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


# ---- Türkçe sayı biçimi (şekiller için; Kılavuz §2.12) -------------------------
def tr_sayi(x: float, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{nd}f}".replace(".", ",")


def tr_yuzde(x: float, nd: int = 1) -> str:
    """0,829 → '%82,9' (Türkçe yazımda yüzde işareti sayının önüne gelir)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return "%" + f"{100 * x:.{nd}f}".replace(".", ",")


_TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})


def tr_lower(s: str) -> str:
    """Türkçeye uygun küçük harfe çevirme (I→ı, İ→i)."""
    return s.translate(_TR_LOWER).lower()


# =============================================================================
# 3) DETERMİNİZM
# =============================================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:  # pragma: no cover
        pass
    logger.info(f"Global seed set: {seed}")


def determinizm_uyarisi() -> None:
    if _CUDA_ONCEDEN_BASLATILMIS:
        msg = ("CUDA bu hücreden ÖNCE başlatılmış; CUBLAS_WORKSPACE_CONFIG etkisiz olabilir. "
               "Tam determinizm için çalışma zamanını yeniden başlatıp bu kodu ilk hücre olarak çalıştırın.")
        logger.warning(msg)
        print(f"  ⚠️  {msg}")


# =============================================================================
# 4) VERİ: UD İNDİRME (ETİKETLİ SÜRÜM), CoNLL-U OKUMA
# =============================================================================

def ud_depo_dizini(repo: str, tag: str) -> str:
    return os.path.join(UD_KOK_DIZINI, tag, repo)


def _git(args: List[str]) -> Optional[str]:
    try:
        r = subprocess.run(["git"] + args, capture_output=True, text=True, timeout=60)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def ud_depo_hazirla(repo: str, tag: str) -> Dict[str, Any]:
    """UD deposunu etiketli sürümle klonlar; commit kimliğini döndürür."""
    d = ud_depo_dizini(repo, tag)
    if not os.path.isdir(d):
        os.makedirs(os.path.dirname(d), exist_ok=True)
        url = f"https://github.com/UniversalDependencies/{repo}.git"
        cmd = ["git", "clone", "--depth", "1", "--branch", tag, url, d]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  {repo}@{tag} klonlandı → {d}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            err = getattr(e, "stderr", None) or str(e)
            raise RuntimeError(f"{repo}@{tag} klonlanamadı: {err}")
    info: Dict[str, Any] = {"repo": repo, "tag": tag, "path": d,
                            "commit": _git(["-C", d, "rev-parse", "HEAD"]),
                            "etiket_dogrulama": _git(["-C", d, "describe", "--tags", "--exact-match"])}
    if info["commit"] is None:
        logger.warning(f"  [{repo}] commit kimliği okunamadı (git yok ya da klasör elle kopyalanmış).")
    if info["etiket_dogrulama"] and info["etiket_dogrulama"] != tag:
        logger.warning(f"  [{repo}] klasördeki sürüm ({info['etiket_dogrulama']}) beklenen etiketten ({tag}) farklı!")
    logger.info(f"  UD {repo} @ {tag} commit={info['commit']}")
    print(f"  UD {repo:<16} etiket={tag:<6} commit={info['commit']}")
    return info


def parse_feats(feats_str: str) -> Dict[str, str]:
    if feats_str in ("_", ""):
        return {}
    return {f.split("=")[0]: f.split("=")[1] for f in feats_str.split("|") if "=" in f}


def _bos_cumle() -> Dict[str, Any]:
    return {"sent_id": None, "tokens": [], "lemmas": [], "upos": [], "feats": [],
            **{h: [] for h in FEAT_HEADS}}


def load_conllu(filepath: str, istatistik: Optional[Dict[str, int]] = None) -> List[Dict]:
    """UD sözdizimsel sözcüklerini (altın tokenleştirme) okur.

    Çok sözcüklü belirteç (MWT, örn. "evdeki" → "evde" + "ki") satırları ve boş
    düğümler ATLANIR ve SAYILIR; model yalnız sözdizimsel sözcükleri görür.
    Ek olarak lemma, ham FEATS sözlüğü ve sent_id saklanır.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dosya bulunamadı: {filepath}")
    st = istatistik if istatistik is not None else {}
    for k in ("cumle", "sozcuk", "mwt_satiri", "bos_dugum"):
        st.setdefault(k, 0)
    sentences: List[Dict] = []
    cur = _bos_cumle()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur["tokens"]:
                    sentences.append(cur)
                    st["cumle"] += 1
                cur = _bos_cumle()
                continue
            if line.startswith("#"):
                if line.startswith("# sent_id") and "=" in line:
                    cur["sent_id"] = line.split("=", 1)[1].strip()
                continue
            fields = line.split("\t")
            if len(fields) < 6:
                continue
            if "-" in fields[0]:
                st["mwt_satiri"] += 1
                continue
            if "." in fields[0]:
                st["bos_dugum"] += 1
                continue
            feats = parse_feats(fields[5])
            cur["tokens"].append(fields[1])
            cur["lemmas"].append(fields[2])
            cur["upos"].append(fields[3])
            cur["feats"].append(feats)
            for h in FEAT_HEADS:
                cur[h].append(feats.get(FEAT_KEY_MAP[h], "NONE"))
            st["sozcuk"] += 1
    if cur["tokens"]:
        sentences.append(cur)
        st["cumle"] += 1
    return sentences


# Geriye dönük uyumluluk
def load_multihead_data(filepath: str) -> List[Dict]:
    return load_conllu(filepath)


def _cumle_kopyala(s: Dict) -> Dict:
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in s.items()}
    out["feats"] = [dict(ft) for ft in s["feats"]]
    return out


def alt_kume(data: List[Dict], n: Optional[int]) -> List[Dict]:
    return data if (n is None or len(data) <= n) else data[:n]


# =============================================================================
# 5) BOUN ANOTASYON TUTARLILIĞI  [E7]
# =============================================================================
#  UD v2.11'den beri BOUN train'i dev/test'ten farklı
#  kurallarla etiketli. Aşağıdaki tanımlar şu sayımları üretir
#  (-(I)yor: train Imp 1 215 / Prog 45, test Prog 177; gelecek: Prosp 741 /
#  Imp 108; -ArAk: Aspect=Prog 333, özelliksiz 244; -(y)ken: özelliksiz 187;
#  iyelikli Nom/Acc %62/%38 ↔ %78/%22).

_RE_ARAK = re.compile(r"(arak|erek)$")
_RE_GENIS_KEN = re.compile(r"[aeıiuü]rken$")
_RE_IYELIK_ACC = re.compile(r"[aeıioöuü]n[ıiuü]$")
_RE_DAR_UNLU_SON = re.compile(r"[ıiuü]$")
BOUN_KURALLARI = ("iyor", "gelecek", "arak", "ken", "iyelik")


def boun_olgulari(form: str, upos: str, feats: Dict[str, str]) -> Dict[str, bool]:
    """Belirtecin hangi anotasyon olgusuna girdiğini (etiketinden bağımsız) belirler."""
    fl = tr_lower(form)
    return {
        "iyor": upos in ("VERB", "AUX") and feats.get("Tense") == "Pres" and "yor" in fl,
        "gelecek": feats.get("Tense") == "Fut",
        "arak": feats.get("VerbForm") == "Conv" and bool(_RE_ARAK.search(fl)),
        "ken": feats.get("VerbForm") == "Conv" and fl.endswith("ken"),
        "iyelik": (upos == "NOUN" and feats.get("Number[psor]") == "Sing"
                   and feats.get("Person[psor]") == "3" and feats.get("Case") in ("Nom", "Acc")),
    }


def boun_ozellik_uyumla(form: str, upos: str, feats: Dict[str, str]) -> Tuple[Dict[str, str], List[str]]:
    """r2.18 BOUN train anotasyonunu dev/test kuralına çeviren 5 deterministik dönüşüm.

    1) -(I)yor, Tense=Pres, VERB/AUX : Aspect=Imp  → Aspect=Prog   (UD belgesi de Prog öngörür)
    2) Tense=Fut                      : Aspect=Prosp → Aspect=Imp
    3) -ArAk zarf-fiili (Conv)        : Aspect=Prog/Imp ya da özelliksiz → Mood=Imp (Aspect kaldırılır)
    4) -(y)ken zarf-fiili (Conv)      : Mood yoksa Mood=Imp (+ geniş zaman -rken ise Aspect=Hab)
    5) 3. tekil iyelikli NOUN, yüzeyde yalnız -(s)I (belirtme -nI yok) : Case=Acc → Case=Nom
    UD v2.10 ile hizalanmış belirteçlerde doğrulama: kural 1 %99,6, kural 2 %87,7,
    kural 5 %98,3 uyum (3–4 için v2.10'da VerbForm=Conv olmadığından referans dev/test'tir).
    """
    ft = dict(feats)
    uyg: List[str] = []
    ol = boun_olgulari(form, upos, feats)
    fl = tr_lower(form)
    if ol["iyor"] and ft.get("Aspect") == "Imp":
        ft["Aspect"] = "Prog"
        uyg.append("iyor")
    if ol["gelecek"] and ft.get("Aspect") == "Prosp":
        ft["Aspect"] = "Imp"
        uyg.append("gelecek")
    if ol["arak"] and (ft.get("Aspect") in ("Prog", "Imp") or ft.get("Mood") != "Imp"):
        if ft.get("Aspect") in ("Prog", "Imp"):
            ft.pop("Aspect")
        ft["Mood"] = "Imp"
        uyg.append("arak")
    if ol["ken"] and "Mood" not in ft:
        ft["Mood"] = "Imp"
        if "Aspect" not in ft and _RE_GENIS_KEN.search(fl):
            ft["Aspect"] = "Hab"
        uyg.append("ken")
    if (ol["iyelik"] and ft.get("Case") == "Acc" and _RE_DAR_UNLU_SON.search(fl)
            and not _RE_IYELIK_ACC.search(fl)):
        ft["Case"] = "Nom"
        uyg.append("iyelik")
    return ft, uyg


def boun_train_uyumla(sentences: List[Dict]) -> Tuple[List[Dict], Counter]:
    """Yalnız TRAIN kümesine uygulanır (dev/test zaten hedef kuralı izler)."""
    out, sayac = [], Counter()
    for s in sentences:
        ns = _cumle_kopyala(s)
        for i in range(len(ns["tokens"])):
            yeni, uyg = boun_ozellik_uyumla(ns["tokens"][i], ns["upos"][i], ns["feats"][i])
            if uyg:
                ns["feats"][i] = yeni
                for h in FEAT_HEADS:
                    ns[h][i] = yeni.get(FEAT_KEY_MAP[h], "NONE")
                sayac.update(uyg)
        out.append(ns)
    return out, sayac


def _asp_mood(ft: Dict[str, str]) -> str:
    a, m = ft.get("Aspect"), ft.get("Mood")
    if a is None and m is None:
        return "özelliksiz"
    return "|".join(x for x in [f"Aspect={a}" if a else "", f"Mood={m}" if m else ""] if x)


def anotasyon_olgu_tablosu(ds: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for sp in ("train", "dev", "test"):
        c = {k: Counter() for k in BOUN_KURALLARI}
        adj_n = adj_feat = 0
        for s in ds[sp]:
            for form, upos, ft in zip(s["tokens"], s["upos"], s["feats"]):
                ol = boun_olgulari(form, upos, ft)
                if ol["iyor"]:
                    c["iyor"][f"Aspect={ft.get('Aspect', '–')}"] += 1
                if ol["gelecek"]:
                    c["gelecek"][f"Aspect={ft.get('Aspect', '–')}"] += 1
                if ol["arak"]:
                    c["arak"][_asp_mood(ft)] += 1
                if ol["ken"]:
                    c["ken"][_asp_mood(ft)] += 1
                if ol["iyelik"]:
                    c["iyelik"][f"Case={ft.get('Case')}"] += 1
                if upos == "ADJ":
                    adj_n += 1
                    adj_feat += int(any(FEAT_KEY_MAP[h] in ft for h in FEAT_HEADS))
        for olgu, cnt in c.items():
            top = sum(cnt.values())
            for etiket, n in cnt.most_common():
                rows.append({"kume": sp, "olgu": olgu, "etiket": etiket, "n": n,
                             "oran": n / max(1, top)})
        rows.append({"kume": sp, "olgu": "ADJ_ozellik_tasiyan", "etiket": f"{adj_feat}/{adj_n}",
                     "n": adj_feat, "oran": adj_feat / max(1, adj_n)})
    return pd.DataFrame(rows)


def form_cogunluk_uyusmazligi(train: List[Dict], diger: List[Dict], min_train: int = 5,
                              min_diger: int = 2, ornek_sayisi: int = 12) -> Dict[str, Any]:
    """Aynı sözcük biçiminin eğitimdeki çoğunluk etiketi (UPOS+demet) ile diğer kümedeki
    çoğunluk etiketinin farklı olduğu belirteçlerin oranı. Anotasyon kaymasının genel,
    kurala bağlı olmayan bir göstergesidir (IMST r2.18: ≈%1,7; BOUN r2.10: ≈%3,2;
    BOUN r2.18: ≈%7,6)."""
    tr_c: Dict[str, Counter] = defaultdict(Counter)
    ot_c: Dict[str, Counter] = defaultdict(Counter)
    for s in train:
        for i, w in enumerate(s["tokens"]):
            tr_c[tr_lower(w)][f"{s['upos'][i]}+{_bundle_of(s, i)}"] += 1
    for s in diger:
        for i, w in enumerate(s["tokens"]):
            ot_c[tr_lower(w)][f"{s['upos'][i]}+{_bundle_of(s, i)}"] += 1
    kapsanan = uyusmayan = 0
    ornekler = []
    for form, c in ot_c.items():
        n_o, n_t = sum(c.values()), sum(tr_c[form].values())
        if n_o < min_diger or n_t < min_train:
            continue
        kapsanan += n_o
        et_o, et_t = c.most_common(1)[0][0], tr_c[form].most_common(1)[0][0]
        if et_o != et_t:
            uyusmayan += n_o
            ornekler.append({"bicim": form, "n": n_o, "egitim_cogunluk": et_t, "diger_cogunluk": et_o})
    ornekler.sort(key=lambda r: -r["n"])
    return {"oran": uyusmayan / max(1, kapsanan), "kapsanan_belirtec": kapsanan,
            "uyusmayan_belirtec": uyusmayan, "ornekler": ornekler[:ornek_sayisi]}


def anotasyon_tutarlilik_denetimi(ad: str, ds: Dict[str, Any]) -> Dict[str, Any]:
    """Her veri seti için anotasyon tutarlılık raporu (IMST kontrol olarak)."""
    tablo = anotasyon_olgu_tablosu(ds)
    tablo.to_csv(cikti_yolu("veri", f"anotasyon_olgulari_{ad.lower()}.csv"), index=False, float_format="%.4f")
    tr_dev = form_cogunluk_uyusmazligi(ds["train"], ds["dev"])
    tr_test = form_cogunluk_uyusmazligi(ds["train"], ds["test"])
    SEP = "─" * 100
    print(f"\n{SEP}\n  ANOTASYON TUTARLILIK DENETİMİ — {ad}  "
          f"(UD {ds['spec']['tag']}, anotasyon={ds['spec']['anotasyon']})\n{SEP}")
    for olgu in list(BOUN_KURALLARI) + ["ADJ_ozellik_tasiyan"]:
        parca = []
        for sp in ("train", "dev", "test"):
            t = tablo[(tablo["kume"] == sp) & (tablo["olgu"] == olgu)]
            if olgu == "ADJ_ozellik_tasiyan":
                parca.append(f"{sp}: " + (f"%{100 * t['oran'].iloc[0]:.1f}" if len(t) else "–"))
            else:
                parca.append(f"{sp}: " + (", ".join(f"{r.etiket} {r.n}" for r in t.head(3).itertuples()) or "–"))
        print(f"  {olgu:<20} " + "  |  ".join(parca))
    print(f"  Biçim-çoğunluk etiket uyuşmazlığı (UPOS+demet): train→dev %{100 * tr_dev['oran']:.1f} "
          f"(n={tr_dev['kapsanan_belirtec']:,}), train→test %{100 * tr_test['oran']:.1f} "
          f"(n={tr_test['kapsanan_belirtec']:,})")
    for r in tr_test["ornekler"][:6]:
        print(f"    {r['bicim']:<14} n={r['n']:<4} eğitim: {r['egitim_cogunluk']:<55} test: {r['diger_cogunluk']}")
    if ds.get("uyumlama_degisimleri"):
        print(f"  Uyumlama ile değiştirilen train belirteçleri: {ds['uyumlama_degisimleri']}")
    print(SEP)
    ozet = {"veri_seti": ad, "ud_etiketi": ds["spec"]["tag"], "anotasyon": ds["spec"]["anotasyon"],
            "train_dev_uyusmazlik": tr_dev["oran"], "train_test_uyusmazlik": tr_test["oran"],
            "train_test_ornekler": tr_test["ornekler"],
            "uyumlama_degisimleri": ds.get("uyumlama_degisimleri", {})}
    json_yaz(ozet, cikti_yolu("veri", f"anotasyon_denetimi_{ad.lower()}.json"))
    return ozet


def veri_seti_tanimlari() -> Dict[str, Dict[str, Any]]:
    specs: Dict[str, Dict[str, Any]] = {
        "IMST": {"repo": "UD_Turkish-IMST", "onek": "tr_imst", "tag": UD_SURUM_ETIKETI, "anotasyon": "ham"},
    }
    if USE_BOUN:
        if BOUN_ANOTASYON_MODU not in ("r2.10", "uyumla", "ham"):
            raise ValueError(f"Geçersiz BOUN_ANOTASYON_MODU: {BOUN_ANOTASYON_MODU}")
        tag = "r2.10" if BOUN_ANOTASYON_MODU == "r2.10" else UD_SURUM_ETIKETI
        anot = "uyumla" if BOUN_ANOTASYON_MODU == "uyumla" else "ham"
        specs["BOUN"] = {"repo": "UD_Turkish-BOUN", "onek": "tr_boun", "tag": tag, "anotasyon": anot}
        if BOUN_HAM_KARSILASTIRMA and BOUN_ANOTASYON_MODU != "ham":
            specs["BOUN_HAM"] = {"repo": "UD_Turkish-BOUN", "onek": "tr_boun",
                                 "tag": UD_SURUM_ETIKETI, "anotasyon": "ham"}
    return specs


def veri_seti_yukle(ad: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    info = ud_depo_hazirla(spec["repo"], spec["tag"])
    out: Dict[str, Any] = {"ad": ad, "spec": dict(spec), "depo": info, "istatistik": {}}
    for sp in ("train", "dev", "test"):
        st: Dict[str, int] = {}
        out[sp] = load_conllu(os.path.join(info["path"], f"{spec['onek']}-ud-{sp}.conllu"), st)
        out["istatistik"][sp] = st
    if spec["anotasyon"] == "uyumla":
        out["train"], degisim = boun_train_uyumla(out["train"])
        out["uyumlama_degisimleri"] = dict(degisim)
    st = out["istatistik"]
    print(f"  {ad} yüklendi: {st['train']['cumle']:,} train | {st['dev']['cumle']:,} dev | "
          f"{st['test']['cumle']:,} test cümle  (MWT satırı atlandı: train {st['train']['mwt_satiri']}, "
          f"dev {st['dev']['mwt_satiri']}, test {st['test']['mwt_satiri']})")
    return out


# =============================================================================
# 6) DEMETLER (BUNDLE) VE KONTROLLÜ BÖLÜNME
# =============================================================================

def get_bundle_string(feat_dict: Dict[str, str]) -> str:
    parts = [
        f"{FEAT_KEY_MAP[head]}={feat_dict[head]}"
        for head in FEAT_HEADS
        if feat_dict.get(head, "NONE") != "NONE"
    ]
    return "|".join(sorted(parts)) if parts else "NONE"


def _bundle_of(sent: Dict, idx: int) -> str:
    return get_bundle_string({h: sent[h][idx] for h in FEAT_HEADS})


def bundle_atomlari(bundle: str) -> List[str]:
    if bundle == "NONE":
        return []
    return [a for a in bundle.split("|") if a != "NONE"]


def bundle_to_heads(bundle: str) -> Dict[str, str]:
    """'Case=Nom|Number=Sing' → {'case': 'Nom', 'number': 'Sing', diğerleri: 'NONE'}."""
    out = {h: "NONE" for h in FEAT_HEADS}
    for a in bundle_atomlari(bundle):
        if "=" in a:
            k, v = a.split("=", 1)
            if k in FEAT_KEY_INV:
                out[FEAT_KEY_INV[k]] = v
    return out


def label_to_upos_bundle(label: str) -> Tuple[str, str]:
    """'NOUN+Case=Nom|...' → ('NOUN', 'Case=Nom|...')."""
    if "+" in label:
        u, b = label.split("+", 1)
        return u, b
    return label, "NONE"


class AtomKorunumuHatasi(AssertionError):
    pass


class ControlledSplit:
    """Kompozisyonel genelleme değerlendirmesi için kontrollü bölünme
    (SPLIT_SEED=0 ile IMST 16/112, BOUN r2.10 18/125 demet)."""

    def __init__(self, held_out_ratio: float = HELD_OUT_RATIO,
                 min_bundle_freq: int = 20, seed: int = SPLIT_SEED):
        self.held_out_ratio = held_out_ratio
        self.min_bundle_freq = min_bundle_freq
        self.seed = seed
        self.held_out_bundles: Set[str] = set()
        self.split_info: Dict = {}

    def fit(self, train_data: List[Dict], dev_data: List[Dict]) -> None:
        combined = train_data + dev_data
        bundle_freq = Counter(
            _bundle_of(s, i) for s in combined for i in range(len(s["tokens"]))
        )
        candidates = [
            b for b, f in bundle_freq.items()
            if f >= self.min_bundle_freq and b != "NONE"
        ]
        candidates.sort(
            key=lambda b: int(hashlib.md5((b + str(self.seed)).encode()).hexdigest(), 16)
        )
        candidate_set = set(candidates)
        feature_to_bundles: Dict[str, Set[str]] = defaultdict(set)
        for b in candidate_set:
            for feat in b.split("|"):
                if feat != "NONE":
                    feature_to_bundles[feat].add(b)

        target_n = max(1, int(len(candidates) * self.held_out_ratio))
        held_out: Set[str] = set()
        for b in candidates:
            if len(held_out) >= target_n:
                break
            feats_in_b = [f for f in b.split("|") if f != "NONE"]
            compositional_ok = all(
                len(feature_to_bundles[feat] - {b} - held_out) >= 1
                for feat in feats_in_b
            )
            if compositional_ok:
                held_out.add(b)

        self.held_out_bundles = held_out
        self.split_info = {
            "held_out_bundles": held_out,
            "held_out_count": len(held_out),
            "total_candidates": len(candidates),
            "target_n": target_n,
            "bundle_freq": bundle_freq,
            "seed": self.seed,
        }

    def transform(self, data: List[Dict], label: str = "data", sessiz: bool = False) -> Tuple[List[Dict], List[Dict]]:
        kept, removed = [], []
        for sent in data:
            has_held_out = any(
                _bundle_of(sent, i) in self.held_out_bundles
                for i in range(len(sent["tokens"]))
            )
            (removed if has_held_out else kept).append(sent)
        if not sessiz:
            print(f"  [{label}] {len(kept)} korundu, {len(removed)} held-out.")
        return kept, removed

    def apply(self, train_data, dev_data, test_data=None, sessiz: bool = False):
        self.fit(train_data, dev_data)
        if not sessiz:
            self._print_held_out_summary()
        new_train, ho_train = self.transform(train_data, "train", sessiz)
        new_dev, ho_dev = self.transform(dev_data, "dev", sessiz)
        if test_data is not None:
            new_test = test_data
            _, held_out_test = self.transform(test_data, "test_filter", sessiz)
        else:
            new_test = ho_train + ho_dev
            held_out_test = []
        return {
            "train": new_train, "dev": new_dev, "test": new_test,
            "held_out_train": ho_train, "held_out_dev": ho_dev,
            "held_out_test": held_out_test,
        }

    def verify_no_leakage(self, train_data, dev_data, test_data: Optional[List[Dict]] = None,
                          sessiz: bool = False) -> bool:
        train_bundles = {_bundle_of(s, i) for s in train_data for i in range(len(s["tokens"]))}
        dev_bundles = {_bundle_of(s, i) for s in dev_data for i in range(len(s["tokens"]))}
        leaked = (train_bundles | dev_bundles) & self.held_out_bundles
        ok = not leaked
        if leaked:
            logger.warning(f"Sızıntı (train/dev): {len(leaked)} bundle")
            print(f"  ⚠️  UYARI: {len(leaked)} bundle train/dev'e sızdı.")
        elif not sessiz:
            print("  [OK] Train/Dev veri sızıntısı yok.")
        if test_data is not None and not sessiz:
            test_bundles = {_bundle_of(s, i) for s in test_data for i in range(len(s["tokens"]))}
            held_in_test = test_bundles & self.held_out_bundles
            print(f"  [Bilgi] Test setinde {len(held_in_test)} / {len(self.held_out_bundles)} held-out bundle "
                  f"en az bir belirteçle temsil ediliyor.")
        return ok

    def verify_atom_coverage(self, train_data: List[Dict], strict: bool = True,
                             sessiz: bool = False) -> Dict[str, int]:
        """Held-out demetlerin HER atomu bölünme SONRASI eğitim kümesinde
        (başka demetler içinde) en az bir kez geçmelidir. İhlalde AtomKorunumuHatasi."""
        atom_say = Counter()
        for s in train_data:
            for i in range(len(s["tokens"])):
                for a in bundle_atomlari(_bundle_of(s, i)):
                    atom_say[a] += 1
        gerekli = sorted({a for b in self.held_out_bundles for a in bundle_atomlari(b)})
        kapsam = {a: int(atom_say.get(a, 0)) for a in gerekli}
        eksik = [a for a, n in kapsam.items() if n == 0]
        self.split_info["atom_kapsami_egitim"] = kapsam
        if not sessiz:
            en_az = sorted(kapsam.items(), key=lambda x: x[1])[:4]
            print("  [OK] Atom korunumu (bölünme sonrası train): en az temsil edilen atomlar: "
                  + ", ".join(f"{a} {n:,}" for a, n in en_az) if not eksik else
                  f"  ⚠️  Atom korunumu İHLALİ: {eksik}")
        if eksik and strict:
            raise AtomKorunumuHatasi(f"Held-out atomları eğitimde yok: {eksik} (seed={self.seed})")
        return kapsam

    def _print_held_out_summary(self):
        info = self.split_info
        print("\n" + "═" * 62)
        print("  CONTROLLED SPLIT ÖZETI")
        print("═" * 62)
        print(f"  Seed                   : {info['seed']}")
        print(f"  Held-out ratio         : {self.held_out_ratio:.0%}")
        print(f"  Min bundle freq        : {self.min_bundle_freq}")
        print(f"  Toplam aday bundle     : {info['total_candidates']}")
        print(f"  Held-out bundle sayısı : {info['held_out_count']}  (hedef ⌊aday×oran⌋ = {info['target_n']})")
        atom_counts = Counter(
            a for b in info["held_out_bundles"] for a in b.split("|") if a != "NONE"
        )
        print("  Held-out atom dağılımı :")
        if atom_counts:
            for atom, cnt in atom_counts.most_common():
                print(f"    {atom:<24} {cnt:>3}")
        else:
            print("    (atom yok)")
        print("═" * 62 + "\n")
        self.split_info["held_out_atom_counts"] = dict(atom_counts)


def build_controlled_split(train_data, dev_data, test_data=None,
                           held_out_ratio=HELD_OUT_RATIO, min_bundle_freq=20,
                           split_seed=SPLIT_SEED, atom_strict: bool = True, sessiz: bool = False):
    splitter = ControlledSplit(held_out_ratio=held_out_ratio,
                               min_bundle_freq=min_bundle_freq, seed=split_seed)
    splits = splitter.apply(train_data, dev_data, test_data, sessiz=sessiz)
    ok = splitter.verify_no_leakage(splits["train"], splits["dev"], test_data, sessiz=sessiz)
    if not ok:
        raise AssertionError("Held-out demetler train/dev kümesine sızdı.")
    splitter.verify_atom_coverage(splits["train"], strict=atom_strict, sessiz=sessiz)
    return splits, splitter


# =============================================================================
# 7) VERİ DÜZEYİ ANALİZLER (eğitim gerektirmez)
# =============================================================================

def analyze_bundle_distribution(train_data, test_data, held_out_bundles: Set[str] = None,
                                sessiz: bool = False) -> Dict:
    """Test demetleri üç gruba ayrılır:
    held-out (Unseen) · held-out olmayan ve eğitimde geçen · held-out olmayan ama bölünme
    sonrası eğitimde HİÇ geçmeyen."""
    train_bc = Counter(_bundle_of(s, i) for s in train_data for i in range(len(s["tokens"])))
    test_bc = Counter(_bundle_of(s, i) for s in test_data for i in range(len(s["tokens"])))
    ho = held_out_bundles or set()
    unseen = {b for b in test_bc if b in ho}
    non_ho = set(test_bc) - unseen
    seen_train = {b for b in non_ho if b in train_bc}
    seen_novel = non_ho - seen_train
    total = sum(test_bc.values())
    tok = lambda bs: sum(test_bc[b] for b in bs)
    res = {
        "train_bundle_counts": train_bc, "test_bundle_counts": test_bc,
        "train_bundles": set(train_bc), "unseen_in_test": unseen,
        "seen_in_test": non_ho, "seen_train_in_test": seen_train, "seen_novel_in_test": seen_novel,
        "total_test_tokens": total, "unseen_test_tokens": tok(unseen),
        "seen_train_tokens": tok(seen_train), "seen_novel_tokens": tok(seen_novel),
        "both_train_and_test": len(set(test_bc) & set(train_bc)),
    }
    if not sessiz:
        print("\n" + "═" * 78 + "\n  DEMET DAĞILIMI ANALİZİ\n" + "═" * 78)
        print(f"  Eğitimdeki benzersiz demet                          : {len(train_bc)}")
        print(f"  Testteki benzersiz demet                            : {len(test_bc)}")
        print(f"  Held-out olmayan test demeti ('Seen' grubu)         : {len(non_ho)}")
        print(f"    ├─ bölünme sonrası eğitimde de geçen              : {len(seen_train)}  ({tok(seen_train):,} belirteç)")
        print(f"    └─ held-out değil ama eğitimde HİÇ geçmeyen       : {len(seen_novel)}  ({tok(seen_novel):,} belirteç)")
        print(f"  Held-out (Unseen) test demeti                       : {len(unseen)}  ({tok(unseen):,} belirteç)")
        print(f"  Hem eğitimde hem testte bulunan demet               : {res['both_train_and_test']}")
        print(f"  Test toplam belirteç                                : {total:,}")
        print(f"  Seen / Unseen belirteç                              : {total - tok(unseen):,} / {tok(unseen):,}"
              f"  (%{100 * tok(unseen) / max(1, total):.1f} Unseen)")
    return res


def compute_majority_class_baseline(test_data, train_data, held_out_bundles, sessiz: bool = False) -> Dict:
    """[E10] En-sık-etiket temel modeli.

    İki sürüm raporlanır: (i) yalnız demet (FEATS); (ii) UPOS + demet —
    model EM'i UPOS'u da içerdiği için YALNIZ bu sürüm model EM'iyle kıyaslanabilir.
    NOT: Held-out demetler eğitimde bulunmadığından bu temel modelin Unseen EM'i TANIM
    GEREĞİ %0'dır; bu bir 'deneysel doğrulama' değildir."""
    tr_b = Counter(_bundle_of(s, i) for s in train_data for i in range(len(s["tokens"])))
    tr_ub = Counter(f"{s['upos'][i]}+{_bundle_of(s, i)}" for s in train_data for i in range(len(s["tokens"])))
    if not tr_b:
        return {}
    maj_b, _ = tr_b.most_common(1)[0]
    maj_ub, _ = tr_ub.most_common(1)[0]
    c = Counter()
    for s in test_data:
        for i in range(len(s["tokens"])):
            b = _bundle_of(s, i)
            ub = f"{s['upos'][i]}+{b}"
            grp = "unseen" if b in held_out_bundles else "seen"
            c[f"{grp}_n"] += 1
            c[f"{grp}_b"] += int(b == maj_b)
            c[f"{grp}_ub"] += int(ub == maj_ub)
    res = {
        "majority_bundle": maj_b, "majority_upos_bundle": maj_ub,
        "seen_em_bundle_only": c["seen_b"] / max(1, c["seen_n"]),
        "unseen_em_bundle_only": c["unseen_b"] / max(1, c["unseen_n"]),
        "seen_em_upos_bundle": c["seen_ub"] / max(1, c["seen_n"]),
        "unseen_em_upos_bundle": c["unseen_ub"] / max(1, c["unseen_n"]),
        "overall_em_upos_bundle": (c["seen_ub"] + c["unseen_ub"]) / max(1, c["seen_n"] + c["unseen_n"]),
        "seen_n": c["seen_n"], "unseen_n": c["unseen_n"],
    }
    if not sessiz:
        print(f"  [En-sık-etiket temel modeli] yalnız demet ({maj_b!r}): Seen EM = {res['seen_em_bundle_only']:.2%}")
        print(f"  [En-sık-etiket temel modeli] UPOS+demet ({maj_ub!r}): Seen EM = {res['seen_em_upos_bundle']:.2%}"
              f"  ← model EM'iyle kıyaslanabilir olan budur")
        print(f"  [En-sık-etiket temel modeli] Unseen EM = {res['unseen_em_upos_bundle']:.2%} "
              f"(tanım gereği %0; deneysel bir bulgu değildir)")
    return res


def _chernoff_coefficient(p_counts, q_counts, alpha=0.1):
    """Keysers vd. (2020), §2.1: C_α(P‖Q) = Σ_w P(w)^α · Q(w)^(1−α)."""
    p_total = sum(p_counts.values()) or 1
    q_total = sum(q_counts.values()) or 1
    c = 0.0
    for k in set(p_counts) | set(q_counts):
        p_w = p_counts.get(k, 0) / p_total
        q_w = q_counts.get(k, 0) / q_total
        if p_w > 0 and q_w > 0:
            c += (p_w ** alpha) * (q_w ** (1 - alpha))
    return c


def _atom_sayimlari(data) -> Counter:
    c = Counter()
    for s in data:
        for i in range(len(s["tokens"])):
            for a in bundle_atomlari(_bundle_of(s, i)):
                c[a] += 1
    return c


def _bilesik_sayimlari(data) -> Counter:
    return Counter(_bundle_of(s, i) for s in data for i in range(len(s["tokens"])))


def compute_atom_and_compound_divergence(train_data, test_data, alpha_atom: float = None,
                                         alpha_compound: float = None, sessiz: bool = False,
                                         etiket: str = "") -> Dict[str, float]:
    """Keysers vd. (2020, §2.1): D_A = 1 − C_0,5 (Bhattacharyya), D_C = 1 − C_0,1.
    Bölünmeyi OPTİMİZE ETMEZ; oluşturulmuş bölünmenin sapmasını geriye dönük ölçer."""
    a_at = DA_ALPHA if alpha_atom is None else alpha_atom
    a_co = DC_ALPHA if alpha_compound is None else alpha_compound
    d_atom = 1.0 - _chernoff_coefficient(_atom_sayimlari(train_data), _atom_sayimlari(test_data), alpha=a_at)
    d_comp = 1.0 - _chernoff_coefficient(_bilesik_sayimlari(train_data), _bilesik_sayimlari(test_data), alpha=a_co)
    if not sessiz:
        print(f"  [DBCA-esinli sapma{(' — ' + etiket) if etiket else ''}] D_A (α={a_at}) = {d_atom:.4f}  |  "
              f"D_C (α={a_co}) = {d_comp:.4f}")
    return {"atom_divergence": d_atom, "compound_divergence": d_comp,
            "alpha_atom": a_at, "alpha_compound": a_co}


def dbca_sapma_raporu(ad: str, ham: Dict[str, Any], splits: Dict[str, List[Dict]],
                      held_out: Set[str], split_seed: int) -> pd.DataFrame:
    """Üç karşılaştırma: (1) kontrollü train ↔ tüm test; (2) UD'nin kendi train ↔ test
    bölünmesi (referans); (3) kontrollü train ↔ yalnız Unseen belirteçler (D_C = 1 beklenir)."""
    unseen_only = []
    for s in splits["test"]:
        idx = [i for i in range(len(s["tokens"])) if _bundle_of(s, i) in held_out]
        if idx:
            unseen_only.append({**{h: [s[h][i] for i in idx] for h in FEAT_HEADS},
                                "tokens": [s["tokens"][i] for i in idx]})
    satirlar = []
    for kiyas, tr, te in [("kontrollu_train__test", splits["train"], splits["test"]),
                          ("ud_train__ud_test", ham["train"], ham["test"]),
                          ("kontrollu_train__unseen", splits["train"], unseen_only)]:
        r = compute_atom_and_compound_divergence(tr, te, sessiz=False, etiket=f"{ad} {kiyas}")
        satirlar.append({"veri_seti": ad, "split_seed": split_seed, "karsilastirma": kiyas, **r})
    df = pd.DataFrame(satirlar)
    df.to_csv(cikti_yolu("veri", f"dbca_sapma_{ad.lower()}_split{split_seed}.csv"), index=False, float_format="%.5f")
    print("  NOT: Karşılaştırma için CFQ rastgele bölünme D_C ≈ 0, MCD bölünmeleri D_C ≈ 0,69–0,71 "
          "(Keysers vd. 2020). Kompozisyonel zorluk yalnız Unseen alt kümesindedir.")
    return df


def heldout_yogunlasma_raporu(ad: str, splits, splitter: ControlledSplit, split_seed: int,
                              ham_train: List[Dict], sessiz: bool = False) -> pd.DataFrame:
    """Held-out demetlerin eğitim+dev frekansı, testteki belirteç sayısı ve payı,
    eğitimden kaç cümleyi düşürdüğü; frekans eşiğinin testte karşılığı."""
    held_out = splitter.held_out_bundles
    freq = splitter.split_info["bundle_freq"]
    test_c = Counter(_bundle_of(s, i) for s in splits["test"] for i in range(len(s["tokens"]))
                     if _bundle_of(s, i) in held_out)
    cumle_dusen = Counter()
    for s in ham_train:
        bs = {_bundle_of(s, i) for i in range(len(s["tokens"]))} & held_out
        for b in bs:
            cumle_dusen[b] += 1
    toplam_unseen = sum(test_c.values())
    rows = []
    for b in sorted(held_out, key=lambda x: -test_c.get(x, 0)):
        rows.append({"bundle": b, "trainDev_frekans": freq.get(b, 0), "test_belirtec": test_c.get(b, 0),
                     "unseen_payi": test_c.get(b, 0) / max(1, toplam_unseen),
                     "icerdigi_egitim_cumlesi": cumle_dusen.get(b, 0),
                     "egitim_cumle_payi": cumle_dusen.get(b, 0) / max(1, len(ham_train)),
                     "atom_sayisi": len(bundle_atomlari(b))})
    df = pd.DataFrame(rows)
    df.to_csv(cikti_yolu("veri", f"heldout_yogunlasma_{ad.lower()}_split{split_seed}.csv"),
              index=False, float_format="%.4f")
    if not sessiz and len(df):
        paylar = df["unseen_payi"].tolist()
        print(f"  [Unseen yoğunlaşması] en büyük demet payı %{100 * paylar[0]:.1f}, ilk üç %{100 * sum(paylar[:3]):.1f}; "
              f"testte hiç örneği olmayan held-out demet: {(df['test_belirtec'] == 0).sum()} / {len(df)}; "
              f"test frekans aralığı {df['test_belirtec'].min()}–{df['test_belirtec'].max()}")
        b0 = df.iloc[0]
        print(f"    En baskın demet: {b0['bundle']}  ({int(b0['test_belirtec'])} test belirteci; "
              f"{int(b0['icerdigi_egitim_cumlesi'])} eğitim cümlesinde geçiyor → eğitimden atıldı)")
    return df


def oov_istatistikleri(ad: str, splits, held_out: Set[str], split_seed: int, sessiz: bool = False) -> Dict[str, float]:
    """Test belirtecinin yüzey biçimi (Türkçe küçük harf) bölünme sonrası eğitimde geçiyor mu?"""
    formlar = {tr_lower(w) for s in splits["train"] for w in s["tokens"]}
    c = Counter()
    for s in splits["test"]:
        for i, w in enumerate(s["tokens"]):
            g = "unseen" if _bundle_of(s, i) in held_out else "seen"
            c[f"{g}_n"] += 1
            c[f"{g}_iv"] += int(tr_lower(w) in formlar)
    res = {"seen_bicim_egitimde_oran": c["seen_iv"] / max(1, c["seen_n"]),
           "unseen_bicim_egitimde_oran": c["unseen_iv"] / max(1, c["unseen_n"]),
           "seen_n": c["seen_n"], "unseen_n": c["unseen_n"]}
    if not sessiz:
        print(f"  [Biçim görülme] Eğitimde yüzey biçimi geçen test belirteci: Seen %{100 * res['seen_bicim_egitimde_oran']:.1f}, "
              f"Unseen %{100 * res['unseen_bicim_egitimde_oran']:.1f}  → Unseen, çoğunlukla görülmemiş SÖZCÜK de içerir")
    json_yaz({"veri_seti": ad, "split_seed": split_seed, **res},
             cikti_yolu("veri", f"bicim_gorulme_{ad.lower()}_split{split_seed}.json"))
    return res


def compute_heterogeneity(bundle, train_bundles):
    """Eğitime yakınlık: held-out demetin eğitimdeki en yakın demete Jaccard/Hamming mesafesi.
    Eşitliklerde en yakın komşu DETERMİNİSTİK (sıralı küme üzerinde ilk bulunan)."""
    if bundle == "NONE":
        return {}
    atoms_b = set(bundle.split("|")) - {"NONE"}
    best_jaccard, best_hamming, nearest_seen = 0.0, len(atoms_b) + 100, "NONE"
    for tb in sorted(train_bundles):
        if tb == "NONE":
            continue
        atoms_t = set(tb.split("|")) - {"NONE"}
        union = len(atoms_b | atoms_t)
        jaccard = len(atoms_b & atoms_t) / union if union else 0.0
        hamming = len(atoms_b.symmetric_difference(atoms_t))
        if jaccard > best_jaccard or (jaccard == best_jaccard and hamming < best_hamming):
            best_jaccard, best_hamming, nearest_seen = jaccard, hamming, tb
    return {"kombinasyon_karmasikligi": len(atoms_b), "jaccard_max": round(best_jaccard, 4),
            "hamming_min": best_hamming, "nearest_seen": nearest_seen}


def heterogeneity_report(test_data, train_data, held_out_bundles, dataset_name="", split_seed: int = SPLIT_SEED):
    """'Heterojenlik' raporu; ölçülen şey aslında EĞİTİME YAKINLIKTIR."""
    train_bundles = {_bundle_of(s, i) for s in train_data for i in range(len(s["tokens"]))}
    test_bundle_freq = Counter(_bundle_of(s, i) for s in test_data for i in range(len(s["tokens"])))
    rows = []
    for b in sorted(held_out_bundles):
        h = compute_heterogeneity(b, train_bundles)
        if not h:
            continue
        rows.append({"bundle": b, "test_freq": test_bundle_freq.get(b, 0), **h})
    df = pd.DataFrame(rows, columns=["bundle", "test_freq", "kombinasyon_karmasikligi",
                                     "jaccard_max", "hamming_min", "nearest_seen"])
    if len(df):
        df = df.sort_values(["jaccard_max", "bundle"], ascending=[False, True], kind="mergesort")
    csv_path = cikti_yolu("veri", f"egitime_yakinlik_{(dataset_name or 'dataset').lower()}_split{split_seed}.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  Eğitime yakınlık (heterojenlik) raporu '{csv_path}' kaydedildi "
          f"(farklı J_max değeri: {df['jaccard_max'].nunique() if len(df) else 0}).")
    return df


def veri_duzeyi_analizler(ad: str, ham: Dict[str, Any], splits, splitter: ControlledSplit,
                          split_seed: int) -> Dict[str, Any]:
    """Bir (veri seti, bölünme) için eğitim gerektirmeyen tüm analizler."""
    held = splitter.held_out_bundles
    print(f"\n  ── Veri düzeyi analizler: {ad}, split_seed={split_seed} ──")
    dag = analyze_bundle_distribution(splits["train"], splits["test"], held)
    temel = compute_majority_class_baseline(splits["test"], splits["train"], held)
    dbca = dbca_sapma_raporu(ad, ham, splits, held, split_seed)
    yog = heldout_yogunlasma_raporu(ad, splits, splitter, split_seed, ham["train"])
    oov = oov_istatistikleri(ad, splits, held, split_seed)
    kayip_train = 1 - len(splits["train"]) / max(1, len(ham["train"]))
    kayip_dev = 1 - len(splits["dev"]) / max(1, len(ham["dev"]))
    ozet = {
        "veri_seti": ad, "split_seed": split_seed, "ud_etiketi": ham["spec"]["tag"],
        "anotasyon": ham["spec"]["anotasyon"], "ud_commit": ham["depo"]["commit"],
        "cumle_train_dev_test_ham": [len(ham["train"]), len(ham["dev"]), len(ham["test"])],
        "cumle_train_dev_bolunme_sonrasi": [len(splits["train"]), len(splits["dev"])],
        "train_kayip_orani": kayip_train, "dev_kayip_orani": kayip_dev,
        "aday_demet": splitter.split_info["total_candidates"],
        "held_out_demet": sorted(held), "held_out_sayisi": len(held),
        "testte_temsil_edilen_held_out": len(dag["unseen_in_test"]),
        "atom_kapsami_egitim": splitter.split_info.get("atom_kapsami_egitim", {}),
        "egitimdeki_benzersiz_demet": len(dag["train_bundles"]),
        "etiket_sayilari_NONE_dahil": {h: len(v) for h, v in build_label_mappings(splits["train"], sessiz=True).items()},
        "seen_belirtec": dag["total_test_tokens"] - dag["unseen_test_tokens"],
        "unseen_belirtec": dag["unseen_test_tokens"],
        "seen_egitimde_olmayan_demet": len(dag["seen_novel_in_test"]),
        "seen_egitimde_olmayan_belirtec": dag["seen_novel_tokens"],
        "temel_model": temel, "dbca": dbca.to_dict(orient="records"), "bicim_gorulme": oov,
        "mwt_istatistik": ham["istatistik"],
        "unseen_en_buyuk_demet_payi": float(yog["unseen_payi"].iloc[0]) if len(yog) else None,
        "unseen_ilk3_payi": float(yog["unseen_payi"].head(3).sum()) if len(yog) else None,
    }
    print(f"  Eğitimden atılan cümle: train %{100 * kayip_train:.1f}, dev %{100 * kayip_dev:.1f}")
    json_yaz(ozet, cikti_yolu("veri", f"veri_ozeti_{ad.lower()}_split{split_seed}.json"))
    return ozet


# =============================================================================
# 8) MODEL: ÇOK-BAŞLI MORFOLOJİK ETİKETLEYİCİ (+ isteğe bağlı birleşik-etiket)
# =============================================================================
import inspect


@dataclass
class MultiHeadMorphOutput(ModelOutput):
    """Çok-başlı modelin çıktısı (alan sırası = _HEAD_ORDER; Trainer logit sırasını buradan alır)."""
    loss: Optional[torch.FloatTensor] = None
    upos_logits: Optional[torch.FloatTensor] = None
    case_logits: Optional[torch.FloatTensor] = None
    number_logits: Optional[torch.FloatTensor] = None
    person_logits: Optional[torch.FloatTensor] = None
    tense_logits: Optional[torch.FloatTensor] = None
    aspect_logits: Optional[torch.FloatTensor] = None
    mood_logits: Optional[torch.FloatTensor] = None
    voice_logits: Optional[torch.FloatTensor] = None
    polarity_logits: Optional[torch.FloatTensor] = None
    verbform_logits: Optional[torch.FloatTensor] = None


@dataclass
class BundleTagOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    upos_logits: Optional[torch.FloatTensor] = None
    bundle_logits: Optional[torch.FloatTensor] = None


class MultiHeadConfig(_BaseConfig):
    model_type = "multi_head_morph"

    def __init__(self,
                 base_model_name: str = "dbmdz/bert-base-turkish-cased",
                 num_upos_labels: int = 17,
                 num_case_labels: int = 8,
                 num_number_labels: int = 3,
                 num_person_labels: int = 4,
                 num_tense_labels: int = 6,
                 num_aspect_labels: int = 5,
                 num_mood_labels: int = 7,
                 num_voice_labels: int = 5,
                 num_polarity_labels: int = 3,
                 num_verbform_labels: int = 6,
                 dropout: float = DROPOUT_RATE,
                 head_init_std: Optional[float] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.num_upos_labels = num_upos_labels
        self.num_case_labels = num_case_labels
        self.num_number_labels = num_number_labels
        self.num_person_labels = num_person_labels
        self.num_tense_labels = num_tense_labels
        self.num_aspect_labels = num_aspect_labels
        self.num_mood_labels = num_mood_labels
        self.num_voice_labels = num_voice_labels
        self.num_polarity_labels = num_polarity_labels
        self.num_verbform_labels = num_verbform_labels
        self.dropout = dropout
        self.head_init_std = head_init_std


class BundleTagConfig(_BaseConfig):
    model_type = "bundle_tag_morph"

    def __init__(self, base_model_name: str = "dbmdz/bert-base-turkish-cased",
                 num_upos_labels: int = 17, num_bundle_labels: int = 100,
                 dropout: float = DROPOUT_RATE, head_init_std: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.num_upos_labels = num_upos_labels
        self.num_bundle_labels = num_bundle_labels
        self.dropout = dropout
        self.head_init_std = head_init_std


def encoder_parmak_izi(encoder: nn.Module) -> str:
    """Kodlayıcının tüm ağırlıklarının SHA-256 özeti (kullanılmayan pooler hariç)."""
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


def _safe_encoder_kwargs(encoder, base_kwargs: dict) -> dict:
    out = {}
    if "token_type_ids" in base_kwargs and base_kwargs["token_type_ids"] is not None:
        if getattr(encoder.config, "type_vocab_size", 0) > 1:
            out["token_type_ids"] = base_kwargs["token_type_ids"]
    return out


class _KodlayiciBaslikMixin:
    """Ortak davranış: önceden eğitilmiş kodlayıcı + doğrusal başlıklar.
    post_init() SONRASI kodlayıcı parmak izi karşılaştırılır; başlıklar
    N(0; initializer_range) ve sıfır sapmayla başlatılır ve bu raporlanır."""

    def _kur(self, config, head_sizes: Dict[str, int], encoder: Optional[nn.Module] = None):
        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(config.base_model_name, revision=_rev(config.base_model_name))
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.classifiers = nn.ModuleDict({h: nn.Linear(hidden, int(n)) for h, n in head_sizes.items()})
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        parmak_once = encoder_parmak_izi(self.encoder)
        self.post_init()
        self.init_raporu = self._baslatma_denetimi(parmak_once, config)

    def _head_init_std(self) -> float:
        std = getattr(self.config, "head_init_std", None)
        if std is None:
            enc = getattr(self, "encoder", None)
            std = getattr(getattr(enc, "config", None), "initializer_range", None)
        return float(std) if std is not None else 0.02

    @torch.no_grad()
    def _init_weights(self, module):
        """YALNIZ görev başlıklarını başlatır; önceden eğitilmiş kodlayıcıya dokunmaz.
        (torch.nn.init kullanılır: Transformers'ın 'zaten başlatıldı' işaretlerine uyar.)"""
        heads = getattr(self, "classifiers", None)
        if heads is None or not isinstance(module, nn.Linear):
            return
        if any(module is m for m in heads.values()):
            nn.init.normal_(module.weight, mean=0.0, std=self._head_init_std())
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _baslatma_denetimi(self, parmak_once: str, config) -> Dict[str, Any]:
        parmak_sonra = encoder_parmak_izi(self.encoder)
        rapor: Dict[str, Any] = {"transformers": transformers.__version__,
                                 "kodlayici_parmak_izi_once": parmak_once,
                                 "kodlayici_parmak_izi_sonra": parmak_sonra,
                                 "kodlayici_degismedi": parmak_once == parmak_sonra,
                                 "hedef_std": self._head_init_std()}
        if not rapor["kodlayici_degismedi"]:
            logger.warning("post_init() kodlayıcı ağırlıklarını değiştirdi — önceden eğitilmiş kodlayıcı yeniden yükleniyor.")
            self.encoder = AutoModel.from_pretrained(config.base_model_name, revision=_rev(config.base_model_name))
            rapor["kodlayici_yeniden_yuklendi"] = True
            rapor["kodlayici_parmak_izi_sonra"] = encoder_parmak_izi(self.encoder)
            rapor["kodlayici_degismedi"] = rapor["kodlayici_parmak_izi_sonra"] == parmak_once
            if not rapor["kodlayici_degismedi"]:
                raise RuntimeError("Önceden eğitilmiş kodlayıcı ağırlıkları korunamadı (parmak izi uyuşmuyor).")
        hedef = rapor["hedef_std"]

        def _ozet():
            return {h: {"ortalama": float(l.weight.detach().float().mean()),
                        "std": float(l.weight.detach().float().std()),
                        "bias_sifir": bool(torch.all(l.bias.detach() == 0))}
                    for h, l in self.classifiers.items()}

        bas = _ozet()
        uygun = all(v["bias_sifir"] and abs(v["std"] - hedef) < 0.5 * hedef for v in bas.values())
        if not uygun:
            with torch.no_grad():
                for lin in self.classifiers.values():
                    lin.weight.normal_(mean=0.0, std=hedef)
                    lin.bias.zero_()
            rapor["basliklar_acikca_yeniden_baslatildi"] = True
            bas = _ozet()
        rapor["basliklar"] = bas
        return rapor

    def _kayip(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        lbl = labels.view(-1)
        if not bool((lbl != -100).any()):
            return logits.sum() * 0.0   # tümü yok sayılmışsa NaN yerine 0
        return self.loss_fn(logits.view(-1, logits.size(-1)), lbl)


class MultiHeadMorphModel(_KodlayiciBaslikMixin, PreTrainedModel):
    config_class = MultiHeadConfig

    def __init__(self, config: MultiHeadConfig, encoder: Optional[nn.Module] = None):
        super().__init__(config)
        self._kur(config, {h: getattr(config, f"num_{h}_labels") for h in _HEAD_ORDER}, encoder)

    def forward(self, input_ids, attention_mask,
                upos_labels=None, case_labels=None, number_labels=None, person_labels=None,
                tense_labels=None, aspect_labels=None, mood_labels=None, voice_labels=None,
                polarity_labels=None, verbform_labels=None, **kwargs) -> MultiHeadMorphOutput:
        encoder_kwargs = _safe_encoder_kwargs(self.encoder, kwargs)
        seq = self.dropout(self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                        **encoder_kwargs).last_hidden_state)
        logit_tensors = {h: self.classifiers[h](seq) for h in _HEAD_ORDER}
        all_labels = [upos_labels, case_labels, number_labels, person_labels, tense_labels,
                      aspect_labels, mood_labels, voice_labels, polarity_labels, verbform_labels]
        total_loss = None
        if all(lbl is not None for lbl in all_labels):
            # Denklem 3.3: 10 başlığın çapraz entropi kayıplarının eşit ağırlıklı ortalaması.
            losses = [self._kayip(logit_tensors[h], lbl) for h, lbl in zip(_HEAD_ORDER, all_labels)]
            total_loss = sum(losses) / len(losses)
        return MultiHeadMorphOutput(loss=total_loss, **{f"{h}_logits": logit_tensors[h] for h in _HEAD_ORDER})


class BundleTagModel(_KodlayiciBaslikMixin, PreTrainedModel):
    """Birleşik-etiket (demet) modeli: UPOS başlığı + eğitimdeki demetler üzerinde tek softmax.
    Held-out demetler eğitimde bulunmadığından Unseen EM tanım gereği %0'dır."""
    config_class = BundleTagConfig

    def __init__(self, config: BundleTagConfig, encoder: Optional[nn.Module] = None):
        super().__init__(config)
        self._kur(config, {"upos": config.num_upos_labels, "bundle": config.num_bundle_labels}, encoder)

    def forward(self, input_ids, attention_mask, upos_labels=None, bundle_labels=None, **kwargs) -> BundleTagOutput:
        encoder_kwargs = _safe_encoder_kwargs(self.encoder, kwargs)
        seq = self.dropout(self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                        **encoder_kwargs).last_hidden_state)
        up, bu = self.classifiers["upos"](seq), self.classifiers["bundle"](seq)
        loss = None
        if upos_labels is not None and bundle_labels is not None:
            loss = (self._kayip(up, upos_labels) + self._kayip(bu, bundle_labels)) / 2
        return BundleTagOutput(loss=loss, upos_logits=up, bundle_logits=bu)


# =============================================================================
# 9) ETİKET SÖZLÜKLERİ, VERİ KÜMESİ SINIFLARI
# =============================================================================

def build_label_mappings(train_data: List[Dict], sessiz: bool = False) -> Dict[str, Dict[str, int]]:
    """Etiket sözlükleri YALNIZ eğitim verisinden (sıralı) oluşturulur."""
    label_mappings: Dict[str, Dict[str, int]] = {}
    for h in MORPH_HEADS:
        unique_labels = sorted({s[h][j] for s in train_data for j in range(len(s[h]))})
        label_mappings[h] = {lbl: i for i, lbl in enumerate(unique_labels)}
        if not sessiz:
            logger.info(f"  [{h}] {len(unique_labels)} etiket: {unique_labels}")
    return label_mappings


def build_bundle_vocab(train_data: List[Dict]) -> Dict[str, int]:
    """Eğitimde geçen demetler (bölünme 0: IMST 432, BOUN r2.10 542 sınıf)."""
    return {b: i for i, b in enumerate(sorted({_bundle_of(s, i) for s in train_data
                                                for i in range(len(s["tokens"]))}))}


def audit_label_coverage(data: List[Dict], label_mappings: Dict[str, Dict[str, int]],
                         split_name: str = "split") -> Dict[str, Counter]:
    missing_by_head: Dict[str, Counter] = {h: Counter() for h in MORPH_HEADS}
    for sent in data:
        for j in range(len(sent["tokens"])):
            for h in MORPH_HEADS:
                if sent[h][j] not in label_mappings[h]:
                    missing_by_head[h][sent[h][j]] += 1
    SEP = "─" * 72
    total_missing = sum(sum(c.values()) for c in missing_by_head.values())
    if not total_missing:
        print(f"  [{split_name}] ✅ Tüm etiketler mapping'de mevcut.")
    else:
        total_tokens = sum(len(s["tokens"]) for s in data)
        pct = total_missing / max(1, total_tokens)
        logger.warning(f"  [{split_name}] Mapping dışı: {total_missing:,} token ({pct:.2%})")
        print(f"\n{SEP}\n  [{split_name}] ⚠️  MAPPING DIŞI ETİKET RAPORU\n{SEP}")
        for h in MORPH_HEADS:
            for lbl, freq in sorted(missing_by_head[h].items(), key=lambda x: -x[1]):
                print(f"  {h:<12}  '{lbl}'  →  {freq:,} token, ignore (-100).")
        print(f"\n  Toplam : {total_missing:,} / {total_tokens:,}  ({pct:.2%})")
        if pct > 0.05:
            print("  ⚠️  Etki oranı %5'i aşıyor! Test metrikleri şişirilmiş olabilir.")
        print(SEP + "\n")
    return missing_by_head


def _compute_boundary_positions(word_ids: List[Optional[int]]) -> Dict[int, int]:
    if SUBWORD_STRATEGY not in ("last", "first"):
        raise NotImplementedError("Yalnız 'first' ve 'last' alt-kelime stratejileri desteklenir.")
    boundary: Dict[int, int] = {}
    if SUBWORD_STRATEGY == "last":
        for pos, w_id in enumerate(word_ids):
            if w_id is not None:
                boundary[w_id] = pos
    else:
        for pos, w_id in enumerate(word_ids):
            if w_id is not None and w_id not in boundary:
                boundary[w_id] = pos
    return boundary


class MultiHeadMorphDataset(Dataset):
    def __init__(self, data, tokenizer, label_mappings, max_length: int = MAX_SEQ_LENGTH):
        self.data = data
        self.tokenizer = tokenizer
        self.label_mappings = label_mappings
        self.max_length = max_length
        self._total_count = len(data)

    def __len__(self):
        return len(self.data)

    def _encode(self, item):
        enc = self.tokenizer(item["tokens"], is_split_into_words=True, max_length=self.max_length,
                             padding="max_length", truncation=True, return_tensors="pt")
        word_ids = enc.word_ids(batch_index=0)
        return enc, word_ids, _compute_boundary_positions(word_ids)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc, word_ids, boundary = self._encode(item)
        labels = {h: [] for h in MORPH_HEADS}
        for pos, w_id in enumerate(word_ids):
            if w_id is None or boundary.get(w_id) != pos:
                for h in MORPH_HEADS:
                    labels[h].append(-100)
            else:
                for h in MORPH_HEADS:
                    labels[h].append(self.label_mappings[h].get(item[h][w_id], -100))
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                **{f"{h}_labels": torch.tensor(labels[h], dtype=torch.long) for h in MORPH_HEADS}}

    def compute_truncation_stats(self) -> Tuple[int, float]:
        truncated = 0
        for item in self.data:
            enc = self.tokenizer(item["tokens"], is_split_into_words=True,
                                 max_length=self.max_length, truncation=True)
            returned = {w for w in enc.word_ids() if w is not None}
            if len(returned) < len(item["tokens"]):
                truncated += 1
        return truncated, truncated / max(1, self._total_count)

    def log_truncation_stats(self):
        truncated, pct = self.compute_truncation_stats()
        msg = f"Truncation: {truncated}/{self._total_count} cümle ({pct:.1%}) max_length={self.max_length}."
        logger.info(msg)
        if pct > 0.05:
            logger.warning("Kesim oranı >%5 — MAX_SEQ_LENGTH artırılabilir.")
        print(f"  [Truncation] {msg}")


class BundleTagDataset(MultiHeadMorphDataset):
    """Etiketler: upos + demet (eğitimde olmayan demet → -100)."""

    def __init__(self, data, tokenizer, label_mappings, bundle_vocab, max_length: int = MAX_SEQ_LENGTH):
        super().__init__(data, tokenizer, label_mappings, max_length)
        self.bundle_vocab = bundle_vocab

    def __getitem__(self, idx):
        item = self.data[idx]
        enc, word_ids, boundary = self._encode(item)
        up, bu = [], []
        for pos, w_id in enumerate(word_ids):
            if w_id is None or boundary.get(w_id) != pos:
                up.append(-100)
                bu.append(-100)
            else:
                up.append(self.label_mappings["upos"].get(item["upos"][w_id], -100))
                bu.append(self.bundle_vocab.get(_bundle_of(item, w_id), -100))
        return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0),
                "upos_labels": torch.tensor(up, dtype=torch.long),
                "bundle_labels": torch.tensor(bu, dtype=torch.long)}


# =============================================================================
# 10) METRİKLER
# =============================================================================
#  F1(N+) : sklearn makro F1 (sınıflar = gerçek ∪ tahmin), NONE dahil.
#  F1(N−) : NONE sınıfı ETİKET LİSTESİNDEN çıkarılarak makro F1 (labels=...); NONE
#           sıfır F1'li sahte bir sınıf olarak ortalamaya GİRMEZ. Biçimi F1_NONONE_YONTEMI seçer:
#    "tum_belirtecler" (varsayılan): tüm belirteçler; özellik kaçırma (FN) ve NONE belirtece
#           özellik uydurma (FP) cezalandırılır (NER'de 'O' sınıfını dışlamakla aynı, standart).
#    "altin_maske": yalnız gerçek etiketi NONE olmayan belirteçler + labels=.
#           Her iki biçim de her koşumda ayrıca kaydedilir.
#  F1(N−, eski/legacy) : yalnız E0 karşılaştırması için.

def f1_nonone(y_true, y_pred, none_label) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.size == 0:
        return float("nan")
    labels = sorted(set(np.unique(y_true).tolist()) | set(np.unique(y_pred).tolist()))
    if none_label is not None:
        labels = [l for l in labels if l != none_label]
    if not labels:
        return float("nan")
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def f1_nonone_goldmask(y_true, y_pred, none_label) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if none_label is not None:
        m = y_true != none_label
        y_true, y_pred = y_true[m], y_pred[m]
    return f1_nonone(y_true, y_pred, none_label)


def f1_nonone_legacy(y_true, y_pred, none_label) -> float:
    """Gerçek etiketi NONE olan belirteçler atılır, fakat tahmin edilen NONE makro
    ortalamada sıfır F1'li bir sınıf olarak kalır. Yalnız E0 karşılaştırması içindir."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if none_label is not None:
        m = y_true != none_label
        y_true, y_pred = y_true[m], y_pred[m]
    if y_true.size == 0:
        return float("nan")
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _nanmean(vals: List[float], bos: float = 0.0) -> float:
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else bos


def _ids_from_output(arr) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 3:
        return a.argmax(axis=-1)
    if a.ndim == 2 and np.issubdtype(a.dtype, np.floating):
        return a.argmax(axis=-1)
    return a


def compute_multihead_metrics(eval_pred, label_mappings: Dict = None):
    """Doğrulama (dev) metrikleri; erken durdurma ölçütü 'avg_f1_nonone' DÜZELTİLMİŞ F1(N−)."""
    raw_preds, raw_labels = eval_pred.predictions, eval_pred.label_ids
    logits_list = list(raw_preds) if isinstance(raw_preds, (tuple, list)) else [raw_preds]
    if isinstance(raw_labels, dict):
        labels_list = [raw_labels[f"{h}_labels"] for h in _HEAD_ORDER if f"{h}_labels" in raw_labels]
    elif isinstance(raw_labels, (tuple, list)):
        labels_list = list(raw_labels)
    else:
        labels_list = [raw_labels]
    if len(logits_list) != len(_HEAD_ORDER) or len(labels_list) != len(_HEAD_ORDER):
        logger.warning(f"compute_multihead_metrics: logit ({len(logits_list)}) / etiket ({len(labels_list)}) "
                       f"sayısı != {len(_HEAD_ORDER)}")

    metrics: Dict[str, float] = {}
    f1s, f1nn, f1nn_t, f1nn_gm, accs = [], [], [], [], []
    pred_all, gold_all = {}, {}
    for idx, h in enumerate(_HEAD_ORDER):
        if idx >= len(logits_list) or idx >= len(labels_list):
            break
        pred = _ids_from_output(logits_list[idx]).reshape(-1)
        gold = np.asarray(labels_list[idx]).reshape(-1)
        if pred.shape != gold.shape:
            n = min(pred.size, gold.size)
            pred, gold = pred[:n], gold[:n]
        pred_all[h], gold_all[h] = pred, gold
        mask = gold != -100
        if not mask.any():
            continue
        y_true, y_pred = gold[mask], pred[mask]
        none_id = label_mappings[h].get("NONE") if label_mappings is not None else None
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        ft = f1_nonone(y_true, y_pred, none_id)
        fg = f1_nonone_goldmask(y_true, y_pred, none_id)
        fn = fg if F1_NONONE_YONTEMI == "altin_maske" else ft
        metrics[f"{h}_accuracy"] = float(acc)
        metrics[f"{h}_f1"] = float(f1)
        metrics[f"{h}_f1_nonone"] = _nan_to_none(float(fn))
        accs.append(acc)
        f1s.append(f1)
        f1nn.append(fn)
        f1nn_t.append(ft)
        f1nn_gm.append(fg)
    metrics["avg_f1"] = _nanmean(f1s)
    metrics["avg_accuracy"] = _nanmean(accs)
    metrics["avg_f1_nonone"] = _nanmean(f1nn, bos=metrics["avg_f1"])
    metrics["avg_f1_nonone_tum"] = _nanmean(f1nn_t, bos=metrics["avg_f1"])
    metrics["avg_f1_nonone_goldmask"] = _nanmean(f1nn_gm, bos=metrics["avg_f1"])
    valid = [x for x in f1nn if not math.isnan(x)]
    metrics["min_f1_nonone"] = float(min(valid)) if valid else metrics["avg_f1_nonone"]
    # Dev Exact Match (UPOS + 9 özellik aynı anda doğru), yalnız tüm başlıkların geçerli olduğu konumlar
    if len(gold_all) == len(_HEAD_ORDER):
        G = np.stack([gold_all[h] for h in _HEAD_ORDER])
        P = np.stack([pred_all[h] for h in _HEAD_ORDER])
        v = (G != -100).all(axis=0)
        metrics["exact_match"] = float((P[:, v] == G[:, v]).all(axis=0).mean()) if v.any() else 0.0
    # Trainer None değerleri sevmez
    return {k: (0.0 if v is None else v) for k, v in metrics.items()}


def compute_bundle_model_metrics(eval_pred):
    """Birleşik-etiket modeli için dev metrikleri."""
    preds, labels = eval_pred.predictions, eval_pred.label_ids
    up_p, bu_p = _ids_from_output(preds[0]).reshape(-1), _ids_from_output(preds[1]).reshape(-1)
    up_g, bu_g = np.asarray(labels[0]).reshape(-1), np.asarray(labels[1]).reshape(-1)
    m = (up_g != -100) & (bu_g != -100)
    out = {"upos_accuracy": float((up_p[up_g != -100] == up_g[up_g != -100]).mean()) if (up_g != -100).any() else 0.0,
           "bundle_accuracy": float((bu_p[bu_g != -100] == bu_g[bu_g != -100]).mean()) if (bu_g != -100).any() else 0.0,
           "exact_match": float(((up_p == up_g) & (bu_p == bu_g))[m].mean()) if m.any() else 0.0}
    return out


def compute_test_metrics(token_results: List[Dict], label_mappings: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """Test metrikleri (belirteç düzeyi sonuçlardan). Eğitimde olmayan gerçek etiket
    (ör. IMST'de 1 Mood=NecPot) ilgili başlıkta dışlanır."""
    metrics: Dict[str, float] = {}
    total = len(token_results)
    for h in MORPH_HEADS:
        pairs = [(r["true"][h], r["pred"][h]) for r in token_results if r["true"][h] in label_mappings[h]]
        exc = total - len(pairs)
        if exc and exc / max(1, total) > 0.02:
            logger.warning(f"  [test_metrics] [{h}] {exc}/{total} token dışlandı.")
        if not pairs:
            for k in ("accuracy", "f1", "f1_nonone", "f1_nonone_tum", "f1_nonone_goldmask", "f1_nonone_legacy"):
                metrics[f"test_{h}_{k}"] = float("nan") if "nonone" in k else 0.0
            metrics[f"test_{h}_n"] = 0
            continue
        yt = np.array([p[0] for p in pairs], dtype=object)
        yp = np.array([p[1] for p in pairs], dtype=object)
        none_lbl = "NONE" if "NONE" in label_mappings[h] else None
        metrics[f"test_{h}_accuracy"] = float(accuracy_score(yt, yp))
        metrics[f"test_{h}_f1"] = float(f1_score(yt, yp, average="macro", zero_division=0))
        metrics[f"test_{h}_f1_nonone_tum"] = f1_nonone(yt, yp, none_lbl)
        metrics[f"test_{h}_f1_nonone_goldmask"] = f1_nonone_goldmask(yt, yp, none_lbl)
        metrics[f"test_{h}_f1_nonone"] = metrics[f"test_{h}_f1_nonone_goldmask" if F1_NONONE_YONTEMI == "altin_maske"
                                                 else f"test_{h}_f1_nonone_tum"]
        metrics[f"test_{h}_f1_nonone_legacy"] = f1_nonone_legacy(yt, yp, none_lbl)
        metrics[f"test_{h}_n"] = len(pairs)
    metrics["test_avg_f1"] = float(np.mean([metrics[f"test_{h}_f1"] for h in MORPH_HEADS]))
    metrics["test_avg_acc"] = float(np.mean([metrics[f"test_{h}_accuracy"] for h in MORPH_HEADS]))
    for ek in ("", "_tum", "_goldmask", "_legacy"):
        metrics[f"test_nonone{ek}_avg_f1"] = _nanmean([metrics[f"test_{h}_f1_nonone{ek}"] for h in MORPH_HEADS])
    return metrics


# Geriye dönük uyumlu adlar
def compute_test_metrics_from_token_results(token_results, label_mappings):
    m = compute_test_metrics(token_results, label_mappings)
    return {k: v for k, v in m.items() if not k.startswith("test_nonone")}


def compute_test_metrics_no_none(token_results, label_mappings):
    m = compute_test_metrics(token_results, label_mappings)
    out = {f"test_nonone_{h}_f1": m[f"test_{h}_f1_nonone"] for h in MORPH_HEADS}
    out["test_nonone_avg_f1"] = m["test_nonone_avg_f1"]
    return out


# =============================================================================
# 11) EĞİTİM
# =============================================================================

def _argmax_logitler(logits, labels):
    """Değerlendirmede logit yerine yalnız argmax biriktirilir (bellek; sonuç değişmez)."""
    if isinstance(logits, (tuple, list)):
        return tuple(l.argmax(dim=-1) for l in logits)
    return logits.argmax(dim=-1)


def _egitim_argumanlari(**kw) -> TrainingArguments:
    """Kurulu Transformers sürümünün desteklemediği parametreleri atıp loglar (4.x/5.x uyumu)."""
    sig = inspect.signature(TrainingArguments.__init__).parameters
    kabul = {k: v for k, v in kw.items() if k in sig}
    atilan = sorted(set(kw) - set(kabul))
    if atilan:
        logger.info(f"  TrainingArguments: bu sürümde olmayan parametreler atlandı: {atilan}")
    return TrainingArguments(**kabul)


def _checkpoint_dizini(short: str, slug: str) -> str:
    base = DRIVE_CHECKPOINT_ROOT if USE_DRIVE_CHECKPOINTS else CHECKPOINT_KOK
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{short}_{slug}")


def _en_iyi_eval(log_history: List[Dict], metric: str) -> Dict:
    key = f"eval_{metric}"
    evals = [e for e in log_history if key in e]
    # Trainer ile aynı kural: eşitlikte İLK en iyi epoch korunur (max ilkini döndürür).
    return max(evals, key=lambda x: x.get(key, float("-inf")), default={})


def train_one_model(model_cfg: Dict, train_ds: Dataset, dev_ds: Dataset, label_mappings: Dict,
                    checkpoint_dir: str, seed: int, model_turu: str = "multihead",
                    bundle_vocab: Optional[Dict[str, int]] = None,
                    max_steps: Optional[int] = None) -> Tuple[Optional[nn.Module], Dict, List, Dict]:
    """Tek model eğitimi. Dönüş: (model, en_iyi_dev_eval, log_history, egitim_bilgisi)."""
    name, hf_id = model_cfg["name"], model_cfg["hf_id"]
    set_global_seed(seed)
    t0 = time.time()
    try:
        if model_turu == "bundle":
            cfg = BundleTagConfig(base_model_name=hf_id, dropout=DROPOUT_RATE,
                                  num_upos_labels=len(label_mappings["upos"]),
                                  num_bundle_labels=len(bundle_vocab))
            model = BundleTagModel(cfg)
            metric_name = "exact_match"
            compute_fn = compute_bundle_model_metrics
            label_names = ["upos_labels", "bundle_labels"]
        else:
            cfg = MultiHeadConfig(base_model_name=hf_id, dropout=DROPOUT_RATE,
                                  **{f"num_{h}_labels": len(label_mappings[h]) for h in MORPH_HEADS})
            model = MultiHeadMorphModel(cfg)
            metric_name = EARLY_STOPPING_METRIC
            compute_fn = partial(compute_multihead_metrics, label_mappings=label_mappings)
            label_names = list(_LABEL_NAMES)
    except Exception as e:
        logger.warning(f"  [{name}] Model kurulamadı: {e}")
        print(f"  [{name}] ⚠️  Model kurulamadı ({e}) — atlanıyor.")
        return None, {}, [], {}

    init_r = getattr(model, "init_raporu", {})
    print(f"  [{name}] Başlatma denetimi: kodlayıcı değişmedi={init_r.get('kodlayici_degismedi')}  "
          f"başlıklar N(0; {init_r.get('hedef_std')}) "
          f"{'(açıkça yeniden başlatıldı)' if init_r.get('basliklar_acikca_yeniden_baslatildi') else ''}")

    steps_per_epoch = math.ceil(len(train_ds) / TRAIN_BATCH_SIZE)
    total_steps = steps_per_epoch * NUM_TRAIN_EPOCHS if max_steps is None else max_steps
    warmup_steps = int(total_steps * WARMUP_RATIO)
    args = _egitim_argumanlari(
        output_dir=checkpoint_dir,
        eval_strategy="epoch" if max_steps is None else "no",
        save_strategy="epoch" if max_steps is None else "no",
        logging_steps=50,
        learning_rate=LEARNING_RATE,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        **({"optim": OPTIMIZER} if OPTIMIZER else {}),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        load_best_model_at_end=max_steps is None,
        metric_for_best_model=metric_name if max_steps is None else None,
        greater_is_better=True,
        report_to="none",
        seed=seed,
        data_seed=seed,
        label_names=label_names,
        save_total_limit=2,
        dataloader_num_workers=0,
        use_cpu=not torch.cuda.is_available(),
        disable_tqdm=False,
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE,
                                       early_stopping_threshold=0.0)] if max_steps is None else []
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=dev_ds if max_steps is None else None,
                      compute_metrics=compute_fn if max_steps is None else None, callbacks=callbacks,
                      preprocess_logits_for_metrics=_argmax_logitler if max_steps is None else None)
    trainer.train()
    model = trainer.model
    log_history = list(trainer.state.log_history)
    actual_epoch = float(trainer.state.epoch or 0.0)
    best_ckpt = trainer.state.best_model_checkpoint
    optim_adi = str(getattr(trainer.args, "optim", ""))
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if max_steps is not None:   # determinizm sınaması gibi kısa koşumlar
        return model, {}, log_history, {"sure_sn": time.time() - t0}

    if actual_epoch < COMPLETION_MIN_EPOCHS:
        logger.error(f"  [{name}] EĞİTİM TAMAMLANMADI: {actual_epoch:.2f} epoch < min {COMPLETION_MIN_EPOCHS}.")
        print(f"  [{name}] ⚠️  EĞİTİM TAMAMLANMADI — ATLANIYOR.")
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        return None, {}, [], {}

    early_stopped = actual_epoch < NUM_TRAIN_EPOCHS
    stop_reason = f"early-stop @ epoch {actual_epoch:.1f}" if early_stopped else "full training"
    best_eval = _en_iyi_eval(log_history, metric_name)
    best_epoch = best_eval.get("epoch")
    bilgi = {
        "secim_olcutu": metric_name,
        "en_iyi_epoch": best_epoch,
        "son_epoch": actual_epoch,
        "en_iyi_son_epoch_mu": (best_epoch is not None and abs(best_epoch - actual_epoch) < 1e-6),
        "erken_durdu": early_stopped,
        "en_iyi_checkpoint": best_ckpt,
        "toplam_adim": total_steps, "isinma_adimi": warmup_steps,
        "optimizer": optim_adi, "lr_scheduler": LR_SCHEDULER,
        "sure_sn": round(time.time() - t0, 1),
        "init_raporu": init_r,
    }
    print(f"  [{name}] Epoch: {actual_epoch:.1f}/{NUM_TRAIN_EPOCHS}  |  Seçim ölçütü {metric_name} (dev): "
          f"{best_eval.get(f'eval_{metric_name}', 0):.4f} @ epoch {best_epoch}  |  "
          f"dev EM: {best_eval.get('eval_exact_match', float('nan')):.4f}  |  "
          f"min-başlık F1(N−): {best_eval.get('eval_min_f1_nonone', float('nan')):.4f}  ({stop_reason})")
    logger.info(f"  [{name}] {json.dumps({k: v for k, v in bilgi.items() if k != 'init_raporu'}, default=str)}")
    return model, best_eval, log_history, bilgi


# =============================================================================
# 12) ÇIKARIM (TEST TAHMİNLERİ)
# =============================================================================

def batch_predict(model, tokenizer, data: List[Dict], label_mappings: Dict[str, Dict[str, int]],
                  batch_size: int = EVAL_BATCH_SIZE, device: torch.device = None,
                  bundle_vocab: Optional[Dict[str, int]] = None) -> List[Dict]:
    """Test kümesi üzerinde belirteç düzeyi tahminler. bundle_vocab verilirse model
    birleşik-etiket modelidir; tahmin edilen demet başlıklara ayrıştırılır."""
    if device is None:
        device = next(model.parameters()).device
    id2label = {h: {v: k for k, v in m.items()} for h, m in label_mappings.items()}
    id2bundle = {v: k for k, v in bundle_vocab.items()} if bundle_vocab is not None else None
    model.eval()
    results: List[Dict] = []
    for start in range(0, len(data), batch_size):
        batch = data[start:start + batch_size]
        enc = tokenizer([s["tokens"] for s in batch], is_split_into_words=True, max_length=MAX_SEQ_LENGTH,
                        padding="max_length", truncation=True, return_tensors="pt")
        extra = {}
        if "token_type_ids" in enc:
            extra = _safe_encoder_kwargs(model.encoder, {"token_type_ids": enc["token_type_ids"].to(device)})
        with torch.no_grad():
            outs = model(input_ids=enc["input_ids"].to(device),
                         attention_mask=enc["attention_mask"].to(device), **extra)
        if id2bundle is None:
            pred_ids = {h: getattr(outs, f"{h}_logits").argmax(-1).cpu().numpy() for h in MORPH_HEADS}
        else:
            pred_ids = {"upos": outs.upos_logits.argmax(-1).cpu().numpy(),
                        "bundle": outs.bundle_logits.argmax(-1).cpu().numpy()}
        assert next(iter(pred_ids.values())).shape[0] == len(batch)
        for li, sent in enumerate(batch):
            word_ids = enc.word_ids(batch_index=li)
            boundary = _compute_boundary_positions(word_ids)
            for pos, w_id in enumerate(word_ids):
                if w_id is None or boundary.get(w_id) != pos:
                    continue
                if id2bundle is None:
                    pred = {h: id2label[h].get(int(pred_ids[h][li, pos]), "NONE") for h in MORPH_HEADS}
                else:
                    pred = {"upos": id2label["upos"].get(int(pred_ids["upos"][li, pos]), "NONE"),
                            **bundle_to_heads(id2bundle.get(int(pred_ids["bundle"][li, pos]), "NONE"))}
                results.append({"sent_idx": start + li, "word_idx": w_id, "token": sent["tokens"][w_id],
                                "true": {h: sent[h][w_id] for h in MORPH_HEADS}, "pred": pred})
    return results


# =============================================================================
# 13) BELİRTEÇ KAYITLARI (em_records) VE EXACT MATCH
# =============================================================================

def _feats_str(ft: Dict[str, str]) -> str:
    return "|".join(f"{k}={v}" for k, v in sorted(ft.items())) if ft else "_"


def token_kayitlari(token_results: List[Dict], test_data: List[Dict], held_out_bundles: Set[str],
                    train_bundles: Optional[Set[str]] = None,
                    train_forms: Optional[Set[str]] = None) -> List[Dict]:
    """Zengin belirteç kaydı: başlık başına gerçek/tahmin, demetler, gruplar, biçim görülme."""
    recs = []
    for r in token_results:
        sent = test_data[r["sent_idx"]]
        j = r["word_idx"]
        true_b = _bundle_of(sent, j)
        pred_b = get_bundle_string({h: r["pred"][h] for h in FEAT_HEADS})
        tu, pu = r["true"]["upos"], r["pred"]["upos"]
        is_seen = true_b not in held_out_bundles
        if not is_seen:
            grp = "unseen"
        elif train_bundles is None:
            grp = "seen"
        else:
            grp = "seen_train" if true_b in train_bundles else "seen_novel"
        rec = {
            "sent_idx": r["sent_idx"], "word_idx": j, "token": r["token"],
            "lemma": sent.get("lemmas", [None] * (j + 1))[j] if sent.get("lemmas") else None,
            "feats": _feats_str(sent["feats"][j]) if sent.get("feats") else None,
            "true_label": f"{tu}+{true_b}", "pred_label": f"{pu}+{pred_b}",
            "true_bundle": true_b, "pred_bundle": pred_b,
            "is_seen": is_seen, "seen_group": grp,
            "form_in_train": (tr_lower(r["token"]) in train_forms) if train_forms is not None else None,
            "is_em": (pu == tu) and (pred_b == true_b),
        }
        for h in MORPH_HEADS:
            rec[f"true_{h}"] = r["true"][h]
            rec[f"pred_{h}"] = r["pred"][h]
        recs.append(rec)
    return recs


def em_ozetle(records: List[Dict]) -> Dict[str, Any]:
    def _m(recs):
        if not recs:
            return {"em": float("nan"), "f1": float("nan"), "n": 0}
        return {"em": sum(r["is_em"] for r in recs) / len(recs),
                "f1": f1_score([r["true_label"] for r in recs], [r["pred_label"] for r in recs],
                               average="macro", zero_division=0),
                "n": len(recs)}

    ov = _m(records)
    sv = _m([r for r in records if r["is_seen"]])
    uv = _m([r for r in records if not r["is_seen"]])
    st = [r for r in records if r.get("seen_group") == "seen_train"]
    sn = [r for r in records if r.get("seen_group") == "seen_novel"]
    return {
        "overall_em": ov["em"], "overall_f1": ov["f1"],
        "seen_em": sv["em"], "seen_f1": sv["f1"],
        "unseen_em": uv["em"], "unseen_f1": uv["f1"],
        "overall_token_count": ov["n"], "seen_token_count": sv["n"], "unseen_token_count": uv["n"],
        "seen_train_em": (sum(r["is_em"] for r in st) / len(st)) if st else float("nan"),
        "seen_novel_em": (sum(r["is_em"] for r in sn) / len(sn)) if sn else float("nan"),
        "seen_train_token_count": len(st), "seen_novel_token_count": len(sn),
        "records": records,
    }


class ExactMatchMetrics:
    """Geriye dönük uyumlu arayüz."""

    def __init__(self, label_mappings: Dict):
        self.label_mappings = label_mappings

    def compute_from_results(self, token_results, test_data, held_out_bundles,
                             train_bundles=None, train_forms=None) -> Dict:
        return em_ozetle(token_kayitlari(token_results, test_data, held_out_bundles, train_bundles, train_forms))


def bootstrap_bundle_ci(records: List[Dict], is_seen: bool, n_boot: int = BOOTSTRAP_N_RESAMPLES,
                        alpha: float = ALPHA, seed: int = BOOTSTRAP_SEED) -> Tuple[float, float]:
    """Demet-bazlı yüzdelik bootstrap %95 GA."""
    subset = [r for r in records if r["is_seen"] == is_seen]
    if not subset:
        return (float("nan"), float("nan"))
    by_bundle: Dict[str, List[bool]] = defaultdict(list)
    for r in subset:
        by_bundle[r["true_bundle"]].append(r["is_em"])
    bundles = list(by_bundle.keys())
    if len(bundles) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    boot = []
    for _ in range(n_boot):
        sampled = rng.choice(bundles, size=len(bundles), replace=True)
        vals = [v for b in sampled for v in by_bundle[b]]
        if vals:
            boot.append(float(np.mean(vals)))
    if not boot:
        return (float("nan"), float("nan"))
    return (float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2))))


# =============================================================================
# 14) RAPORLAYICILAR VE AYRINTILI ANALİZLER
# =============================================================================

class SeenUnseenReporter:
    def __init__(self):
        self.rows = []

    def add(self, model_name: str, dataset_name: str, em_result: Dict, split_seed: int = None, seed: int = None):
        self.rows.append({
            "model": model_name, "dataset": dataset_name, "split_seed": split_seed, "seed": seed,
            "overall_em": em_result["overall_em"], "overall_f1": em_result["overall_f1"],
            "seen_em": em_result["seen_em"], "seen_f1": em_result["seen_f1"],
            "unseen_em": em_result["unseen_em"], "unseen_f1": em_result["unseen_f1"],
            "seen_train_em": em_result.get("seen_train_em"), "seen_novel_em": em_result.get("seen_novel_em"),
            "seen_n": em_result["seen_token_count"], "unseen_n": em_result["unseen_token_count"],
            "seen_novel_n": em_result.get("seen_novel_token_count"),
            "total_n": em_result["overall_token_count"],
            "seen_em_ci_low": em_result.get("seen_em_ci_low", float("nan")),
            "seen_em_ci_high": em_result.get("seen_em_ci_high", float("nan")),
            "unseen_em_ci_low": em_result.get("unseen_em_ci_low", float("nan")),
            "unseen_em_ci_high": em_result.get("unseen_em_ci_high", float("nan")),
        })

    def report(self) -> pd.DataFrame:
        if not self.rows:
            print("  [UYARI] Raporlanacak sonuç yok.")
            return pd.DataFrame()
        df = pd.DataFrame(self.rows)
        SEP1, SEP2 = "═" * 132, "─" * 132
        print(f"\n{SEP1}\n  SEEN / UNSEEN EXACT MATCH & F1 RAPORU\n{SEP1}")
        print(f"  {'Model':<12} {'Dataset':<9} {'OvrlEM':>8} {'OvrlF1':>8} {'SeenEM':>8} {'SeenF1':>8} "
              f"{'UnseenEM':>9} {'UnseenF1':>9} {'Seen-tr':>8} {'Seen-yeni':>9} {'SeenN':>7} {'UnseenN':>8} {'TotalN':>7}")
        print(SEP2)
        for _, r in df.iterrows():
            print(f"  {r['model']:<12} {r['dataset']:<9} {r['overall_em']:>7.1%} {r['overall_f1']:>8.1%} "
                  f"{r['seen_em']:>7.1%} {r['seen_f1']:>8.1%} {r['unseen_em']:>8.1%} {r['unseen_f1']:>9.1%} "
                  f"{r['seen_train_em']:>7.1%} {r['seen_novel_em']:>9.1%} {int(r['seen_n']):>7,} "
                  f"{int(r['unseen_n']):>8,} {int(r['total_n']):>7,}")
        print(f"\n  Demet-bazlı bootstrap %{100 * (1 - ALPHA):.0f} GA (n_boot={BOOTSTRAP_N_RESAMPLES}, "
              f"tohum={BOOTSTRAP_SEED}; belirteç değil demet yeniden örneklenir):")
        for _, r in df.iterrows():
            def _ci(lo, hi):
                return "n/a (yetersiz demet)" if (pd.isna(lo) or pd.isna(hi)) else f"[{lo:.1%}, {hi:.1%}]"
            print(f"    {r['model']:<12} [{r['dataset']}]  Seen EM: {r['seen_em']:.1%} {_ci(r['seen_em_ci_low'], r['seen_em_ci_high'])}"
                  f"   Unseen EM: {r['unseen_em']:.1%} {_ci(r['unseen_em_ci_low'], r['unseen_em_ci_high'])}")
        print("\n  NOT: 'Seen' = held-out'a alınMAMIŞ demet. 'Seen-yeni' = held-out değil ama bölünme sonrası")
        print("       eğitimde HİÇ geçmeyen demet (tezde Seen içinde sayılıyordu; denetim O3).")
        print(SEP1 + "\n")
        return df

    def save_csv(self, path: str):
        df = pd.DataFrame(self.rows)
        df.to_csv(path, index=False, float_format="%.4f")
        print(f"  Rapor '{path}' dosyasına kaydedildi.")
        return df


def compute_per_head_accuracy_breakdown(token_results, test_data, held_out_bundles) -> Dict[str, Dict[str, float]]:
    counts = {h: {"seen_correct": 0, "seen_total": 0, "unseen_correct": 0, "unseen_total": 0} for h in MORPH_HEADS}
    for r in token_results:
        prefix = "unseen" if _bundle_of(test_data[r["sent_idx"]], r["word_idx"]) in held_out_bundles else "seen"
        for h in MORPH_HEADS:
            counts[h][f"{prefix}_correct"] += int(r["pred"][h] == r["true"][h])
            counts[h][f"{prefix}_total"] += 1
    res = {}
    for h in MORPH_HEADS:
        c = counts[h]
        res[h] = {"seen_acc": round(c["seen_correct"] / max(1, c["seen_total"]), 4),
                  "unseen_acc": round(c["unseen_correct"] / max(1, c["unseen_total"]), 4),
                  "overall_acc": round((c["seen_correct"] + c["unseen_correct"]) /
                                       max(1, c["seen_total"] + c["unseen_total"]), 4),
                  "seen_n": c["seen_total"], "unseen_n": c["unseen_total"]}
    return res


def print_per_head_breakdown(breakdown, model_name: str, dataset_name: str):
    SEP = "─" * 78
    print(f"\n{SEP}\n  BAŞLIK BAŞINA DOĞRULUK — {model_name} [{dataset_name}]\n{SEP}")
    print(f"  {'Head':<12} {'Overall':>9} {'Seen':>9} {'Unseen':>9} {'SeenN':>7} {'UnseenN':>9}")
    print(f"  {'-' * 60}")
    for h in MORPH_HEADS:
        r = breakdown[h]
        print(f"  {h:<12} {r['overall_acc']:>8.1%} {r['seen_acc']:>8.1%} {r['unseen_acc']:>8.1%} "
              f"{r['seen_n']:>7,} {r['unseen_n']:>9,}")
    print(f"{SEP}\n  NOT: Sayımlar sözcük-sınırı belirteç bazlıdır. NONE sınıfı dahildir.\n{SEP}")


def collect_detailed_predictions(token_results, test_data, held_out_bundles) -> List[Dict]:
    recs = []
    for r in token_results:
        tb = _bundle_of(test_data[r["sent_idx"]], r["word_idx"])
        pb = get_bundle_string({h: r["pred"][h] for h in FEAT_HEADS})
        recs.append({"token": r["token"], "true_bundle": tb, "pred_bundle": pb,
                     "is_seen": tb not in held_out_bundles, "is_correct": tb == pb})
    return recs


def bundle_level_report(records, model_name, dataset_name, save_path: str = None) -> pd.DataFrame:
    st = defaultdict(lambda: {"total": 0, "correct": 0, "seen": None})
    for r in records:
        b = r["true_bundle"]
        st[b]["total"] += 1
        st[b]["correct"] += int(r["is_correct"])
        st[b]["seen"] = r["is_seen"]
    df = pd.DataFrame([{"bundle": b, "status": "SEEN" if s["seen"] else "UNSEEN", "freq": s["total"],
                        "correct": s["correct"], "accuracy": s["correct"] / s["total"]} for b, s in st.items()])
    df = df.sort_values(["status", "freq", "bundle"], ascending=[True, False, True], kind="mergesort")
    sep = "─" * 90
    print(f"\n{sep}\n  {model_name} — {dataset_name} | Demet Başına Tahmin ÖZETİ (yalnız demet, UPOS hariç)\n{sep}")
    for stt, etiket in (("SEEN", "Seen  "), ("UNSEEN", "Unseen")):
        d = df[df["status"] == stt]
        if len(d):
            print(f"  {etiket} : {len(d):>3} benzersiz | {d['freq'].sum():>6,} belirteç | doğruluk: "
                  f"{d['correct'].sum() / d['freq'].sum():.1%} | makro (demet ortalaması): {d['accuracy'].mean():.1%}")
    print(sep)
    if save_path:
        df.to_csv(save_path, index=False, float_format="%.4f")
    return df


def error_analysis(records, model_name, dataset_name, held_out_bundles: Set[str], top_n=10, save_path: str = None):
    errors = Counter()
    for r in records:
        if not r["is_correct"]:
            errors[(r["true_bundle"], r["pred_bundle"])] += 1
    print(f"\n  {model_name} — {dataset_name} | En Sık {top_n} Demet Hatası")
    print(f"  {'Gerçek  →  Tahmin':<82} {'N':>4}\n  {'-' * 87}")
    for (tb, pb), cnt in errors.most_common(top_n):
        tag = " ⚡UNSEEN" if tb in held_out_bundles else ""
        print(f"  {tb + '  →  ' + pb:<82} {cnt:>4}{tag}")
    if save_path:
        pd.DataFrame([{"gercek": tb, "tahmin": pb, "n": c, "unseen": tb in held_out_bundles}
                      for (tb, pb), c in errors.most_common()]).to_csv(save_path, index=False)


# =============================================================================
# 15) İSTATİSTİK: McNEMAR (+duyarlılık), HOLM, SPEARMAN GÜCÜ, EŞLEŞTİRİLMİŞ TESTLER
# =============================================================================

def holm_duzeltmesi(pvals: List[float]) -> List[float]:
    """Holm–Bonferroni düzeltilmiş p-değerleri (NaN'ler korunur)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return out.tolist()
    order = idx[np.argsort(p[idx])]
    m = order.size
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)
        out[i] = running
    return out.tolist()


def mcnemar_pairwise(em_results, model_short_names, subset: str = "overall", dataset_name: str = "",
                     seed=None, split_seed=None, common_keys=None, sessiz: bool = False) -> List[Dict]:
    """Eşleştirilmiş McNEMAR (belirteç düzeyi). Birincil p: b+c<25 ise kesin (binom), değilse
    süreklilik düzeltmesiz χ². Duyarlılık için süreklilik
    düzeltmeli ve kesin p-değerleri ayrıca raporlanır. UYARI: belirteçler bağımsız değildir
    (aynı demetteki belirteçler kümelenir) → sonuçlar KEŞİFSEL; Unseen için demet düzeyinde
    eşleştirilmiş bootstrap'a bakınız (E3)."""
    pairs = [(model_short_names[i], model_short_names[j])
             for i in range(len(model_short_names)) for j in range(i + 1, len(model_short_names))]
    n_pairs = len(pairs)
    alpha_adj = ALPHA / n_pairs if n_pairs else ALPHA
    nm = NAME_MAP()
    if not sessiz:
        SEP = "─" * 118
        print(f"\n{SEP}\n  McNEMAR [{dataset_name} | split={split_seed} | seed={seed} | {subset.upper()}] — "
              f"Bonferroni ailesi: bu alt küme içindeki {n_pairs} çift (α_adj={alpha_adj:.4f}); KEŞİFSEL\n{SEP}")
        print(f"  {'Çift':<24} {'N':>7} {'b':>5} {'c':>5} {'test':>6} {'χ²':>9} {'p':>9} "
              f"{'p(düz.)':>9} {'p(kesin)':>9} {'p<α_adj':>8}")
    rows = []
    for s1, s2 in pairs:
        r1, r2 = em_results.get(s1, {}).get("records", []), em_results.get(s2, {}).get("records", [])
        if not r1 or not r2:
            continue

        def _index(recs):
            return {(r["sent_idx"], r["word_idx"]): bool(r["is_em"]) for r in recs
                    if subset == "overall" or (subset == "seen") == bool(r["is_seen"])}

        i1, i2 = _index(r1), _index(r2)
        common = set(i1) & set(i2)
        if common_keys is not None:
            common &= common_keys
        if not common:
            continue
        b = sum(1 for k in common if i1[k] and not i2[k])
        c = sum(1 for k in common if not i1[k] and i2[k])
        label = f"{nm.get(s1, s1)} vs {nm.get(s2, s2)}"
        if b + c == 0:
            chi2 = p_main = p_cc = p_ex = float("nan")
            test = "none"
        else:
            chi2 = (b - c) ** 2 / (b + c)
            p_chi = float(sps.chi2.sf(chi2, 1))
            p_cc = float(sps.chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1))
            p_ex = float(sps.binomtest(min(b, c), b + c, 0.5).pvalue)
            test = "exact" if (b + c) < 25 else "chi2"
            p_main = p_ex if test == "exact" else p_chi
        sig = bool(p_main < alpha_adj) if not math.isnan(p_main) else False
        better = nm.get(s1, s1) if b > c else (nm.get(s2, s2) if c > b else "eşit")
        rows.append({"dataset": dataset_name, "split_seed": split_seed, "seed": seed, "subset": subset,
                     "model_a": nm.get(s1, s1), "model_b": nm.get(s2, s2), "n_matched": len(common),
                     "b": b, "c": c, "test": test, "chi2": chi2, "p_value": p_main,
                     "p_continuity": p_cc, "p_exact": p_ex, "alpha_adj": alpha_adj,
                     "significant": sig, "daha_iyi": better})
        if not sessiz:
            print(f"  {label:<24} {len(common):>7,} {b:>5} {c:>5} {test:>6} "
                  f"{chi2:>9.3f} {p_main:>9.4f} {p_cc:>9.4f} {p_ex:>9.4f} {('✓' if sig else '✗'):>8}")
    return rows


def spearman_power(n: int, rho: float = 0.5, alpha: float = ALPHA) -> float:
    """Fisher z dönüşümüne dayalı yaklaşık güç (Spearman için 1,06 düzeltmeli)."""
    from scipy.stats import norm
    if n < 5:
        return float("nan")
    z_r = 0.5 * np.log((1 + rho) / (1 - rho))
    se = np.sqrt(1.06 / (n - 3))
    z_a = norm.ppf(1 - alpha / 2)
    return float(norm.cdf(z_r / se - z_a) + norm.cdf(-z_r / se - z_a))


def eslestirilmis_demet_bootstrap(recs_a: List[Dict], recs_b: List[Dict], subset: str = "unseen",
                                  n_boot: int = None, seed: int = None, alpha: float = ALPHA,
                                  common_keys: Optional[Set] = None) -> Dict[str, Any]:
    """[E3] İki modelin EM farkı için DEMET düzeyinde eşleştirilmiş bootstrap.
    Her tekrarda demetler yerine koyarak örneklenir; seçilen demetlerin tüm (ortak)
    belirteçlerinde EM_A − EM_B hesaplanır (belirteç ağırlıklı). Ayrıca makro (demet
    ortalaması) fark ve demet başına farklar üzerinde Wilcoxon işaretli sıra testi."""
    n_boot = BOOTSTRAP_N_RESAMPLES if n_boot is None else n_boot
    seed = BOOTSTRAP_SEED if seed is None else seed

    def _ok(r):
        return subset == "overall" or (subset == "seen") == bool(r["is_seen"])

    ia = {(r["sent_idx"], r["word_idx"]): r for r in recs_a if _ok(r)}
    ib = {(r["sent_idx"], r["word_idx"]): r for r in recs_b if _ok(r)}
    keys = set(ia) & set(ib)
    if common_keys is not None:
        keys &= common_keys
    agg: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])
    for k in keys:
        b = ia[k]["true_bundle"]
        agg[b][0] += 1
        agg[b][1] += int(bool(ia[k]["is_em"]))
        agg[b][2] += int(bool(ib[k]["is_em"]))
    bundles = sorted(agg)
    out: Dict[str, Any] = {"subset": subset, "n_demet": len(bundles), "n_belirtec": len(keys)}
    if len(bundles) < 2:
        out.update({"fark": float("nan"), "ga_alt": float("nan"), "ga_ust": float("nan"), "p_boot": float("nan"),
                    "makro_fark": float("nan"), "makro_ga_alt": float("nan"), "makro_ga_ust": float("nan"),
                    "wilcoxon_p": float("nan")})
        return out
    n = np.array([agg[b][0] for b in bundles], dtype=float)
    a = np.array([agg[b][1] for b in bundles], dtype=float)
    bb = np.array([agg[b][2] for b in bundles], dtype=float)
    per_bundle = a / n - bb / n
    rng = np.random.RandomState(seed)
    counts = rng.multinomial(len(bundles), np.full(len(bundles), 1.0 / len(bundles)), size=n_boot)
    diffs = (counts @ a - counts @ bb) / (counts @ n)
    macro = (counts @ per_bundle) / len(bundles)
    try:
        w_p = float(sps.wilcoxon(per_bundle).pvalue) if np.any(per_bundle != 0) else 1.0
    except ValueError:
        w_p = float("nan")
    out.update({
        "em_a": float(a.sum() / n.sum()), "em_b": float(bb.sum() / n.sum()),
        "fark": float((a.sum() - bb.sum()) / n.sum()),
        "ga_alt": float(np.percentile(diffs, 100 * alpha / 2)),
        "ga_ust": float(np.percentile(diffs, 100 * (1 - alpha / 2))),
        "p_boot": float(min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))),
        "makro_fark": float(per_bundle.mean()),
        "makro_ga_alt": float(np.percentile(macro, 100 * alpha / 2)),
        "makro_ga_ust": float(np.percentile(macro, 100 * (1 - alpha / 2))),
        "wilcoxon_p": w_p,
        "a_daha_iyi_demet": int((per_bundle > 0).sum()), "b_daha_iyi_demet": int((per_bundle < 0).sum()),
    })
    return out


def tohum_duzeyi_testler(deger: Dict[str, Dict[int, float]], etiket: str = "") -> pd.DataFrame:
    """Tohum düzeyinde eşleştirilmiş t-testi.
    deger: {model_kisa_ad: {tohum: metrik}}. Yalnız EĞİTİM varyansını kapsar;
    demet (bölünme) seçiminin varyansını KAPSAMAZ."""
    nm = NAME_MAP()
    shorts = [m["short"] for m in MODELS if m["short"] in deger]
    rows = []
    for i in range(len(shorts)):
        for j in range(len(shorts)):
            if i == j:
                continue
            a, b = shorts[i], shorts[j]
            if [m["short"] for m in MODELS].index(a) > [m["short"] for m in MODELS].index(b):
                continue
            ortak = sorted(set(deger[a]) & set(deger[b]))
            d = np.array([deger[a][t] - deger[b][t] for t in ortak], dtype=float)
            row = {"karsilastirma": etiket, "model_a": nm.get(a, a), "model_b": nm.get(b, b),
                   "tohumlar": ortak, "farklar": d.round(6).tolist(), "n": len(d),
                   "ort_fark": float(d.mean()) if len(d) else float("nan")}
            if len(d) >= 2:
                sd = float(d.std(ddof=1))
                se = sd / math.sqrt(len(d))
                tcrit = float(sps.t.ppf(1 - ALPHA / 2, len(d) - 1))
                row.update({"sd": sd, "ga_alt": row["ort_fark"] - tcrit * se, "ga_ust": row["ort_fark"] + tcrit * se,
                            "t": row["ort_fark"] / se if se > 0 else float("nan"),
                            "p": float(sps.ttest_1samp(d, 0.0).pvalue) if se > 0 else float("nan")})
            rows.append(row)
    return pd.DataFrame(rows)


def std_guven_araligi(s: float, n: int, alpha: float = ALPHA) -> Tuple[float, float]:
    """Örneklem std'si için χ² tabanlı GA: n = 3 ile çok geniştir (σ̂ = 0,90 → [0,47; 5,68])."""
    if n < 2 or s is None or (isinstance(s, float) and math.isnan(s)):
        return (float("nan"), float("nan"))
    df_ = n - 1
    return (math.sqrt(df_ * s ** 2 / sps.chi2.ppf(1 - alpha / 2, df_)),
            math.sqrt(df_ * s ** 2 / sps.chi2.ppf(alpha / 2, df_)))


def build_common_token_keys(em_results) -> set:
    key_sets = [{(r["sent_idx"], r["word_idx"]) for r in res.get("records", [])}
                for res in em_results.values() if res.get("records")]
    if not key_sets:
        return set()
    common = set.intersection(*key_sets)
    total = max(len(s) for s in key_sets)
    print(f"  [Ortak belirteç] {len(common):,} / {total:,} ({len(common) / max(1, total):.3%}) tüm modellerde ortak.")
    return common


# =============================================================================
# 16) JACCARD–EM SPEARMAN (keşifsel)
# =============================================================================

def compute_jaccard_em_spearman(het_df, em_results_flat, dataset_name, save_path=None,
                                min_tokens: int = 1) -> Dict[str, Dict]:
    results_out = {}
    if het_df is None or het_df.empty:
        print("  [UYARI] Yakınlık tablosu boş — Spearman atlandı.")
        return results_out
    nm = NAME_MAP()
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5 * len(MODELS), 4.6), squeeze=False)
    axes = axes[0]
    tez_stili(fig, axes)
    for ax_idx, m_cfg in enumerate(MODELS):
        short, ax = m_cfg["short"], axes[ax_idx]
        recs = em_results_flat.get(short, {}).get("records", [])
        bundle_em = defaultdict(list)
        for r in recs:
            if not r["is_seen"]:
                bundle_em[r["true_bundle"]].append(r["is_em"])
        jv, ev = [], []
        for _, row in het_df.iterrows():
            b = row["bundle"]
            if b in bundle_em and len(bundle_em[b]) >= min_tokens:
                jv.append(float(row["jaccard_max"]))
                ev.append(float(np.mean(bundle_em[b])))
        n = len(jv)
        base = {"n": n, "n_farkli_jmax": len(set(jv)), "power": None}
        if not recs:
            results_out[short] = {**base, "rho": None, "p_value": None, "reason": "no_records"}
            ax.set_title(f"{m_cfg['name']}\n(veri yok)", fontsize=9)
            continue
        if n < 3 or len(set(ev)) < 2 or len(set(jv)) < 2:
            reason = "insufficient_n" if n < 3 else "sabit_deger"
            results_out[short] = {**base, "rho": None, "p_value": None, "reason": reason}
            ax.set_title(f"{m_cfg['name']}\n(n={n}; ρ hesaplanamaz)", fontsize=9)
            continue
        rho, p_val = spearmanr(jv, ev)
        power = spearman_power(n=n)
        results_out[short] = {**base, "rho": round(float(rho), 4), "p_value": round(float(p_val), 4),
                              "power": None if np.isnan(power) else round(power, 4), "reason": ""}
        ax.scatter(jv, ev, color=m_cfg["color"], alpha=0.85, s=55, zorder=3, edgecolors="black", linewidths=0.4)
        ax.set_title(f"{m_cfg['name']}\nρ = {tr_sayi(rho, 3)}; p = {tr_sayi(p_val, 3)}; n = {n}", fontsize=9)
        ax.set_xlabel("En yakın eğitim demetine Jaccard benzerliği", fontsize=8)
        ax.set_ylabel("Unseen demet EM", fontsize=8)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.08)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: tr_sayi(x, 1)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: tr_sayi(x, 1)))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    SEP = "─" * 90
    print(f"\n{SEP}\n  JACCARD–EM SPEARMAN — {dataset_name} (min {min_tokens} belirteç/demet)")
    print(f"  ⚠️  Küçük n ve yoğun eşitlikler → düşük güç. YALNIZCA KEŞİFSEL.\n{SEP}")
    print(f"  {'Model':<12} {'ρ':>8} {'p':>9} {'n':>5} {'farklı J':>9} {'Güç(ρ=0,5)':>11}")
    for m_cfg in MODELS:
        r = results_out.get(m_cfg["short"], {})
        if r.get("rho") is None:
            print(f"  {m_cfg['name']:<12} {'N/A':>8} {'N/A':>9} {r.get('n', 0):>5} {r.get('n_farkli_jmax', 0):>9} "
                  f"{'N/A':>11}  {r.get('reason', '')}")
        else:
            pw = f"{r['power']:.1%}" if r.get("power") is not None else "N/A"
            print(f"  {m_cfg['name']:<12} {r['rho']:>8.4f} {r['p_value']:>9.4f} {r['n']:>5} "
                  f"{r['n_farkli_jmax']:>9} {pw:>11}")
    print(SEP)
    if save_path:
        pd.DataFrame([{"model": nm.get(s, s), "short": s, **v} for s, v in results_out.items()]).to_csv(
            save_path.rsplit(".", 1)[0] + "_stats.csv", index=False, float_format="%.4f")
    return results_out


# =============================================================================
# 17) YENİDEN ANALİZLER (E0–E10) — em_records üzerinden, EĞİTİM GEREKTİRMEZ
# =============================================================================

@dataclass
class BolunmeBaglami:
    """Bir (veri seti, bölünme tohumu) için analizlerin ihtiyaç duyduğu her şey."""
    ad: str
    split_seed: int
    ham: Dict[str, Any]
    splits: Dict[str, List[Dict]]
    splitter: ControlledSplit
    label_mappings: Dict[str, Dict[str, int]]
    train_bundles: Set[str]
    train_forms: Set[str]
    train_upos_bundles: Set[str]
    upos_ozellik_ciftleri: Set[Tuple[str, str]]

    @property
    def held_out(self) -> Set[str]:
        return self.splitter.held_out_bundles

    def test_baglami(self, sent_idx: int, word_idx: int) -> Tuple[str, str, Dict[str, str]]:
        s = self.splits["test"][sent_idx]
        return s["tokens"][word_idx], s["upos"][word_idx], s["feats"][word_idx]


def bolunme_baglami_kur(ad: str, ham: Dict[str, Any], split_seed: int, sessiz: bool = False,
                        atom_strict: bool = True) -> BolunmeBaglami:
    splits, splitter = build_controlled_split(
        ham["train"], ham["dev"], ham["test"], held_out_ratio=HELD_OUT_RATIO,
        min_bundle_freq=DATASET_MIN_BUNDLE_FREQ.get(ad, 20), split_seed=split_seed,
        atom_strict=atom_strict, sessiz=sessiz)
    tr = splits["train"]
    ciftler = set()
    for s in tr:
        for i in range(len(s["tokens"])):
            for a in bundle_atomlari(_bundle_of(s, i)):
                ciftler.add((s["upos"][i], a.split("=", 1)[0]))
    return BolunmeBaglami(
        ad=ad, split_seed=split_seed, ham=ham, splits=splits, splitter=splitter,
        label_mappings=build_label_mappings(tr, sessiz=True),
        train_bundles={_bundle_of(s, i) for s in tr for i in range(len(s["tokens"]))},
        train_forms={tr_lower(w) for s in tr for w in s["tokens"]},
        train_upos_bundles={f"{s['upos'][i]}+{_bundle_of(s, i)}" for s in tr for i in range(len(s["tokens"]))},
        upos_ozellik_ciftleri=ciftler)


_KAYIT_STR_SUTUNLARI = (["token", "lemma", "feats", "true_label", "pred_label", "true_bundle", "pred_bundle",
                         "seen_group", "model", "short", "dataset", "ud_tag", "anotasyon"]
                        + [f"{s}_{h}" for s in ("true", "pred") for h in MORPH_HEADS])


def kayit_oku(path: str) -> pd.DataFrame:
    """em_records CSV'sini güvenli okur: 'NA', 'nan', '1990' gibi belirteçler metin olarak kalır."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""],
                       dtype={c: str for c in _KAYIT_STR_SUTUNLARI})


def kayitlari_df(kaynak) -> pd.DataFrame:
    """Eski (yalnız true_label/pred_label) ya da yeni biçimdeki kayıtları analiz biçimine getirir.
    UPOS, demet ve başlık sütunları HER SATIR için true_label/pred_label'dan yeniden türetilir
    (iki biçimde de bulunan tek kaynak); böylece eski ve yeni biçim karışsa bile boş sütun kalmaz.
    Yeni biçimde bu türetim, kayıttaki sütunlarla birebir aynıdır."""
    df = kaynak.copy() if isinstance(kaynak, pd.DataFrame) else pd.DataFrame(kaynak)
    if df.empty:
        return df
    for side in ("true", "pred"):
        ub = df[f"{side}_label"].astype(str).map(label_to_upos_bundle)
        df[f"{side}_upos"] = ub.map(lambda x: x[0])
        df[f"{side}_bundle"] = ub.map(lambda x: x[1])
        heads = df[f"{side}_bundle"].map(bundle_to_heads)
        for h in FEAT_HEADS:
            df[f"{side}_{h}"] = heads.map(lambda d, h=h: d[h])
    for col in ("is_seen", "is_em"):
        if df[col].dtype != bool:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "1.0"])
    for col in ("sent_idx", "word_idx"):
        df[col] = df[col].astype(int)
    if "split_seed" not in df.columns:
        df["split_seed"] = SPLIT_SEED
    df["split_seed"] = df["split_seed"].fillna(SPLIT_SEED).astype(int)
    if "seed" in df.columns:
        df["seed"] = df["seed"].astype(int)
    return df


def _gruplar(df: pd.DataFrame):
    return df.groupby(["dataset", "split_seed", "seed", "model"], sort=True)


def _token_results_from_df(g: pd.DataFrame) -> List[Dict]:
    recs = g.to_dict("records")
    return [{"true": {h: r[f"true_{h}"] for h in MORPH_HEADS}, "pred": {h: r[f"pred_{h}"] for h in MORPH_HEADS}}
            for r in recs]


def _seen_group(df: pd.DataFrame, bag: Optional[BolunmeBaglami]) -> pd.Series:
    if "seen_group" in df.columns and df["seen_group"].notna().all() and (df["seen_group"] != "seen").any():
        return df["seen_group"]
    if bag is None:
        return np.where(df["is_seen"], "seen", "unseen")
    return pd.Series(np.where(~df["is_seen"], "unseen",
                              np.where(df["true_bundle"].isin(bag.train_bundles), "seen_train", "seen_novel")),
                     index=df.index)


def e0_e1_metrikler(df: pd.DataFrame, baglamlar: Dict[Tuple[str, int], BolunmeBaglami],
                    eski_json: Optional[Dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """E0: önceki raporlanan sayıların em_records'tan yeniden üretimi (farklar ≈ 0 olmalı).
    E1: düzeltilmiş F1(N−) (+ altın-maske sürümü) ve başlık düzeyinde eski/yeni F1(N−)."""
    rows, head_rows = [], []
    for (ds, sp, seed, model), g in _gruplar(df):
        bag = baglamlar.get((ds, sp))
        if bag is None:
            continue
        m = compute_test_metrics(_token_results_from_df(g), bag.label_mappings)
        grp = _seen_group(g, bag)
        row = {"dataset": ds, "split_seed": sp, "seed": seed, "model": model, "n": len(g),
               "test_avg_f1": m["test_avg_f1"], "test_avg_acc": m["test_avg_acc"],
               "f1_nonone_eski": m["test_nonone_legacy_avg_f1"],
               "f1_nonone_duzeltilmis": m["test_nonone_avg_f1"],
               "f1_nonone_tum_belirtecler": m["test_nonone_tum_avg_f1"],
               "f1_nonone_altin_maske": m["test_nonone_goldmask_avg_f1"],
               "overall_em": g["is_em"].mean(), "seen_em": g.loc[g["is_seen"], "is_em"].mean(),
               "unseen_em": g.loc[~g["is_seen"], "is_em"].mean(),
               "seen_train_em": g.loc[grp == "seen_train", "is_em"].mean(),
               "seen_novel_em": g.loc[grp == "seen_novel", "is_em"].mean(),
               "seen_novel_n": int((grp == "seen_novel").sum())}
        eski_tag, eski_anot, eski_split = REANALIZ_VERI.get(ds, (None, None, SPLIT_SEED))
        ayni_aile = (bag.ham["spec"]["tag"], bag.ham["spec"]["anotasyon"]) == (eski_tag, eski_anot)
        # eski JSON yalnız eski ana bölünmeye ve eski veri ailesine (REANALIZ_VERI) aittir
        if eski_json and ayni_aile and int(sp) == int(eski_split):
            nm_inv = {v: k for k, v in NAME_MAP().items()}
            short = nm_inv.get(model, model)
            for rec in eski_json.get(ds, []):
                if int(rec.get("seed", -1)) == int(seed) and short in rec and isinstance(rec[short], dict):
                    e = rec[short]
                    # v5 JSON'unda eski hesap 'test_nonone_legacy_avg_f1' adıyla durur; eski JSON'da
                    # 'test_nonone_avg_f1' zaten eski (hatalı) hesaptır.
                    eski_nn = e.get("test_nonone_legacy_avg_f1", e.get("test_nonone_avg_f1", float("nan")))
                    row["E0_fark_F1N+"] = row["test_avg_f1"] - e.get("test_avg_f1", float("nan"))
                    row["E0_fark_F1N-_eski"] = row["f1_nonone_eski"] - eski_nn
                    row["E0_fark_Acc"] = row["test_avg_acc"] - e.get("test_avg_acc", float("nan"))
                    row["E0_fark_SeenEM"] = row["seen_em"] - e.get("mh_seen", float("nan"))
                    row["E0_fark_UnseenEM"] = row["unseen_em"] - e.get("mh_unseen", float("nan"))
        rows.append(row)
        for h in MORPH_HEADS:
            head_rows.append({"dataset": ds, "split_seed": sp, "seed": seed, "model": model, "baslik": h,
                              "f1_Nplus": m[f"test_{h}_f1"], "f1_Nminus_eski": m[f"test_{h}_f1_nonone_legacy"],
                              "f1_Nminus_duzeltilmis": m[f"test_{h}_f1_nonone"],
                              "f1_Nminus_tum_belirtecler": m[f"test_{h}_f1_nonone_tum"],
                              "f1_Nminus_altin_maske": m[f"test_{h}_f1_nonone_goldmask"],
                              "dogruluk": m[f"test_{h}_accuracy"]})
    return pd.DataFrame(rows), pd.DataFrame(head_rows)


def e2_unseen_makro_loo(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Belirteç ağırlıklı (mikro) ve demet ortalamalı (makro) Unseen EM; her demeti
    dışarıda bırakarak (leave-one-out) Unseen EM."""
    ozet, loo = [], []
    for (ds, sp, seed, model), g in _gruplar(df):
        u = g[~g["is_seen"]]
        if u.empty:
            continue
        pb = u.groupby("true_bundle")["is_em"].agg(["mean", "size", "sum"]).sort_values(
            "size", ascending=False, kind="mergesort")   # eşitlikte demet adına göre (deterministik)
        top = pb.index[0]
        tot_n, tot_c = pb["size"].sum(), pb["sum"].sum()
        loo_vals = {}
        for b, r in pb.iterrows():
            rest_n = tot_n - r["size"]
            loo_vals[b] = (tot_c - r["sum"]) / rest_n if rest_n else float("nan")
            loo.append({"dataset": ds, "split_seed": sp, "seed": seed, "model": model, "dislanan_demet": b,
                        "demet_belirtec": int(r["size"]), "demet_em": r["mean"], "unseen_em_demet_haric": loo_vals[b]})
        ozet.append({"dataset": ds, "split_seed": sp, "seed": seed, "model": model,
                     "unseen_em_mikro": tot_c / tot_n, "unseen_em_makro": pb["mean"].mean(),
                     "n_demet": len(pb), "n_belirtec": int(tot_n),
                     "en_buyuk_demet": top, "en_buyuk_demet_payi": pb["size"].iloc[0] / tot_n,
                     "ilk3_payi": pb["size"].head(3).sum() / tot_n,
                     "unseen_em_en_buyuk_haric": loo_vals[top],
                     "loo_min": np.nanmin(list(loo_vals.values())), "loo_max": np.nanmax(list(loo_vals.values()))})
    return pd.DataFrame(ozet), pd.DataFrame(loo)


def e3_demet_bootstrap(df: pd.DataFrame) -> pd.DataFrame:
    """Model çiftleri için demet düzeyinde eşleştirilmiş bootstrap (Unseen ve Seen)."""
    rows = []
    order = [m["name"] for m in MODELS]
    for (ds, sp, seed), g in df.groupby(["dataset", "split_seed", "seed"]):
        modeller = [m for m in order if m in set(g["model"])] + sorted(set(g["model"]) - set(order))
        recs = {m: g[g["model"] == m].to_dict("records") for m in modeller}
        ortak = set.intersection(*[{(r["sent_idx"], r["word_idx"]) for r in v} for v in recs.values()]) if recs else set()
        for i in range(len(modeller)):
            for j in range(i + 1, len(modeller)):
                for subset in ("unseen", "seen"):
                    r = eslestirilmis_demet_bootstrap(recs[modeller[i]], recs[modeller[j]], subset=subset,
                                                      common_keys=ortak)
                    rows.append({"dataset": ds, "split_seed": sp, "seed": seed,
                                 "model_a": modeller[i], "model_b": modeller[j], **r})
    return pd.DataFrame(rows)


def e4_bicim_gorulme(df: pd.DataFrame, baglamlar: Dict[Tuple[str, int], BolunmeBaglami]) -> pd.DataFrame:
    """EM'i (Seen/Unseen) × (biçim eğitimde geçiyor / geçmiyor) kırılımında verir.
    Kompozisyonel kısmı ayırmak için Unseen EM, Seen-OOV EM ile kıyaslanmalıdır."""
    rows = []
    for (ds, sp, seed, model), g in _gruplar(df):
        bag = baglamlar.get((ds, sp))
        if "form_in_train" in g.columns and g["form_in_train"].notna().all():
            iv = g["form_in_train"].astype(str).str.lower().isin(["true", "1"])
        elif bag is not None:
            iv = g["token"].astype(str).map(tr_lower).isin(bag.train_forms)
        else:
            continue
        for grp_ad, mask in (("Seen", g["is_seen"]), ("Unseen", ~g["is_seen"])):
            for bic, m2 in (("egitimde_var", iv), ("egitimde_yok", ~iv)):
                sub = g[mask & m2]
                rows.append({"dataset": ds, "split_seed": sp, "seed": seed, "model": model,
                             "grup": grp_ad, "bicim": bic, "n": len(sub),
                             "em": sub["is_em"].mean() if len(sub) else float("nan")})
    return pd.DataFrame(rows)


NOM3SG = "Case=Nom|Number=Sing|Person=3"


def e5_nom_none_upos(df: pd.DataFrame) -> pd.DataFrame:
    """Nom3Sg ↔ NONE demet hatalarında UPOS da yanlış mı?"""
    rows = []
    for (ds, sp, seed, model), g in _gruplar(df):
        for yon, mask in (("Nom3Sg→NONE", (g["true_bundle"] == NOM3SG) & (g["pred_bundle"] == "NONE")),
                          ("NONE→Nom3Sg", (g["true_bundle"] == "NONE") & (g["pred_bundle"] == NOM3SG))):
            sub = g[mask]
            upos_hata = (sub["true_upos"] != sub["pred_upos"])
            ciftler = Counter(zip(sub.loc[upos_hata, "true_upos"], sub.loc[upos_hata, "pred_upos"]))
            rows.append({"dataset": ds, "split_seed": sp, "seed": seed, "model": model, "yon": yon,
                         "n_hata": len(sub), "upos_da_yanlis": int(upos_hata.sum()),
                         "upos_yanlis_payi": float(upos_hata.mean()) if len(sub) else float("nan"),
                         "en_sik_upos_ciftleri": "; ".join(f"{a}→{b}:{c}" for (a, b), c in ciftler.most_common(4)),
                         "gercek_upos_dagilimi": "; ".join(f"{a}:{c}" for a, c in Counter(sub["true_upos"]).most_common(4))})
    return pd.DataFrame(rows)


_RE_DIR = re.compile(r"[dt][ıiuü]r$")


def e6_fazladan_cau(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Gerçek Voice'u olmayan belirteçlere Voice=Cau atanması: hangi biçimler, hangi demetler,
    kaçı -DIr ile bitiyor?"""
    ozet, bicim = [], []
    for (ds, sp, seed, model), g in _gruplar(df):
        sub = g[(g["true_voice"] == "NONE") & (g["pred_voice"] == "Cau")]
        if sub.empty:
            ozet.append({"dataset": ds, "split_seed": sp, "seed": seed, "model": model, "n": 0})
            continue
        low = sub["token"].astype(str).map(tr_lower)
        ozet.append({"dataset": ds, "split_seed": sp, "seed": seed, "model": model, "n": len(sub),
                     "unseen_icinde": int((~sub["is_seen"]).sum()),
                     "DIr_ile_biten": int(low.str.contains(_RE_DIR).sum()),
                     "DIr_payi": float(low.str.contains(_RE_DIR).mean()),
                     "en_sik_demet": sub["true_bundle"].value_counts().index[0],
                     "en_sik_demet_n": int(sub["true_bundle"].value_counts().iloc[0])})
        for (tb, f), c in Counter(zip(sub["true_bundle"], low)).most_common():
            bicim.append({"dataset": ds, "split_seed": sp, "seed": seed, "model": model, "gercek_demet": tb,
                          "bicim": f, "n": c, "DIr_ile_biter": bool(_RE_DIR.search(f)),
                          "unseen_demet": tb not in set(g.loc[g["is_seen"], "true_bundle"])})
    return pd.DataFrame(ozet), pd.DataFrame(bicim)


def _kanonik(feats9: Dict[str, str], form: str, gold_upos: str, gold_feats: Dict[str, str], kurallar: Set[str]) -> Dict[str, str]:
    """E7: BOUN eğitim/test kural farkına karşı eşdeğerlik (hem gerçek hem tahmine uygulanır;
    olgu GERÇEK bağlamdan belirlenir)."""
    f = dict(feats9)
    ol = boun_olgulari(form, gold_upos, gold_feats)
    if "iyor" in kurallar and ol["iyor"] and f["aspect"] == "Imp":
        f["aspect"] = "Prog"
    if "gelecek" in kurallar and ol["gelecek"] and f["aspect"] == "Prosp":
        f["aspect"] = "Imp"
    if "arak" in kurallar and ol["arak"] and f["aspect"] == "Prog" and f["mood"] == "NONE":
        f["aspect"], f["mood"] = "NONE", "Imp"
    if "arak_genis" in kurallar and ol["arak"]:
        if f["aspect"] in ("Prog", "Imp"):
            f["aspect"] = "NONE"
        if f["mood"] == "NONE":
            f["mood"] = "Imp"
    if "ken" in kurallar and ol["ken"]:
        if f["mood"] == "NONE":
            f["mood"] = "Imp"
        if f["aspect"] == "Hab":
            f["aspect"] = "NONE"
    fl = tr_lower(form)
    if ("iyelik" in kurallar and ol["iyelik"] and f["case"] == "Acc" and _RE_DAR_UNLU_SON.search(fl)
            and not _RE_IYELIK_ACC.search(fl)):
        f["case"] = "Nom"
    return f


E7_KURAL_SETLERI = {
    "3_kural_denetim": {"iyor", "gelecek", "arak"},
    "5_kural_genis": {"iyor", "gelecek", "arak_genis", "ken", "iyelik"},
}


def e7_boun_esdegerlik(df: pd.DataFrame, baglamlar: Dict[Tuple[str, int], BolunmeBaglami]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """[E7] BOUN için: (i) ham EM, (ii) eşdeğerlik kurallarıyla EM, (iii) uyumsuzluk-baskın
    held-out demetler dışlanarak Unseen EM. Bu bir ÖN KESTİRİMDİR."""
    rows, baskin_rows = [], []
    for (ds, sp), gds in df[df["dataset"].str.upper().str.startswith("BOUN")].groupby(["dataset", "split_seed"]):
        bag = baglamlar.get((ds, sp))
        if bag is None:
            continue
        spec = bag.ham["spec"]
        m_ = re.match(r"r(\d+)\.(\d+)", str(spec.get("tag", "")))
        surum = (int(m_.group(1)), int(m_.group(2))) if m_ else (0, 0)
        if spec.get("anotasyon") != "ham" or surum < (2, 11):
            print(f"  [E7] {ds}: UD {spec.get('tag')} / anotasyon={spec.get('anotasyon')} — eğitim/test kural farkı "
                  f"beklenmediğinden E7 atlandı (yalnız r2.11+ ham BOUN için anlamlıdır).")
            continue
        # Uyumsuzluk-baskın held-out demetler: test belirteçlerinin ≥%50'si -(I)yor/gelecek/-ArAk/-(y)ken
        say, top = Counter(), Counter()
        for s in bag.splits["test"]:
            for i in range(len(s["tokens"])):
                b = _bundle_of(s, i)
                if b in bag.held_out:
                    top[b] += 1
                    ol = boun_olgulari(s["tokens"][i], s["upos"][i], s["feats"][i])
                    say[b] += int(ol["iyor"] or ol["gelecek"] or ol["arak"] or ol["ken"])
        baskin = {b for b in top if say[b] / top[b] >= 0.5}
        for b in sorted(baskin):
            baskin_rows.append({"dataset": ds, "split_seed": sp, "demet": b, "test_belirtec": top[b],
                                "kural_olgusu_belirtec": say[b]})
        for (seed, model), g in gds.groupby(["seed", "model"]):
            recs = g.to_dict("records")
            sonuc = {"dataset": ds, "split_seed": sp, "seed": seed, "model": model,
                     "seen_em_ham": g.loc[g["is_seen"], "is_em"].mean(),
                     "unseen_em_ham": g.loc[~g["is_seen"], "is_em"].mean(),
                     "uyumsuz_demetler": "; ".join(sorted(baskin))}
            u_rest = g[(~g["is_seen"]) & (~g["true_bundle"].isin(baskin))]
            sonuc["unseen_em_baskin_demetler_haric"] = u_rest["is_em"].mean() if len(u_rest) else float("nan")
            sonuc["unseen_n_baskin_haric"] = len(u_rest)
            for ad_k, kur in E7_KURAL_SETLERI.items():
                em_s, em_u = [], []
                for r in recs:
                    form, gu, gf = bag.test_baglami(r["sent_idx"], r["word_idx"])
                    t9 = _kanonik({h: r[f"true_{h}"] for h in FEAT_HEADS}, form, gu, gf, kur)
                    p9 = _kanonik({h: r[f"pred_{h}"] for h in FEAT_HEADS}, form, gu, gf, kur)
                    ok = (r["true_upos"] == r["pred_upos"]) and all(t9[h] == p9[h] for h in FEAT_HEADS)
                    (em_s if r["is_seen"] else em_u).append(ok)
                sonuc[f"seen_em_{ad_k}"] = float(np.mean(em_s)) if em_s else float("nan")
                sonuc[f"unseen_em_{ad_k}"] = float(np.mean(em_u)) if em_u else float("nan")
            rows.append(sonuc)
    return pd.DataFrame(rows), pd.DataFrame(baskin_rows)


def e8_gecersiz_kombinasyon(df: pd.DataFrame, baglamlar: Dict[Tuple[str, int], BolunmeBaglami]) -> pd.DataFrame:
    """Bağımsız başlıklar geçersiz birleşimler üretebilir. İki oran:
    (i) tahmin edilen UPOS+demet eğitimde hiç geçmiyor (yeni birleşim);
    (ii) tahminde eğitimde HİÇ birlikte görülmemiş (UPOS, özellik) çifti var (geçersiz)."""
    rows = []
    for (ds, sp, seed, model), g in _gruplar(df):
        bag = baglamlar.get((ds, sp))
        if bag is None:
            continue
        yeni = ~g["pred_label"].isin(bag.train_upos_bundles) if "pred_label" in g.columns else \
            ~(g["pred_upos"] + "+" + g["pred_bundle"]).isin(bag.train_upos_bundles)

        def _gecersiz(r):
            return any((r["pred_upos"], FEAT_KEY_MAP[h]) not in bag.upos_ozellik_ciftleri
                       for h in FEAT_HEADS if r[f"pred_{h}"] != "NONE")

        gec = g.apply(_gecersiz, axis=1)
        for grp_ad, mask in (("tum", pd.Series(True, index=g.index)), ("Seen", g["is_seen"]), ("Unseen", ~g["is_seen"])):
            sub = g[mask]
            if sub.empty:
                continue
            rows.append({"dataset": ds, "split_seed": sp, "seed": seed, "model": model, "grup": grp_ad,
                         "n": len(sub), "yeni_birlesim_orani": float(yeni[mask].mean()),
                         "gecersiz_birlesim_orani": float(gec[mask].mean()),
                         "yeni_ve_dogru": int((yeni[mask] & sub["is_em"]).sum())})
    return pd.DataFrame(rows)


def e9_tokenizasyon(bag: BolunmeBaglami, model_listesi: Optional[List[Dict]] = None) -> pd.DataFrame:
    """[E9] Sözcük başına alt-kelime sayısı ve kök–ek sınırı hizalanması (lemma-önek
    sezgiseli: biçim lemma ile başlıyorsa sınır = len(lemma)). Yalnız tokenizatör dosyaları gerekir."""
    rows = []
    kelimeler, gruplar_, lemmalar = [], [], []
    for s in bag.splits["test"]:
        for i, w in enumerate(s["tokens"]):
            kelimeler.append(w)
            lemmalar.append(s["lemmas"][i])
            gruplar_.append("Unseen" if _bundle_of(s, i) in bag.held_out else "Seen")
    for m_cfg in (model_listesi or MODELS):
        try:
            tok = AutoTokenizer.from_pretrained(m_cfg["hf_id"], revision=_rev(m_cfg["hf_id"]))
        except Exception as e:
            logger.warning(f"  [E9] {m_cfg['name']} tokenizatörü yüklenemedi: {e}")
            continue
        enc = tok(kelimeler, add_special_tokens=False, return_offsets_mapping=True)
        stats = defaultdict(lambda: {"n": 0, "parca": 0, "bolunmus": 0, "sinir_n": 0, "sinir_hizali": 0})
        for w, lem, grp, ids, offs in zip(kelimeler, lemmalar, gruplar_, enc["input_ids"], enc["offset_mapping"]):
            st = stats[grp]
            st["n"] += 1
            st["parca"] += len(ids)
            st["bolunmus"] += int(len(ids) > 1)
            wl, ll = tr_lower(w), tr_lower(lem or "")
            if ll and wl.startswith(ll) and len(wl) > len(ll):
                st["sinir_n"] += 1
                bitisler = {e for (s_, e) in offs if 0 < e < len(w)}
                st["sinir_hizali"] += int(len(ll) in bitisler)
        for grp, st in stats.items():
            rows.append({"dataset": bag.ad, "split_seed": bag.split_seed, "model": m_cfg["name"], "grup": grp,
                         "n_sozcuk": st["n"], "sozcuk_basina_altkelime": st["parca"] / max(1, st["n"]),
                         "bolunmus_sozcuk_orani": st["bolunmus"] / max(1, st["n"]),
                         "kok_ek_siniri_hizali_orani": st["sinir_hizali"] / max(1, st["sinir_n"]),
                         "sinir_degerlendirilen": st["sinir_n"]})
    return pd.DataFrame(rows)


def tohum_ozeti(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Kayıtlardan tohum düzeyi EM tablosu + eşleştirilmiş t-testleri (Unseen ve Seen)."""
    em = []
    for (ds, sp, seed, model), g in _gruplar(df):
        em.append({"dataset": ds, "split_seed": sp, "seed": seed, "model": model,
                   "seen_em": g.loc[g["is_seen"], "is_em"].mean(), "unseen_em": g.loc[~g["is_seen"], "is_em"].mean()})
    em = pd.DataFrame(em)
    testler = []
    nm_inv = {v: k for k, v in NAME_MAP().items()}
    for (ds, sp), g in em.groupby(["dataset", "split_seed"]):
        for metrik in ("unseen_em", "seen_em"):
            deger = defaultdict(dict)
            for _, r in g.iterrows():
                deger[nm_inv.get(r["model"], r["model"])][int(r["seed"])] = float(r[metrik])
            t = tohum_duzeyi_testler(deger, etiket=metrik)
            if len(t):
                t.insert(0, "split_seed", sp)
                t.insert(0, "dataset", ds)
                testler.append(t)
    return em, (pd.concat(testler, ignore_index=True) if testler else pd.DataFrame())


def reanaliz_calistir(df: pd.DataFrame, baglamlar: Dict[Tuple[str, int], BolunmeBaglami],
                      alt_dizin: str = "reanaliz", eski_json: Optional[Dict] = None,
                      e9: bool = False) -> Dict[str, pd.DataFrame]:
    """E0–E10 analizlerini çalıştırır ve CSV olarak kaydeder."""
    df = kayitlari_df(df)
    out: Dict[str, pd.DataFrame] = {}
    SEP = "═" * 100
    print(f"\n{SEP}\n  YENİDEN ANALİZ (E0–E10) — {df['dataset'].nunique()} veri seti, "
          f"{df.groupby(['dataset', 'split_seed', 'seed', 'model']).ngroups} koşum, {len(df):,} belirteç kaydı\n{SEP}")

    def _kaydet(ad, d):
        if d is not None and len(d):
            d.to_csv(cikti_yolu(alt_dizin, f"{ad}.csv"), index=False, float_format="%.5f")
        out[ad] = d

    e01, e1h = e0_e1_metrikler(df, baglamlar, eski_json)
    _kaydet("E0_E1_metrikler", e01)
    _kaydet("E1_baslik_f1_nonone", e1h)
    if len(e01):
        print(f"\n  [E1] F1(N−): eski (hatalı) → düzeltilmiş ({F1_NONONE_YONTEMI})  |  Seen-yeni EM (held-out değil, eğitimde yok)")
        for _, r in e01.iterrows():
            e0 = ""
            if "E0_fark_F1N+" in r and not pd.isna(r.get("E0_fark_F1N+")):
                e0 = (f"  E0 fark (F1N+, F1N−eski, SeenEM, UnseenEM) = {r['E0_fark_F1N+']:+.5f}, "
                      f"{r['E0_fark_F1N-_eski']:+.5f}, {r['E0_fark_SeenEM']:+.5f}, {r['E0_fark_UnseenEM']:+.5f}")
            print(f"    {r['dataset']:<8} s{r['split_seed']} t{r['seed']:<3} {r['model']:<8} F1(N−) "
                  f"{r['f1_nonone_eski']:.2%} → {r['f1_nonone_duzeltilmis']:.2%} (tüm belirteçler {r['f1_nonone_tum_belirtecler']:.2%}; "
                  f"altın-maske {r['f1_nonone_altin_maske']:.2%})"
                  f"  Seen-yeni EM {r['seen_novel_em']:.1%} (n={r['seen_novel_n']}){e0}")
    e2, e2l = e2_unseen_makro_loo(df)
    _kaydet("E2_unseen_makro_loo", e2)
    _kaydet("E2_leave_one_out", e2l)
    if len(e2):
        print("\n  [E2] Unseen EM: mikro (belirteç) / makro (demet) / en büyük demet hariç")
        for _, r in e2.iterrows():
            print(f"    {r['dataset']:<8} s{r['split_seed']} t{r['seed']:<3} {r['model']:<8} {r['unseen_em_mikro']:.1%} / "
                  f"{r['unseen_em_makro']:.1%} / {r['unseen_em_en_buyuk_haric']:.1%}  (en büyük demet payı "
                  f"%{100 * r['en_buyuk_demet_payi']:.1f}; LOO aralığı {r['loo_min']:.1%}–{r['loo_max']:.1%})")
    e3 = e3_demet_bootstrap(df)
    _kaydet("E3_demet_eslestirilmis_bootstrap", e3)
    if len(e3):
        print("\n  [E3] Demet düzeyinde eşleştirilmiş bootstrap — Unseen EM farkı (A − B) [%95 GA], Wilcoxon (demet başına)")
        for _, r in e3[e3["subset"] == "unseen"].iterrows():
            print(f"    {r['dataset']:<8} s{r['split_seed']} t{r['seed']:<3} {r['model_a']:>8} − {r['model_b']:<8} "
                  f"{100 * r['fark']:+6.2f} puan [{100 * r['ga_alt']:+.2f}; {100 * r['ga_ust']:+.2f}]  p_boot={r['p_boot']:.3f}  "
                  f"makro {100 * r['makro_fark']:+.2f}  Wilcoxon p={r['wilcoxon_p']:.3f}  (n_demet={r['n_demet']})")
    e4 = e4_bicim_gorulme(df, baglamlar)
    _kaydet("E4_bicim_gorulme_em", e4)
    if len(e4):
        piv = e4.groupby(["dataset", "split_seed", "model", "grup", "bicim"])["em"].mean().unstack(["grup", "bicim"])
        print("\n  [E4] EM: Seen/Unseen × biçim eğitimde var/yok (tohum ortalaması)")
        print(piv.to_string(float_format=lambda x: f"{x:.1%}"))
    e5 = e5_nom_none_upos(df)
    _kaydet("E5_nom3sg_none_upos", e5)
    e6, e6b = e6_fazladan_cau(df)
    _kaydet("E6_fazladan_cau_ozet", e6)
    _kaydet("E6_fazladan_cau_bicimler", e6b)
    if len(e6) and "DIr_ile_biten" in e6.columns:
        print("\n  [E6] Gerçek Voice'u olmayan belirtece Voice=Cau atanması")
        for _, r in e6[e6["n"] > 0].iterrows():
            print(f"    {r['dataset']:<8} s{r['split_seed']} t{r['seed']:<3} {r['model']:<8} n={int(r['n']):<4} Unseen içinde "
                  f"{int(r['unseen_icinde'])}; -DIr ile biten {int(r['DIr_ile_biten'])} (%{100 * r['DIr_payi']:.0f})")
    e7, e7b = e7_boun_esdegerlik(df, baglamlar)
    _kaydet("E7_boun_esdegerlik", e7)
    _kaydet("E7_uyumsuzluk_baskin_demetler", e7b)
    if len(e7):
        print("\n  [E7] BOUN: Unseen EM ham → 3 kural → 5 kural → uyumsuzluk-baskın demetler hariç")
        for _, r in e7.iterrows():
            print(f"    {r['dataset']:<8} s{r['split_seed']} t{r['seed']:<3} {r['model']:<8} {r['unseen_em_ham']:.1%} → "
                  f"{r['unseen_em_3_kural_denetim']:.1%} → {r['unseen_em_5_kural_genis']:.1%} → "
                  f"{r['unseen_em_baskin_demetler_haric']:.1%} (n={r['unseen_n_baskin_haric']})")
    e8 = e8_gecersiz_kombinasyon(df, baglamlar)
    _kaydet("E8_gecersiz_kombinasyon", e8)
    em_tohum, t_test = tohum_ozeti(df)
    _kaydet("tohum_em", em_tohum)
    _kaydet("tohum_duzeyi_eslestirilmis_t", t_test)
    if len(t_test):
        print("\n  [Bölüm 8] Tohum düzeyinde eşleştirilmiş t-testi (Unseen EM, puan) — yalnız eğitim varyansı")
        for _, r in t_test[t_test["karsilastirma"] == "unseen_em"].iterrows():
            if r["n"] >= 2:
                print(f"    {r['dataset']:<8} s{r['split_seed']} {r['model_a']:>8} − {r['model_b']:<8} "
                      f"{100 * r['ort_fark']:+.2f} [{100 * r['ga_alt']:+.2f}; {100 * r['ga_ust']:+.2f}]  p={r['p']:.3f}")
    # E10 ve E9 bağlamdan
    e10 = []
    for (ds, sp), bag in baglamlar.items():
        tm = compute_majority_class_baseline(bag.splits["test"], bag.splits["train"], bag.held_out, sessiz=True)
        e10.append({"dataset": ds, "split_seed": sp, **tm})
    _kaydet("E10_en_sik_etiket_temel_modeli", pd.DataFrame(e10))
    if e9 and E9_TOKENIZASYON_ANALIZI:
        e9_list = [e9_tokenizasyon(bag) for bag in baglamlar.values()]
        e9_list = [x for x in e9_list if len(x)]
        if e9_list:
            e9df = pd.concat(e9_list, ignore_index=True)
            _kaydet("E9_tokenizasyon", e9df)
            print("\n  [E9] Tokenizasyon: sözcük başına alt-kelime / bölünmüş sözcük / kök–ek sınırı hizalı")
            print(e9df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n  Yeniden analiz çıktıları: {os.path.join(CIKTI_DIZINI, alt_dizin)}/\n{SEP}")
    return out


# =============================================================================
# 18) GRAFİKLER — tez biçimi: beyaz zemin, Türkçe etiket, ondalık virgül
# =============================================================================

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "savefig.facecolor": "white", "figure.facecolor": "white"})

BASLIK_ADLARI_TR = {"upos": "UPOS", "case": "Durum\n(Case)", "number": "Sayı\n(Number)",
                    "person": "Kişi\n(Person)", "tense": "Zaman\n(Tense)", "aspect": "Görünüş\n(Aspect)",
                    "mood": "Kip\n(Mood)", "voice": "Çatı\n(Voice)", "polarity": "Olumsuzluk\n(Polarity)",
                    "verbform": "Fiil biçimi\n(VerbForm)"}


def tez_stili(fig, axes) -> None:
    fig.patch.set_facecolor("white")
    for ax in (np.ravel(axes) if isinstance(axes, (list, tuple, np.ndarray)) else [axes]):
        ax.set_facecolor("white")
        ax.tick_params(colors="black")
        for sp_ in ax.spines.values():
            sp_.set_edgecolor("#444444")
        ax.grid(alpha=0.35, color="#bbbbbb", axis="y", linestyle=":")
        ax.set_axisbelow(True)


def _yuzde_ekseni(ax, ust: float = 1.0):
    """Yüzde ekseni: tikler en fazla %100'e kadar; üstteki boşluk yalnız değer etiketleri içindir."""
    ax.set_ylim(0, ust)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "%" + tr_sayi(100 * y, 0)))


def _kaydet_sekil(fig, save_path: Optional[str]):
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  [Grafik] {save_path} kaydedildi.")
    plt.close(fig)


def plot_per_head_breakdown(breakdown_per_model, dataset_name: str, save_path: str = None):
    shorts = [m["short"] for m in MODELS if m["short"] in breakdown_per_model]
    if not shorts:
        return
    x = np.arange(len(MORPH_HEADS))
    w = 0.8 / len(shorts)
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    tez_stili(fig, axes)
    for ai, (key, baslik) in enumerate([("seen_acc", "Seen (held-out olmayan demetler)"),
                                        ("unseen_acc", "Unseen (held-out demetler)")]):
        ax = axes[ai]
        for i, s in enumerate(shorts):
            m = next(mm for mm in MODELS if mm["short"] == s)
            vals = [breakdown_per_model[s][h][key] for h in MORPH_HEADS]
            ax.bar(x - 0.4 + w / 2 + i * w, vals, w * 0.95, label=m["name"], color=m["color"],
                   hatch=m.get("hatch", ""), edgecolor="black", linewidth=0.4)
        ax.set_title(f"{dataset_name} — başlık başına doğruluk: {baslik}")
        ax.set_ylabel("Doğruluk")
        _yuzde_ekseni(ax, 1.05)
        ax.legend(loc="lower left", fontsize=8, ncol=len(shorts))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([BASLIK_ADLARI_TR[h] for h in MORPH_HEADS], fontsize=8)
    _kaydet_sekil(fig, save_path)


def _cubuk_grafik(ax, degerler, hatalar=None, ust=1.15):
    x = np.arange(len(MODELS))
    bars = ax.bar(x, degerler, 0.6, color=[m["color"] for m in MODELS],
                  hatch=None, edgecolor="black", linewidth=0.5, zorder=3)
    for b, m in zip(bars, MODELS):
        b.set_hatch(m.get("hatch", ""))
    if hatalar is not None:
        ax.errorbar(x, degerler, yerr=hatalar, fmt="none", color="black", capsize=5, linewidth=1.2, zorder=4)
    for i, v in enumerate(degerler):
        e = hatalar[i] if hatalar is not None else 0.0
        etiket = tr_yuzde(v) + (f"\n±{tr_sayi(100 * e, 1)}" if hatalar is not None else "")
        ax.annotate(etiket, xy=(x[i], v + e), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([m["name"] for m in MODELS])
    _yuzde_ekseni(ax, ust)


def plot_model_comparison_bar(all_results, title_suffix="", save_path=None):
    metrics = [("test_avg_f1", "Ort. F1 (NONE dahil)"), ("test_nonone_avg_f1", "Ort. F1 (NONE hariç, düzeltilmiş)"),
               ("mh_seen", "Seen EM"), ("mh_unseen", "Unseen EM")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    tez_stili(fig, axes)
    for ax, (key, title) in zip(axes, metrics):
        _cubuk_grafik(ax, [all_results.get(m["short"], {}).get(key, 0.0) or 0.0 for m in MODELS])
        ax.set_title(title)
    fig.suptitle(f"Model karşılaştırması{title_suffix}", y=1.02)
    _kaydet_sekil(fig, save_path)


def plot_head_f1_multimodel(head_metrics_per_model, title_suffix="", save_path=None):
    x = np.arange(len(MORPH_HEADS))
    w = 0.8 / len(MODELS)
    fig, ax = plt.subplots(figsize=(16, 5))
    tez_stili(fig, ax)
    for i, m in enumerate(MODELS):
        vals = [head_metrics_per_model.get(m["short"], {}).get(h, 0.0) for h in MORPH_HEADS]
        ax.bar(x - 0.4 + w / 2 + i * w, vals, w * 0.95, label=m["name"], color=m["color"],
               hatch=m.get("hatch", ""), edgecolor="black", linewidth=0.4)
    ax.set_title(f"Başlık başına makro F1 (test, NONE dahil){title_suffix}")
    ax.set_ylabel("Makro F1")
    ax.set_xticks(x)
    ax.set_xticklabels([BASLIK_ADLARI_TR[h] for h in MORPH_HEADS], fontsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: tr_sayi(y, 1)))
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    _kaydet_sekil(fig, save_path)


def plot_training_curves_multimodel(all_log_histories, dataset_name="", save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    tez_stili(fig, axes)
    for m in MODELS:
        hist = all_log_histories.get(m["short"], [])
        tr = [(e["step"], e["loss"]) for e in hist if "loss" in e and "eval_loss" not in e]
        ev = [(e["epoch"], e["eval_loss"]) for e in hist if "eval_loss" in e]
        if tr:
            axes[0].plot(*zip(*tr), color=m["color"], linewidth=1.6, label=m["name"])
        if ev:
            axes[1].plot(*zip(*ev), color=m["color"], linewidth=1.8, marker="o", markersize=4, label=m["name"])
    for ax, title, xlabel in zip(axes, ["Eğitim kaybı", "Doğrulama kaybı"], ["Adım", "Epoch"]):
        ax.set_title(f"{title} — {dataset_name}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Kayıp")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: tr_sayi(y, 2)))
        ax.grid(alpha=0.35, color="#bbbbbb", linestyle=":")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    _kaydet_sekil(fig, save_path)


def plot_seen_unseen_em_bar(em_results_all, dataset_name="", save_path=None):
    metrics = [("overall_em", "Genel EM"), ("seen_em", "Seen EM"), ("unseen_em", "Unseen EM")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    tez_stili(fig, axes)
    for ax, (key, title) in zip(axes, metrics):
        _cubuk_grafik(ax, [em_results_all.get(m["short"], {}).get(key, 0.0) or 0.0 for m in MODELS])
        ax.set_title(title)
    fig.suptitle(f"Tam eşleşme (EM) — {dataset_name}", y=1.02)
    _kaydet_sekil(fig, save_path)


def plot_multi_seed_results(seed_results, dataset_name: str = "", save_path=None):
    """Şekil 4.1–4.2 karşılığı: ortalama ± örneklem std (ddof = 1) hata çubukları."""
    if not seed_results:
        return
    metrics = [("test_avg_f1", "Ort. F1 (NONE dahil)"), ("test_nonone_avg_f1", "Ort. F1 (NONE hariç)"),
               ("mh_seen", "Seen EM"), ("mh_unseen", "Unseen EM")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    tez_stili(fig, axes)
    for ax, (met, title) in zip(axes, metrics):
        means, sds = [], []
        for m in MODELS:
            vals = [rec[m["short"]][met] for rec in seed_results
                    if isinstance(rec.get(m["short"]), dict) and rec[m["short"]].get(met) is not None]
            means.append(float(np.mean(vals)) if vals else 0.0)
            sds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        _cubuk_grafik(ax, means, sds, ust=1.2)
        ax.set_title(title)
    fig.suptitle(f"{dataset_name} — {len(seed_results)} eğitim tohumu (ort. ± std, ddof = 1)", y=1.02)
    _kaydet_sekil(fig, save_path)


def plot_split_variance(split_df: pd.DataFrame, dataset_name: str, save_path=None):
    """Bölünme tohumları arasında Unseen EM (her nokta = bir bölünmenin tohum ortalaması)."""
    if split_df is None or split_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    tez_stili(fig, ax)
    for i, m in enumerate(MODELS):
        d = split_df[split_df["model"] == m["name"]]
        if d.empty:
            continue
        xs = np.full(len(d), i) + np.linspace(-0.12, 0.12, len(d))
        ax.scatter(xs, d["unseen_em_mean"], color=m["color"], edgecolors="black", s=45, zorder=3)
        ax.hlines(d["unseen_em_mean"].mean(), i - 0.25, i + 0.25, color="black", linewidth=2, zorder=4)
        for xx, (_, r) in zip(xs, d.iterrows()):
            ax.annotate(str(int(r["split_seed"])), (xx, r["unseen_em_mean"]), xytext=(4, 0),
                        textcoords="offset points", fontsize=7)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([m["name"] for m in MODELS])
    ax.set_ylabel("Unseen EM (tohum ortalaması)")
    ax.set_title(f"{dataset_name} — bölünme tohumlarına göre Unseen EM")
    _yuzde_ekseni(ax, 1.0)
    _kaydet_sekil(fig, save_path)


def plot_dative_locative_confusion(token_results, model_name, dataset_name, save_path=None):
    true_cases = [r["true"]["case"] for r in token_results if r["true"]["case"] != "NONE"]
    pred_cases = [r["pred"]["case"] for r in token_results if r["true"]["case"] != "NONE"]
    if not true_cases:
        print(f"  [UYARI] {model_name}: Case karışıklık matrisi için veri yok.")
        return
    labels = sorted(set(true_cases) | set(pred_cases))
    li = {l: i for i, l in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(true_cases, pred_cases):
        mat[li[t], li[p]] += 1
    rs = mat.sum(axis=1, keepdims=True)
    norm = np.divide(mat, rs, out=np.zeros_like(mat, dtype=float), where=rs > 0)   # uyarısız bölme
    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(labels) * 1.4), max(5.5, len(labels) * 0.9)))
    tez_stili(fig, axes)
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Greys", vmin=0, vmax=1, xticklabels=labels, yticklabels=labels,
                linewidths=0.4, linecolor="#999999", ax=axes[0], cbar_kws={"shrink": 0.8})
    axes[0].set_title("Satır bazında normalize")
    focus = [l for l in ["Dat", "Loc", "Nom", "Acc", "Gen", "Abl", "Ins"] if l in li]
    fi = [li[l] for l in focus]
    sub = mat[np.ix_(fi, fi)]
    sub_n = np.divide(sub, sub.sum(axis=1, keepdims=True), out=np.zeros_like(sub, dtype=float),
                      where=sub.sum(axis=1, keepdims=True) > 0)
    sns.heatmap(sub_n, annot=True, fmt=".2f", cmap="Greys", vmin=0, vmax=1, xticklabels=focus, yticklabels=focus,
                linewidths=0.4, linecolor="#999999", ax=axes[1], cbar_kws={"shrink": 0.8})
    axes[1].set_title("Yönelme (Dat) – Bulunma (Loc) odaklı")
    for ax in axes:
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gerçek")
        for t in ax.texts:
            t.set_text(t.get_text().replace(".", ","))
        cb = ax.collections[0].colorbar if ax.collections else None
        if cb is not None:
            cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: tr_sayi(y, 1)))
    fig.suptitle(f"Durum (Case) karışıklık matrisi — {model_name} [{dataset_name}]", y=1.02)
    if "Dat" in li and "Loc" in li:
        d, l = li["Dat"], li["Loc"]
        print(f"  [{model_name}] Dat→Loc: {mat[d, l]}/{mat[d].sum()} ({mat[d, l] / max(1, mat[d].sum()):.1%})  "
              f"Loc→Dat: {mat[l, d]}/{mat[l].sum()} ({mat[l, d] / max(1, mat[l].sum()):.1%})")
    _kaydet_sekil(fig, save_path)


# =============================================================================
# 19) HUGGING FACE HUB VE DRIVE YEDEKLEME
# =============================================================================

def _hf_api():
    try:
        from huggingface_hub import HfApi
        return HfApi()
    except Exception as e:  # pragma: no cover
        logger.warning(f"huggingface_hub kullanılamıyor: {e}")
        return None


def hf_model_yukle(model, tokenizer, label_mappings: Dict, bilgi: Dict, repo_id: str) -> bool:
    api = _hf_api()
    if api is None:
        return False
    clean_dir = os.path.join(CHECKPOINT_KOK, "_yukleme", repo_id.replace("/", "__"))
    shutil.rmtree(clean_dir, ignore_errors=True)
    os.makedirs(clean_dir, exist_ok=True)
    model.save_pretrained(clean_dir)
    tokenizer.save_pretrained(clean_dir)
    json_yaz(label_mappings, os.path.join(clean_dir, "label_mappings.json"))
    json_yaz(bilgi, os.path.join(clean_dir, "kosu_bilgisi.json"))
    with open(os.path.join(clean_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"# {repo_id}\n\nTürkçe çok-başlı morfolojik etiketleyici (tez deney kodu v5).\n\n"
                f"```json\n{json.dumps({k: v for k, v in bilgi.items() if k != 'init_raporu'}, ensure_ascii=False, indent=2, default=str)}\n```\n")
    ok = False
    for deneme in range(3):
        try:
            api.create_repo(repo_id, private=HF_DEPO_GIZLI, exist_ok=True)
            with contextlib.suppress(Exception):
                api.update_repo_settings(repo_id=repo_id, private=HF_DEPO_GIZLI)
            api.upload_folder(folder_path=clean_dir, repo_id=repo_id,
                              commit_message=f"v5 koşumu: {bilgi.get('slug', '')}")
            ok = True
            print(f"  [HF] Yüklendi: https://huggingface.co/{repo_id}  ({'gizli' if HF_DEPO_GIZLI else 'herkese açık'})")
            break
        except Exception as e:
            logger.warning(f"  HF model yükleme {deneme + 1}/3 başarısız ({repo_id}): {e}")
            time.sleep(10 * (deneme + 1))
    shutil.rmtree(clean_dir, ignore_errors=True)
    return ok


def hf_sonuclari_yukle(mesaj: str = "sonuçlar") -> bool:
    """Sonuç klasörünü yükler: .csv, .csv.gz (em_records), .json, .png, .log, .txt, .md."""
    if not HF_SONUC_YUKLE:
        return False
    api = _hf_api()
    if api is None:
        return False
    try:
        api.create_repo(HF_SONUC_DEPOSU, repo_type="dataset", private=HF_DEPO_GIZLI, exist_ok=True)
        api.upload_folder(folder_path=CIKTI_DIZINI, repo_id=HF_SONUC_DEPOSU, repo_type="dataset",
                          allow_patterns=["*.csv", "*.csv.gz", "*.json", "*.png", "*.log", "*.txt", "*.md"],
                          commit_message=mesaj)
        print(f"  [HF] Sonuçlar yüklendi → https://huggingface.co/datasets/{HF_SONUC_DEPOSU}")
        return True
    except Exception as e:
        logger.warning(f"  HF sonuç yükleme başarısız: {e}  (dosyalar yerelde duruyor: {CIKTI_DIZINI})")
        return False


def hf_oturum_kontrol() -> bool:
    """Yükleme açıksa Hub oturumunu BAŞTA bir kez denetler; oturum yoksa yüklemeleri kapatır
    (aksi hâlde her modelde 3 başarısız deneme ile zaman kaybedilir)."""
    global HF_MODEL_YUKLE, HF_SONUC_YUKLE, HF_ARTIMLI_YEDEK
    if not (HF_MODEL_YUKLE or HF_SONUC_YUKLE):
        return False
    api = _hf_api()
    try:
        kim = api.whoami() if api is not None else None
        print(f"  [HF] Oturum: {kim.get('name') if isinstance(kim, dict) else kim}")
        return True
    except Exception as e:
        print(f"  ⚠️  [HF] Hub oturumu yok ya da erişilemiyor ({type(e).__name__}); model/sonuç yüklemeleri KAPATILDI. "
              f"Açmak için: from huggingface_hub import login; login()")
        HF_MODEL_YUKLE = HF_SONUC_YUKLE = HF_ARTIMLI_YEDEK = False
        return False


def _drive_dizini_coz() -> Optional[str]:
    """DRIVE_YEDEK_DIZINI = "otomatik" → Colab'da Drive bağlıysa MyDrive altında aynı adlı klasör."""
    global DRIVE_YEDEK_DIZINI
    if DRIVE_YEDEK_DIZINI == "otomatik":
        if "google.colab" in sys.modules and os.path.isdir("/content/drive/MyDrive"):
            DRIVE_YEDEK_DIZINI = os.path.join("/content/drive/MyDrive", os.path.basename(os.path.abspath(CIKTI_DIZINI)))
        else:
            DRIVE_YEDEK_DIZINI = None
            if "google.colab" in sys.modules:
                print("  ℹ  Drive bağlı değil; VM yenilenirse sonuçlar yalnız HF sonuç deposundan geri yüklenebilir. "
                      "Önerilen: from google.colab import drive; drive.mount('/content/drive')")
    return DRIVE_YEDEK_DIZINI


def _artimli_kopya(kaynak: str, hedef: str) -> int:
    """Yalnız yeni/değişmiş dosyaları kopyalar (boyut ya da değiştirilme zamanı farklıysa)."""
    n = 0
    for kok, _, dosyalar in os.walk(kaynak):
        for ad_ in dosyalar:
            if ".tmp" in ad_:
                continue
            s = os.path.join(kok, ad_)
            d = os.path.join(hedef, os.path.relpath(s, kaynak))
            try:
                ss = os.stat(s)
                if os.path.exists(d):
                    ds = os.stat(d)
                    if ds.st_size == ss.st_size and ds.st_mtime >= ss.st_mtime:
                        continue
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copy2(s, d)
                n += 1
            except OSError as e:
                logger.warning(f"  Kopyalanamadı: {s} ({e})")
    return n


def drive_yedekle() -> None:
    _drive_dizini_coz()
    if not DRIVE_YEDEK_DIZINI:
        return
    try:
        n = _artimli_kopya(CIKTI_DIZINI, DRIVE_YEDEK_DIZINI)
        print(f"  [Yedek] {n} yeni/değişen dosya → {DRIVE_YEDEK_DIZINI}")
    except Exception as e:
        logger.warning(f"  Drive yedeği alınamadı: {e}")


def sonuclari_geri_yukle() -> None:
    """Yerelde tamamlanmış koşum yoksa (ör. Colab VM yenilendi) 'kosular' klasörünü önce Drive
    yedeğinden, yoksa HF sonuç deposundan geri yükler. Yeniden kullanım yine imza denetiminden geçer."""
    _drive_dizini_coz()
    yerel = os.path.join(CIKTI_DIZINI, "kosular")
    if os.path.isdir(yerel) and os.listdir(yerel):
        return
    if DRIVE_YEDEK_DIZINI and os.path.isdir(os.path.join(DRIVE_YEDEK_DIZINI, "kosular")):
        n = _artimli_kopya(os.path.join(DRIVE_YEDEK_DIZINI, "kosular"), yerel)
        print(f"  [Geri yükleme] Drive yedeğinden {n} dosya geri yüklendi → tamamlanan koşumlar atlanacak.")
        return
    if HF_SONUC_YUKLE:
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=HF_SONUC_DEPOSU, repo_type="dataset", local_dir=CIKTI_DIZINI,
                              allow_patterns=["kosular/*"])
            n = sum(len(f) for _, _, f in os.walk(yerel)) if os.path.isdir(yerel) else 0
            if n:
                print(f"  [Geri yükleme] HF sonuç deposundan {n} dosya geri yüklendi → tamamlanan koşumlar atlanacak.")
        except Exception as e:
            print(f"  [Geri yükleme] Geri yüklenecek önceki sonuç bulunamadı ({type(e).__name__}); sıfırdan başlanıyor.")


def artimli_yedek(mesaj: str) -> None:
    drive_yedekle()
    if HF_ARTIMLI_YEDEK:
        hf_sonuclari_yukle(mesaj)


def hf_depo_gorunurlugu_ayarla(repo_idler: List[str], gizli: bool, repo_type: Optional[str] = None) -> None:
    """Yardımcı: mevcut depoların görünürlüğünü değiştirir. Otomatik ÇAĞRILMAZ."""
    api = _hf_api()
    for r in repo_idler:
        try:
            api.update_repo_settings(repo_id=r, private=gizli, repo_type=repo_type)
            print(f"  [HF] {r}: {'gizli' if gizli else 'herkese açık'}")
        except Exception as e:
            print(f"  [HF] {r}: değiştirilemedi ({e})")


# =============================================================================
# 20) TEK (VERİ SETİ, BÖLÜNME, TOHUM) BLOĞU
# =============================================================================

def _veri_etiketi(ad: str, bag: BolunmeBaglami) -> str:
    """Dosya/klasör adlarında veri kimliği: veri seti + UD etiketi + anotasyon modu."""
    sp_ = bag.ham["spec"]
    return f"{ad.lower()}_{sp_['tag']}_{sp_['anotasyon']}"


def _slug(ad: str, bag: BolunmeBaglami, seed: int) -> str:
    return f"{_veri_etiketi(ad, bag)}_split{bag.split_seed}_seed{seed}"


def kosu_imzasi(m_cfg: Dict, bag: BolunmeBaglami, model_turu: str) -> Dict[str, Any]:
    """Sürdürmede bir koşumun yeniden kullanılabilmesi için BİREBİR eşleşmesi gereken ayarlar."""
    return {
        "kod_surumu": KOD_SURUMU, "model_turu": model_turu, "hf_id": m_cfg["hf_id"],
        "ud_tag": bag.ham["spec"]["tag"], "anotasyon": bag.ham["spec"]["anotasyon"],
        "ud_commit": bag.ham["depo"].get("commit"), "split_seed": bag.split_seed,
        "held_out_ozeti": hashlib.sha1("|".join(sorted(bag.held_out)).encode()).hexdigest(),
        "HIZLI_TEST": bool(HIZLI_TEST), "HIZLI_TEST_CUMLE": list(HIZLI_TEST_CUMLE) if HIZLI_TEST else None,
        "NUM_TRAIN_EPOCHS": NUM_TRAIN_EPOCHS, "LEARNING_RATE": LEARNING_RATE, "TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE,
        "EVAL_BATCH_SIZE": EVAL_BATCH_SIZE, "WARMUP_RATIO": WARMUP_RATIO, "WEIGHT_DECAY": WEIGHT_DECAY,
        "DROPOUT_RATE": DROPOUT_RATE, "MAX_SEQ_LENGTH": MAX_SEQ_LENGTH, "OPTIMIZER": OPTIMIZER,
        "LR_SCHEDULER": LR_SCHEDULER, "EARLY_STOPPING_METRIC": EARLY_STOPPING_METRIC,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE, "COMPLETION_MIN_EPOCHS": COMPLETION_MIN_EPOCHS,
        "F1_NONONE_YONTEMI": F1_NONONE_YONTEMI, "SUBWORD_STRATEGY": SUBWORD_STRATEGY,
        "HELD_OUT_RATIO": HELD_OUT_RATIO, "BOOTSTRAP_N_RESAMPLES": BOOTSTRAP_N_RESAMPLES,
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED, "ALPHA": ALPHA,
    }


def _imza_esit(a: Optional[Dict], b: Dict) -> bool:
    return a is not None and json.loads(json.dumps(a)) == json.loads(json.dumps(b))


def tamamlanmis_kayit_yolu(slug: str, m_cfg: Dict, bag: BolunmeBaglami, model_turu: str = "multihead") -> Optional[str]:
    """Kayıt dosyası YALNIZ özet JSON'u mevcut ve imzası şimdiki ayarlarla aynıysa döndürülür."""
    on_ek = "" if model_turu == "multihead" else "birlesik_"
    kdir = os.path.join(CIKTI_DIZINI, "kosular", slug)
    ozet_p = os.path.join(kdir, f"{on_ek}{m_cfg['short']}_ozet.json")
    kayit_p = os.path.join(kdir, f"{on_ek}em_records_{m_cfg['short']}.csv.gz")
    if not (os.path.exists(ozet_p) and os.path.exists(kayit_p)):
        return None
    try:
        with open(ozet_p, encoding="utf-8") as f:
            imza = json.load(f).get("kosu_imzasi")
    except (OSError, ValueError):
        return None
    return kayit_p if _imza_esit(imza, kosu_imzasi(m_cfg, bag, model_turu)) else None


def _kayit_meta(df: pd.DataFrame, ad: str, bag: BolunmeBaglami, seed: int, m_cfg: Dict) -> pd.DataFrame:
    df = df.copy()
    df["model"] = m_cfg["name"]
    df["short"] = m_cfg["short"]
    df["dataset"] = ad
    df["seed"] = seed
    df["split_seed"] = bag.split_seed
    df["ud_tag"] = bag.ham["spec"]["tag"]
    df["anotasyon"] = bag.ham["spec"]["anotasyon"]
    return df


def _records_from_csv(path: str) -> List[Dict]:
    d = kayitlari_df(kayit_oku(path))
    return d.to_dict("records")


def _veri_alt_kumeleri(bag: BolunmeBaglami) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    tr, dv, ts = bag.splits["train"], bag.splits["dev"], bag.splits["test"]
    if HIZLI_TEST:
        n_tr, n_dv, n_ts = HIZLI_TEST_CUMLE
        tr, dv, ts = alt_kume(tr, n_tr), alt_kume(dv, n_dv), alt_kume(ts, n_ts)
    return tr, dv, ts


def _model_kosusu(m_cfg: Dict, ad: str, bag: BolunmeBaglami, seed: int, slug: str,
                  model_turu: str = "multihead") -> Optional[Dict[str, Any]]:
    name, short = m_cfg["name"], m_cfg["short"]
    on_ek = "" if model_turu == "multihead" else "birlesik_"
    kdir = os.path.dirname(cikti_yolu("kosular", slug, "x"))
    ozet_p = os.path.join(kdir, f"{on_ek}{short}_ozet.json")
    kayit_p = os.path.join(kdir, f"{on_ek}em_records_{short}.csv.gz")
    imza = kosu_imzasi(m_cfg, bag, model_turu)
    if ATLA_TAMAMLANANLAR and os.path.exists(ozet_p) and os.path.exists(kayit_p):
        try:
            with open(ozet_p, encoding="utf-8") as f:
                oz = json.load(f)
        except (OSError, ValueError):
            oz = {}
        eski_imza = oz.get("kosu_imzasi")
        if _imza_esit(eski_imza, imza):
            recs = _records_from_csv(kayit_p)
            em = em_ozetle(recs)
            em.update({k: oz["em"].get(k) for k in ("seen_em_ci_low", "seen_em_ci_high", "unseen_em_ci_low", "unseen_em_ci_high")})
            print(f"  [Sürdürme] {name} [{slug}] tamamlanmış (ayar imzası aynı) — kayıtlardan yüklendi.")
            return {"summary": oz["summary"], "em": em, "breakdown": oz["breakdown"], "head_f1": oz["head_f1"],
                    "log_history": oz.get("log_history", []), "bilgi": oz.get("bilgi", {})}
        farkli = sorted(k for k in set(imza) | set(eski_imza or {})
                        if json.dumps((eski_imza or {}).get(k)) != json.dumps(imza.get(k)))
        print(f"  ⚠️  [Sürdürme] {name} [{slug}]: önceki koşumun ayarları farklı ({farkli or 'imza yok'}) — "
              f"YENİDEN eğitiliyor; eski dosyalar '.onceki_ayar' uzantısıyla kenara alındı.")
        for p_ in (ozet_p, kayit_p):   # yeniden eğitim başarısız olsa bile eski kayıtlar analize KARIŞMAZ
            with contextlib.suppress(OSError):
                os.replace(p_, p_ + ".onceki_ayar")

    print(f"\n{'=' * 72}\n  MODEL: {name}{' (birleşik-etiket, N4)' if on_ek else ''}  [{ad} | split={bag.split_seed} | "
          f"seed={seed}]\n{'=' * 72}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(m_cfg["hf_id"], revision=_rev(m_cfg["hf_id"]))
    except Exception as e:
        logger.warning(f"  [{name}] Tokenizer yüklenemedi: {e}")
        print(f"  [{name}] ⚠️  Tokenizer yüklenemedi — atlanıyor.")
        return None
    tr, dv, ts = _veri_alt_kumeleri(bag)
    bundle_vocab = build_bundle_vocab(bag.splits["train"]) if model_turu == "bundle" else None
    if model_turu == "bundle":
        train_ds = BundleTagDataset(tr, tokenizer, bag.label_mappings, bundle_vocab)
        dev_ds = BundleTagDataset(dv, tokenizer, bag.label_mappings, bundle_vocab)
    else:
        train_ds = MultiHeadMorphDataset(tr, tokenizer, bag.label_mappings)
        dev_ds = MultiHeadMorphDataset(dv, tokenizer, bag.label_mappings)
    train_ds.log_truncation_stats()
    ckpt = _checkpoint_dizini(f"{on_ek}{short}", slug)
    model, best_eval, log_hist, bilgi = train_one_model(m_cfg, train_ds, dev_ds, bag.label_mappings, ckpt, seed,
                                                        model_turu=model_turu, bundle_vocab=bundle_vocab)
    if model is None:
        return None
    bilgi["slug"] = slug
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  [{name}] Test tahminleri üretiliyor...")
    token_results = batch_predict(model, tokenizer, ts, bag.label_mappings, device=device, bundle_vocab=bundle_vocab)
    tm = compute_test_metrics(token_results, bag.label_mappings)
    recs = token_kayitlari(token_results, ts, bag.held_out, bag.train_bundles, bag.train_forms)
    em = em_ozetle(recs)
    em["seen_em_ci_low"], em["seen_em_ci_high"] = bootstrap_bundle_ci(recs, is_seen=True)
    em["unseen_em_ci_low"], em["unseen_em_ci_high"] = bootstrap_bundle_ci(recs, is_seen=False)
    breakdown = compute_per_head_accuracy_breakdown(token_results, ts, bag.held_out)
    head_f1 = {h: tm[f"test_{h}_f1"] for h in MORPH_HEADS}
    summary = {
        "test_avg_f1": tm["test_avg_f1"], "test_avg_acc": tm["test_avg_acc"],
        "test_nonone_avg_f1": tm["test_nonone_avg_f1"],
        "test_nonone_tum_avg_f1": tm["test_nonone_tum_avg_f1"],
        "test_nonone_goldmask_avg_f1": tm["test_nonone_goldmask_avg_f1"],
        "test_nonone_legacy_avg_f1": tm["test_nonone_legacy_avg_f1"],
        "mh_overall": em["overall_em"], "mh_seen": em["seen_em"], "mh_unseen": em["unseen_em"],
        "mh_seen_train": em["seen_train_em"], "mh_seen_novel": em["seen_novel_em"],
        "mh_seen_ci_low": em["seen_em_ci_low"], "mh_seen_ci_high": em["seen_em_ci_high"],
        "mh_unseen_ci_low": em["unseen_em_ci_low"], "mh_unseen_ci_high": em["unseen_em_ci_high"],
        "overall_f1_bundle": em["overall_f1"],
        "dev_selection_metric": bilgi.get("secim_olcutu"),
        "dev_avg_f1": best_eval.get("eval_avg_f1"), "dev_avg_f1_nonone": best_eval.get("eval_avg_f1_nonone"),
        "dev_min_f1_nonone": best_eval.get("eval_min_f1_nonone"), "dev_exact_match": best_eval.get("eval_exact_match"),
        "best_epoch": bilgi.get("en_iyi_epoch"), "best_is_last_epoch": bilgi.get("en_iyi_son_epoch_mu"),
        "train_time_sec": bilgi.get("sure_sn"),
    }
    print(f"  [{name}] Test F1(N+): {summary['test_avg_f1']:.2%}  F1(N−, düzeltilmiş): {summary['test_nonone_avg_f1']:.2%}  "
          f"[eski hesap: {summary['test_nonone_legacy_avg_f1']:.2%}]  Acc: {summary['test_avg_acc']:.2%}")
    ci = (f"[{em['unseen_em_ci_low']:.1%}, {em['unseen_em_ci_high']:.1%}]"
          if not math.isnan(em["unseen_em_ci_low"]) else "n/a")
    print(f"  [{name}] Seen EM: {em['seen_em']:.2%} (eğitimde görülen demet {em['seen_train_em']:.2%}; "
          f"held-out değil ama eğitimde yok {em['seen_novel_em']:.2%}, n={em['seen_novel_token_count']})  "
          f"Unseen EM: {em['unseen_em']:.2%}  (demet-bazlı %95 GA {ci})")
    print_per_head_breakdown(breakdown, name, ad)
    if model_turu == "multihead":
        plot_dative_locative_confusion(token_results, name, ad, save_path=os.path.join(kdir, f"case_karisiklik_{short}.png"))
        det = collect_detailed_predictions(token_results, ts, bag.held_out)
        bundle_level_report(det, name, ad, save_path=os.path.join(kdir, f"demet_duzeyi_{short}.csv"))
        error_analysis(det, name, ad, bag.held_out, save_path=os.path.join(kdir, f"hatalar_{short}.csv"))
    # --- kalıcı kayıt (sürdürme + reanaliz için); özet JSON en SONDA yazılır = tamamlanma işareti ---
    if os.path.exists(ozet_p):
        os.remove(ozet_p)
    df_rec = _kayit_meta(pd.DataFrame(recs), ad, bag, seed, m_cfg)
    gecici = f"{kayit_p}.tmp{os.getpid()}"
    df_rec.to_csv(gecici, index=False, compression="gzip")
    os.replace(gecici, kayit_p)
    # --- HF yükleme ve checkpoint temizliği ---
    yuklendi = False
    yukle = HF_MODEL_YUKLE and (bag.split_seed == SPLIT_SEED or not HF_MODEL_YUKLE_SADECE_ANA_BOLUNME)
    if yukle:
        repo_id = f"{HF_USERNAME}/" + f"{on_ek}{short}-{_veri_etiketi(ad, bag)}-split{bag.split_seed}-seed{seed}-v5".replace("_", "-")
        yuklendi = hf_model_yukle(model, tokenizer, bag.label_mappings if model_turu == "multihead"
                                  else {"upos": bag.label_mappings["upos"], "bundle": bundle_vocab}, bilgi, repo_id)
    if CHECKPOINT_SIL and (yuklendi or not yukle):
        shutil.rmtree(ckpt, ignore_errors=True)
    elif os.path.exists(ckpt):
        logger.warning(f"  [{name}] Checkpoint diskte tutuluyor: {ckpt}")
    bilgi["model_hub_yuklendi"] = yuklendi
    json_yaz({"kosu_imzasi": imza, "summary": summary, "em": {k: v for k, v in em.items() if k != "records"},
              "breakdown": breakdown, "head_f1": head_f1, "test_metrics": tm, "best_eval": best_eval,
              "log_history": log_hist, "bilgi": bilgi}, ozet_p)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"summary": summary, "em": em, "breakdown": breakdown, "head_f1": head_f1,
            "log_history": log_hist, "bilgi": bilgi}


def print_summary_table(all_results, em_results=None, dataset_name=""):
    SEP = "─" * 104
    print(f"\n{SEP}  [{dataset_name}]")
    print("  ℹ  F1(N+): NONE dahil  |  F1(N−): NONE etiket listesinden çıkarılarak (DÜZELTİLMİŞ)  |  "
          "F1(N−)eski: önceki hatalı hesap")
    print(f"  {'Model':<10} {'F1(N+)':>8} {'F1(N−)':>8} {'F1(N−)eski':>11} {'Acc':>8} {'OvrlEM':>8} "
          f"{'SeenEM':>8} {'UnseenEM':>9} {'EnİyiEp':>8}")
    for m in MODELS:
        s = all_results.get(m["short"], {})
        if not s:
            continue
        print(f"  {m['name']:<10} {s['test_avg_f1']:>8.2%} {s['test_nonone_avg_f1']:>8.2%} "
              f"{s['test_nonone_legacy_avg_f1']:>11.2%} {s['test_avg_acc']:>8.2%} {s['mh_overall']:>8.2%} "
              f"{s['mh_seen']:>8.2%} {s['mh_unseen']:>9.2%} {str(s.get('best_epoch')):>8}")
    print(SEP)


def run_single_dataset_experiment(ad: str, bag: BolunmeBaglami, train_seed: int,
                                  het_df: Optional[pd.DataFrame] = None) -> Tuple[Dict, Dict, Dict, Dict, List[Dict]]:
    slug = _slug(ad, bag, train_seed)
    print(f"\n{'#' * 72}\n  {ad} | split_seed={bag.split_seed} | train_seed={train_seed}\n{'#' * 72}")
    tr, dv, ts = _veri_alt_kumeleri(bag)
    print(f"  {len(tr)} train | {len(dv)} dev | {len(ts)} test cümle"
          f"{'  (HIZLI_TEST alt kümesi)' if HIZLI_TEST else ''}  |  cihaz: "
          f"{'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  Etiket sayıları (NONE dahil): { {h: len(v) for h, v in bag.label_mappings.items()} }")
    audit_label_coverage(dv, bag.label_mappings, "dev")
    audit_label_coverage(ts, bag.label_mappings, "test")

    all_results, em_results, log_histories, head_f1s, breakdowns = {}, {}, {}, {}, {}
    reporter = SeenUnseenReporter()
    for m_cfg in MODELS:
        r = _model_kosusu(m_cfg, ad, bag, train_seed, slug)
        if r is None:
            logger.warning(f"  [{m_cfg['name']}] {slug} atlandı.")
            continue
        s = m_cfg["short"]
        all_results[s], em_results[s] = r["summary"], r["em"]
        log_histories[s], head_f1s[s], breakdowns[s] = r["log_history"], r["head_f1"], r["breakdown"]
        reporter.add(m_cfg["name"], ad, r["em"], bag.split_seed, train_seed)

    kdir = os.path.dirname(cikti_yolu("kosular", slug, "x"))
    reporter.report()
    reporter.save_csv(os.path.join(kdir, "seen_unseen_em.csv"))
    if breakdowns:   # Çizelge 4.4–4.5 karşılığı (başlık başına Seen/Unseen doğruluk)
        pd.DataFrame([{"model": NAME_MAP().get(s, s), "baslik": h, **v} for s, bd in breakdowns.items()
                      for h, v in bd.items()]).to_csv(os.path.join(kdir, "baslik_dogrulugu.csv"), index=False,
                                                      float_format="%.4f")
    print_summary_table(all_results, em_results, dataset_name=f"{ad} split={bag.split_seed} seed={train_seed}")
    plot_per_head_breakdown(breakdowns, f"{ad} (split {bag.split_seed}, tohum {train_seed})",
                            save_path=os.path.join(kdir, "baslik_dogrulugu.png"))
    if het_df is not None and len(em_results):
        compute_jaccard_em_spearman(het_df, em_results, ad, save_path=os.path.join(kdir, "spearman_jaccard_em.png"))
        compute_jaccard_em_spearman(het_df, em_results, f"{ad} (≥{SPEARMAN_MIN_TOKENS} belirteç)",
                                    save_path=os.path.join(kdir, f"spearman_jaccard_em_min{SPEARMAN_MIN_TOKENS}.png"),
                                    min_tokens=SPEARMAN_MIN_TOKENS)
    if all_results:
        plot_model_comparison_bar(all_results, f" — {ad}", save_path=os.path.join(kdir, "model_karsilastirma.png"))
        plot_head_f1_multimodel(head_f1s, f" — {ad}", save_path=os.path.join(kdir, "baslik_f1.png"))
        plot_training_curves_multimodel(log_histories, ad, save_path=os.path.join(kdir, "egitim_egrileri.png"))
        plot_seen_unseen_em_bar(em_results, ad, save_path=os.path.join(kdir, "em_cubuk.png"))

    mc_rows: List[Dict] = []
    shorts_ok = [m["short"] for m in MODELS if m["short"] in em_results]
    if len(shorts_ok) >= 2:
        common = build_common_token_keys(em_results)
        for subset in ("overall", "seen", "unseen"):
            mc_rows += mcnemar_pairwise(em_results, shorts_ok, subset=subset, dataset_name=ad, seed=train_seed,
                                        split_seed=bag.split_seed, common_keys=common)
        if mc_rows:
            pd.DataFrame(mc_rows).to_csv(os.path.join(kdir, "mcnemar.csv"), index=False, float_format="%.6f")

    if BIRLESIK_ETIKET_MODELI_KOS:
        rows = []
        for m_cfg in MODELS:
            rb = _model_kosusu(m_cfg, ad, bag, train_seed, slug, model_turu="bundle")
            if rb is None:
                continue
            mh = all_results.get(m_cfg["short"], {})
            rows.append({"model": m_cfg["name"], "birlesik_seen_em": rb["summary"]["mh_seen"],
                         "birlesik_unseen_em": rb["summary"]["mh_unseen"], "cok_basli_seen_em": mh.get("mh_seen"),
                         "cok_basli_unseen_em": mh.get("mh_unseen")})
        if rows:
            d = pd.DataFrame(rows)
            d.to_csv(os.path.join(kdir, "birlesik_etiket_karsilastirma.csv"), index=False, float_format="%.4f")
            print("\n  [N4] Birleşik-etiket vs çok-başlı (Seen EM / Unseen EM):")
            print(d.to_string(index=False, float_format=lambda x: f"{x:.2%}"))
    logger.info(f"Blok tamamlandı: {slug}")
    return all_results, em_results, log_histories, head_f1s, mc_rows


# =============================================================================
# 21) ÇOK-TOHUM, BÖLÜNME VARYANSI VE ÇAPRAZ VERİ SETİ ÖZETLERİ
# =============================================================================

OZET_METRIKLERI = [("test_avg_f1", "F1(N+)"), ("test_nonone_avg_f1", "F1(N−)"), ("test_avg_acc", "Acc"),
                   ("mh_overall", "OvrlEM"), ("mh_seen", "SeenEM"), ("mh_unseen", "UnseenEM"),
                   ("mh_seen_novel", "Seen-yeniEM"), ("test_nonone_legacy_avg_f1", "F1(N−)eski")]


def cok_tohum_ozeti(ad: str, split_seed: int, seed_recs: List[Dict]) -> pd.DataFrame:
    """Ortalama ± örneklem std (ddof = 1) + std'nin %95 GA'sı (n = 3'te çok geniştir; 'kararlılık'
    etiketleri istatistiksel değil BETİMSELDİR) + tohumların bootstrap GA ortalaması."""
    SEP = "═" * 126
    n = len(seed_recs)
    print(f"\n{SEP}\n  ÇOK-TOHUM ÖZETİ — {ad} | split_seed={split_seed} | n={n} tohum  "
          f"(varyans kaynağı: başlangıç ağırlıkları + batch sırası; bölünme SABİT)\n{SEP}")
    print(f"  {'Model':<10} " + " ".join(f"{lbl:>17}" for _, lbl in OZET_METRIKLERI[:6]) + f" {'EnİyiEp':>10}")
    rows = []
    for m in MODELS:
        s = m["short"]
        recs = [r[s] for r in seed_recs if isinstance(r.get(s), dict)]
        if not recs:
            continue
        row = {"dataset": ad, "split_seed": split_seed, "model": m["name"], "n_seeds": len(recs)}
        parts = []
        for key, lbl in OZET_METRIKLERI:
            vals = [x[key] for x in recs if x.get(key) is not None and not (isinstance(x[key], float) and math.isnan(x[key]))]
            mu = float(np.mean(vals)) if vals else float("nan")
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            lo, hi = std_guven_araligi(sd, len(vals))
            row.update({f"{key}_mean": mu, f"{key}_std": sd, f"{key}_std_ga_alt": lo, f"{key}_std_ga_ust": hi})
            parts.append(f"{mu:>9.2%} ±{(sd if not math.isnan(sd) else 0):>6.2%}")
        ci_lo = [x.get("mh_unseen_ci_low") for x in recs if x.get("mh_unseen_ci_low") is not None]
        ci_hi = [x.get("mh_unseen_ci_high") for x in recs if x.get("mh_unseen_ci_high") is not None]
        row["unseen_boot_ga_alt_ort"] = float(np.nanmean(ci_lo)) if ci_lo else float("nan")
        row["unseen_boot_ga_ust_ort"] = float(np.nanmean(ci_hi)) if ci_hi else float("nan")
        row["en_iyi_epochlar"] = [x.get("best_epoch") for x in recs]
        rows.append(row)
        print(f"  {m['name']:<10} " + " ".join(f"{p:>17}" for p in parts[:6]) + f" {str(row['en_iyi_epochlar']):>10}")
    print(SEP)
    print("  NOT: ± = örneklem std (ddof = 1). n = 3 ile std'nin %95 GA'sı çok geniştir (CSV'de *_std_ga_* sütunları).")
    df = pd.DataFrame(rows)
    df.to_csv(cikti_yolu("ozet", f"cok_tohum_ozeti_{ad.lower()}_split{split_seed}.csv"), index=False, float_format="%.5f")
    # Tohum düzeyinde eşleştirilmiş t-testleri (Unseen ve Seen EM)
    tlist = []
    for key in ("mh_unseen", "mh_seen"):
        deger = {m["short"]: {r["seed"]: r[m["short"]][key] for r in seed_recs if isinstance(r.get(m["short"]), dict)}
                 for m in MODELS}
        t = tohum_duzeyi_testler({k: v for k, v in deger.items() if v}, etiket=key)
        if len(t):
            t.insert(0, "split_seed", split_seed)
            t.insert(0, "dataset", ad)
            tlist.append(t)
    if tlist:
        tt = pd.concat(tlist, ignore_index=True)
        tt.to_csv(cikti_yolu("ozet", f"tohum_duzeyi_t_{ad.lower()}_split{split_seed}.csv"), index=False, float_format="%.5f")
        print("  Tohum düzeyinde eşleştirilmiş t (Unseen EM, puan): " + "; ".join(
            f"{r['model_a']}−{r['model_b']} {100 * r['ort_fark']:+.2f} (p={r.get('p', float('nan')):.3f})"
            for _, r in tt[tt["karsilastirma"] == "mh_unseen"].iterrows() if r["n"] >= 2))
    return df


def tez_cizelgesi(ad: str, split_seed: int, seed_recs: List[Dict]) -> pd.DataFrame:
    """Çizelge 4.1/4.2 biçiminde: tohum satırları + ortalama ± std satırı."""
    rows = []
    for m in MODELS:
        for r in seed_recs:
            s = r.get(m["short"])
            if isinstance(s, dict):
                rows.append({"model": m["name"], "tohum": r["seed"], **{lbl: s.get(k) for k, lbl in OZET_METRIKLERI},
                             "en_iyi_epoch": s.get("best_epoch")})
    df = pd.DataFrame(rows)
    df.to_csv(cikti_yolu("ozet", f"tez_cizelgesi_{ad.lower()}_split{split_seed}.csv"), index=False, float_format="%.5f")
    return df


def bolunme_varyansi_ozeti(ad: str, per_split: Dict[int, List[Dict]]) -> pd.DataFrame:
    """Bölünme tohumları arasında Unseen EM (her bölünmede tohum ortalaması)."""
    rows = []
    for sp, recs in sorted(per_split.items()):
        for m in MODELS:
            vals = [r[m["short"]]["mh_unseen"] for r in recs if isinstance(r.get(m["short"]), dict)]
            sv = [r[m["short"]]["mh_seen"] for r in recs if isinstance(r.get(m["short"]), dict)]
            if vals:
                rows.append({"dataset": ad, "split_seed": sp, "model": m["name"], "n_seeds": len(vals),
                             "unseen_em_mean": float(np.mean(vals)), "seen_em_mean": float(np.mean(sv))})
    df = pd.DataFrame(rows)
    if df.empty or df["split_seed"].nunique() < 2:
        return df
    df.to_csv(cikti_yolu("ozet", f"bolunme_varyansi_{ad.lower()}.csv"), index=False, float_format="%.5f")
    SEP = "═" * 100
    print(f"\n{SEP}\n  BÖLÜNME VARYANSI — {ad}  ({df['split_seed'].nunique()} bölünme tohumu; her nokta "
          f"{int(df['n_seeds'].max())} eğitim tohumunun ortalaması)\n{SEP}")
    for m in MODELS:
        d = df[df["model"] == m["name"]]
        if len(d):
            print(f"  {m['name']:<10} Unseen EM ort. {d['unseen_em_mean'].mean():.2%}  std {d['unseen_em_mean'].std(ddof=1):.2%}  "
                  f"aralık {d['unseen_em_mean'].min():.2%}–{d['unseen_em_mean'].max():.2%}")
    piv = df.pivot(index="split_seed", columns="model", values="unseen_em_mean")
    names = [m["name"] for m in MODELS if m["name"] in piv.columns]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = piv[names[i]] - piv[names[j]]
            print(f"  {names[i]} − {names[j]}: bölünme başına fark (puan) "
                  + ", ".join(f"s{k}:{100 * v:+.1f}" for k, v in d.items())
                  + f"  → işaret {'TUTARLI' if (d > 0).all() or (d < 0).all() else 'DEĞİŞKEN'}")
    print(SEP)
    plot_split_variance(df, ad, save_path=cikti_yolu("ozet", f"bolunme_varyansi_{ad.lower()}.png"))
    return df


def _cross_dataset_comparison(results: Dict[Tuple[str, int], List[Dict]]):
    SEP = "─" * 100
    print(f"\n{SEP}\n  ÇAPRAZ VERİ SETİ KARŞILAŞTIRMASI (ana bölünme, split_seed={SPLIT_SEED}; tohum ortalaması)\n{SEP}")
    print(f"  {'Model':<10} {'Dataset':<9} {'F1(N+)':>8} {'F1(N−)':>8} {'Acc':>8} {'SeenEM':>8} {'UnseenEM':>9}")
    for (ad, sp), recs in results.items():
        if sp != SPLIT_SEED:
            continue
        for m in MODELS:
            v = [r[m["short"]] for r in recs if isinstance(r.get(m["short"]), dict)]
            if not v:
                continue
            mu = {k: float(np.mean([x[k] for x in v])) for k in ("test_avg_f1", "test_nonone_avg_f1", "test_avg_acc", "mh_seen", "mh_unseen")}
            print(f"  {m['name']:<10} {ad:<9} {mu['test_avg_f1']:>8.2%} {mu['test_nonone_avg_f1']:>8.2%} "
                  f"{mu['test_avg_acc']:>8.2%} {mu['mh_seen']:>8.2%} {mu['mh_unseen']:>9.2%}")
        print()
    print(SEP)


def determinizm_sinamasi(bag: BolunmeBaglami) -> Dict[str, Any]:
    """Aynı tohumla iki kısa eğitim (8 adım) → kayıp ve tüm parametrelerin SHA-256 özeti karşılaştırılır."""
    m = MODELS[0]
    tok = AutoTokenizer.from_pretrained(m["hf_id"], revision=_rev(m["hf_id"]))
    ds_ = MultiHeadMorphDataset(bag.splits["train"][:64], tok, bag.label_mappings)
    sonuc = []
    for k in range(2):
        gecici_dizin = os.path.join(CHECKPOINT_KOK, f"_determinizm_{k}")
        model, _, hist, _ = train_one_model(m, ds_, None, bag.label_mappings, gecici_dizin, DEFAULT_SEED, max_steps=8)
        kayip = [e.get("train_loss") for e in hist if "train_loss" in e]
        sonuc.append({"train_loss": kayip[-1] if kayip else None, "parmak_izi": encoder_parmak_izi(model)})
        del model
        gc.collect()
        shutil.rmtree(gecici_dizin, ignore_errors=True)
    ayni = sonuc[0] == sonuc[1]
    rapor = {"model": m["name"], "kosular": sonuc, "ayni_sonuc": ayni, "ortam": ortam_bilgisi()}
    json_yaz(rapor, cikti_yolu("manifest", "determinizm_sinamasi.json"))
    print(f"\n  [Determinizm sınaması] {m['name']}, 8 adım × 2: "
          f"{'AYNI sonuç (kayıp ve parametre özeti eşit)' if ayni else 'FARKLI sonuç — tezde yalnız mümkün olduğunca deterministik denmeli'}")
    return rapor


# =============================================================================
# 22) ANA AKIŞLAR (MODLAR)
# =============================================================================

def _hizli_test_ayarlari() -> Optional[str]:
    """HIZLI_TEST: küçük alt küme; çıktılar AYRI klasöre; (varsayılan) yükleme/yedek kapalı."""
    global NUM_TRAIN_EPOCHS, COMPLETION_MIN_EPOCHS, SEEDS, EK_SPLIT_SAYISI, EK_SPLIT_SEEDS
    global CIKTI_DIZINI, CHECKPOINT_KOK, HF_MODEL_YUKLE, HF_SONUC_YUKLE, HF_ARTIMLI_YEDEK, DRIVE_YEDEK_DIZINI
    if not HIZLI_TEST:
        return None
    NUM_TRAIN_EPOCHS, COMPLETION_MIN_EPOCHS = HIZLI_TEST_EPOCH, 1
    SEEDS = SEEDS[:2]
    EK_SPLIT_SAYISI = {k: min(int(v), 1) for k, v in EK_SPLIT_SAYISI.items()}
    EK_SPLIT_SEEDS = {k: list(v)[:1] for k, v in EK_SPLIT_SEEDS.items()}
    if not CIKTI_DIZINI.rstrip("/").endswith("_HIZLI_TEST"):
        CIKTI_DIZINI = CIKTI_DIZINI.rstrip("/") + "_HIZLI_TEST"
        CHECKPOINT_KOK = CHECKPOINT_KOK.rstrip("/") + "_HIZLI_TEST"
    if HIZLI_TEST_YUKLEME_KAPALI:
        HF_MODEL_YUKLE = HF_SONUC_YUKLE = HF_ARTIMLI_YEDEK = False
        DRIVE_YEDEK_DIZINI = None
    return (f"  ⚠️  HIZLI_TEST açık: alt küme {HIZLI_TEST_CUMLE}, {HIZLI_TEST_EPOCH} epoch, {len(SEEDS)} tohum; "
            f"çıktılar '{CIKTI_DIZINI}' klasörüne. Sonuçlar tez için KULLANILAMAZ.")


def manifest_yaz(datasets: Dict[str, Dict[str, Any]]) -> None:
    m = {"ortam": ortam_bilgisi(), "yapilandirma": yapilandirma_ozeti(),
         "ud_depolari": {ad: d["depo"] for ad, d in datasets.items()},
         "veri_istatistikleri": {ad: d["istatistik"] for ad, d in datasets.items()},
         "modeller": MODELS}
    json_yaz(m, cikti_yolu("manifest", "calisma_manifesti.json"))
    pip_freeze_yaz(cikti_yolu("manifest", "pip_freeze.txt"))
    o = m["ortam"]
    print(f"  Ortam: Python {o['python']} | torch {o['torch']} | transformers {o['transformers']} | "
          f"CUDA {o['torch_cuda_version']} | GPU: {o.get('gpu_adi', 'yok')}")


def ek_bolunme_tohumlarini_sec(ad: str, ham: Dict[str, Any]) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Ek bölünme tohumlarını YALNIZ veri ölçütüyle seçer (eğitim korunumu ≥ eşik ve atom
    korunumu). Reddedilen tohumlar gerekçesiyle raporlanır."""
    anahtar = ad.split("_")[0]
    elle = EK_SPLIT_SEEDS.get(anahtar) or EK_SPLIT_SEEDS.get(ad)
    hedef = len(elle) if elle else int(EK_SPLIT_SAYISI.get(anahtar, 0))
    adaylar = list(elle) if elle else list(EK_SPLIT_ADAY_TOHUMLARI)
    secilen, kayit = [], []
    for sp in adaylar:
        if len(secilen) >= hedef:
            break
        if sp == SPLIT_SEED:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                splits, _ = build_controlled_split(ham["train"], ham["dev"], ham["test"],
                                                   min_bundle_freq=DATASET_MIN_BUNDLE_FREQ.get(ad, 20),
                                                   split_seed=sp, sessiz=True)
                korunum = len(splits["train"]) / max(1, len(ham["train"]))
                neden = "" if korunum >= EK_SPLIT_MIN_EGITIM_KORUNUM else \
                    f"eğitim korunumu %{100 * korunum:.1f} < %{100 * EK_SPLIT_MIN_EGITIM_KORUNUM:.0f}"
            except AtomKorunumuHatasi as e:
                korunum, neden = float("nan"), f"atom korunumu ihlali ({e})"
        kayit.append({"veri_seti": ad, "split_seed": sp, "egitim_korunumu": korunum,
                      "kabul": not neden, "neden": neden})
        if not neden:
            secilen.append(sp)
    if hedef:
        red = [k for k in kayit if not k["kabul"]]
        print(f"  [N2] {ad}: ek bölünme tohumları {secilen} (hedef {hedef}); reddedilen: "
              + (", ".join(f"{k['split_seed']} ({k['neden']})" for k in red) if red else "yok"))
        pd.DataFrame(kayit).to_csv(cikti_yolu("veri", f"ek_bolunme_secimi_{ad.lower()}.csv"), index=False,
                                   float_format="%.4f")
    return secilen, kayit


def _baglamlari_kur(datasets: Dict[str, Dict[str, Any]], veri_analizi: bool = True) -> Tuple[Dict, Dict]:
    baglamlar, het = {}, {}
    for ad, ham in datasets.items():
        ek, _ = ek_bolunme_tohumlarini_sec(ad, ham)
        for sp in [SPLIT_SEED] + ek:
            print(f"\n{'═' * 72}\n  KONTROLLÜ BÖLÜNME — {ad} | split_seed={sp}\n{'═' * 72}")
            bag = bolunme_baglami_kur(ad, ham, sp)
            baglamlar[(ad, sp)] = bag
            if veri_analizi:
                veri_duzeyi_analizler(ad, ham, bag.splits, bag.splitter, sp)
            het[(ad, sp)] = heterogeneity_report(bag.splits["test"], bag.splits["train"], bag.held_out, ad, sp)
    return baglamlar, het


def run_thesis_experiments() -> Dict:
    specs = veri_seti_tanimlari()
    print(f"\n{'#' * 72}\n  VERİ YÜKLEME (UD {UD_SURUM_ETIKETI}; BOUN modu: {BOUN_ANOTASYON_MODU})\n{'#' * 72}")
    datasets = {ad: veri_seti_yukle(ad, sp) for ad, sp in specs.items()}
    manifest_yaz(datasets)
    hf_oturum_kontrol()
    if ATLA_TAMAMLANANLAR:
        sonuclari_geri_yukle()
    for ad, ham in datasets.items():
        anotasyon_tutarlilik_denetimi(ad, ham)
    seeds = SEEDS if MULTI_SEED_MODE else [DEFAULT_SEED]
    print(f"\n{'#' * 72}")
    print(f"  Eğitim tohumları           : {seeds}")
    print(f"  Ana bölünme tohumu         : {SPLIT_SEED};  ek bölünme sayısı: {EK_SPLIT_SAYISI} "
          f"(kabul ölçütü: eğitim korunumu ≥ %{100 * EK_SPLIT_MIN_EGITIM_KORUNUM:.0f})")
    print(f"  Seçim/erken durdurma ölçütü: {EARLY_STOPPING_METRIC} (DÜZELTİLMİŞ; F1(N−) biçimi: {F1_NONONE_YONTEMI})")
    print(f"  Optimizer / LR takvimi     : {OPTIMIZER} / {LR_SCHEDULER} (ısınma oranı {WARMUP_RATIO})")
    print(f"  Bootstrap (n, tohum)       : {BOOTSTRAP_N_RESAMPLES}, {BOOTSTRAP_SEED}")
    print(f"  Modeller                   : {[m['name'] for m in MODELS]}")
    print(f"  HF depoları                : {'GİZLİ' if HF_DEPO_GIZLI else 'HERKESE AÇIK'}")
    print(f"{'#' * 72}")
    baglamlar, het = _baglamlari_kur(datasets)
    if DETERMINIZM_TESTI and baglamlar:
        determinizm_sinamasi(next(iter(baglamlar.values())))

    sirali = sorted(baglamlar, key=lambda k: (k[1] != SPLIT_SEED, list(datasets).index(k[0]), k[1]))
    results: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    mcnemar_all: List[Dict] = []
    for key in sirali:
        ad, sp = key
        for seed in seeds:
            res, _, _, _, mc = run_single_dataset_experiment(ad, baglamlar[key], seed, het.get(key))
            results[key].append({"seed": seed, **res})
            mcnemar_all += mc
            artimli_yedek(f"{ad} split{sp} seed{seed}")

    print(f"\n{'#' * 72}\n  ÖZETLER\n{'#' * 72}")
    for (ad, sp), recs in results.items():
        cok_tohum_ozeti(ad, sp, recs)
        tez_cizelgesi(ad, sp, recs)
        plot_multi_seed_results(recs, f"{ad} (split {sp})",
                                save_path=cikti_yolu("ozet", f"cok_tohum_{ad.lower()}_split{sp}.png"))
    for ad in datasets:
        per_split = {sp: recs for (a, sp), recs in results.items() if a == ad}
        bolunme_varyansi_ozeti(ad, per_split)
    if len(datasets) > 1:
        _cross_dataset_comparison(results)
    if mcnemar_all:
        mc = pd.DataFrame(mcnemar_all)
        mc["p_holm_tum_aile"] = holm_duzeltmesi(mc["p_value"].tolist())
        mc["anlamli_holm"] = mc["p_holm_tum_aile"] < ALPHA
        mc.to_csv(cikti_yolu("ozet", "mcnemar_tum_testler.csv"), index=False, float_format="%.6f")
        print(f"\n  McNemar: {len(mc)} test. Aile tanımı: her (veri seti × bölünme × tohum × alt küme) için 3 çift, "
              f"Bonferroni α_adj = {ALPHA / 3:.4f}. Tüm {len(mc)} test için Holm düzeltmesiyle anlamlı kalan: "
              f"{int(mc['anlamli_holm'].sum())}. Belirteç bağımlılığı nedeniyle KEŞİFSEL (bkz. E3).")
    # E0–E10: yeni kayıtlar üzerinde (veri seti × bölünme başına, bellek dostu)
    for (ad, sp), bag in baglamlar.items():
        # yalnız bu koşumun tohum × modelleri VE imzası şimdiki ayarlarla aynı olan kayıtlar
        dosyalar = [tamamlanmis_kayit_yolu(_slug(ad, bag, sd), m, bag) for sd in seeds for m in MODELS]
        dosyalar = [f for f in dosyalar if f]
        if not dosyalar:
            continue
        df = pd.concat([kayit_oku(f) for f in dosyalar], ignore_index=True)
        reanaliz_calistir(df, {(ad, sp): bag}, alt_dizin=os.path.join("analizler", f"{_veri_etiketi(ad, bag)}_split{sp}"),
                          e9=(sp == SPLIT_SEED))
        del df
        gc.collect()
    ser = {f"{ad}|split{sp}": [{k: v for k, v in r.items()} for r in recs] for (ad, sp), recs in results.items()}
    json_yaz(ser, cikti_yolu("all_results_v5.json"))
    # Ana bölünme, eski dosyanın yapısında (veri seti → tohum listesi). Ad bilerek FARKLI: v5'te
    # test_nonone_avg_f1 DÜZELTİLMİŞ değerdir; eski hesap test_nonone_legacy_avg_f1 adıyla durur.
    eski_bicim = {ad: recs for (ad, sp), recs in results.items() if sp == SPLIT_SEED}
    json_yaz(eski_bicim, cikti_yolu("all_results_ana_bolunme_v5.json"))
    print(f"\n  Sonuçlar '{cikti_yolu('all_results_v5.json')}' ve (ana bölünme) "
          f"'{cikti_yolu('all_results_ana_bolunme_v5.json')}' dosyalarına kaydedildi.")
    yuklenemeyen = []
    for f in glob.glob(os.path.join(CIKTI_DIZINI, "kosular", "*", "*_ozet.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                oz = json.load(fh)
        except (OSError, ValueError):
            continue
        b = oz.get("bilgi", {})
        ana = oz.get("kosu_imzasi", {}).get("split_seed") == SPLIT_SEED
        if b.get("model_hub_yuklendi") is False and HF_MODEL_YUKLE and (ana or not HF_MODEL_YUKLE_SADECE_ANA_BOLUNME):
            yuklenemeyen.append({"ozet": os.path.relpath(f, CIKTI_DIZINI), "checkpoint": b.get("en_iyi_checkpoint")})
    if yuklenemeyen:
        pd.DataFrame(yuklenemeyen).to_csv(cikti_yolu("ozet", "hub_yuklenemeyen_modeller.csv"), index=False)
        print(f"  ⚠️  Hub'a yüklenemeyen {len(yuklenemeyen)} model var (liste: ozet/hub_yuklenemeyen_modeller.csv); "
              f"yerel checkpoint'leri silinmedi.")
    drive_yedekle()
    hf_sonuclari_yukle("v5: tüm deneyler tamamlandı")
    if HF_DEPO_GIZLI:
        print("  HATIRLATMA [Y8]: Depolar GİZLİ. Tezde 'genel erişime açık' yazıyorsa ya depoları açın "
              "(hf_depo_gorunurlugu_ayarla) ya da tez metnini düzeltin.")
    logger.info("Tüm deneyler tamamlandı.")
    return {"results": results, "baglamlar": baglamlar}


def veri_modu() -> Dict:
    specs = veri_seti_tanimlari()
    datasets = {ad: veri_seti_yukle(ad, sp) for ad, sp in specs.items()}
    manifest_yaz(datasets)
    for ad, ham in datasets.items():
        anotasyon_tutarlilik_denetimi(ad, ham)
    baglamlar, _ = _baglamlari_kur(datasets)
    if E9_TOKENIZASYON_ANALIZI:
        e9 = [e9_tokenizasyon(b) for (ad, sp), b in baglamlar.items() if sp == SPLIT_SEED]
        e9 = [x for x in e9 if len(x)]
        if e9:
            d = pd.concat(e9, ignore_index=True)
            d.to_csv(cikti_yolu("veri", "E9_tokenizasyon.csv"), index=False, float_format="%.4f")
            print(d.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n  Veri analizleri tamamlandı → {os.path.join(CIKTI_DIZINI, 'veri')}/")
    return {"baglamlar": baglamlar}


def _eski_spec(ds: str, tag: str, anot: str) -> Dict[str, Any]:
    base = "IMST" if ds.upper().startswith("IMST") else "BOUN"
    return {"repo": f"UD_Turkish-{base}", "onek": f"tr_{base.lower()}", "tag": tag, "anotasyon": anot}


def _veri_aileleri(g: pd.DataFrame, ds: str) -> Set[Tuple[str, str]]:
    """Bir (veri seti, bölünme) grubundaki kayıtların veri aileleri (UD etiketi, anotasyon modu).
    Sürüm bilgisi olmayan eski biçim satırları REANALIZ_VERI'den alınır."""
    aileler: Set[Tuple[str, str]] = set()
    if "ud_tag" in g.columns and "anotasyon" in g.columns:
        bilinen = g[g["ud_tag"].notna() & g["anotasyon"].notna()]
        aileler |= set(zip(bilinen["ud_tag"].astype(str), bilinen["anotasyon"].astype(str)))
        eski_satir = int(len(g) - len(bilinen))
    else:
        eski_satir = len(g)
    if eski_satir and ds in REANALIZ_VERI:
        aileler.add((REANALIZ_VERI[ds][0], REANALIZ_VERI[ds][1]))
    return aileler


def _aile_denetimi(df: pd.DataFrame) -> None:
    for (ds, sp), g in df.groupby(["dataset", "split_seed"]):
        aileler = _veri_aileleri(g, ds)
        if len(aileler) > 1:
            dosya = sorted(g["_kaynak_dosya"].unique())[:6] if "_kaynak_dosya" in g.columns else []
            raise ValueError(f"'{ds}' split {sp} için FARKLI veri ailelerinin kayıtları karışmış: {sorted(aileler)} "
                             f"(dosyalar: {dosya}). Her aileyi ayrı bir klasörde analiz edin ya da eski kayıtlar "
                             f"için REANALIZ_VERI'yi düzeltin.")


def _reanaliz_baglamlari(df: pd.DataFrame) -> Dict[Tuple[str, int], BolunmeBaglami]:
    baglamlar, yuklu = {}, {}
    _aile_denetimi(df)
    for (ds, sp), g in df.groupby(["dataset", "split_seed"]):
        aileler = _veri_aileleri(g, ds)
        if not aileler:
            logger.warning(f"  {ds}: veri sürümü bilinmiyor (REANALIZ_VERI'ye ekleyin) — atlandı.")
            continue
        tag, anot = next(iter(aileler))
        k = (ds, tag, anot)
        if k not in yuklu:
            yuklu[k] = veri_seti_yukle(ds, _eski_spec(ds, tag, anot))
        baglamlar[(ds, int(sp))] = bolunme_baglami_kur(ds, yuklu[k], int(sp), sessiz=True)
    return baglamlar


def reanaliz_modu(kayit_dizini: Optional[str] = None, alt_klasorler: Optional[bool] = None) -> Dict:
    d = kayit_dizini or REANALIZ_KAYIT_DIZINI
    alt = REANALIZ_ALT_KLASORLER if alt_klasorler is None else alt_klasorler
    dosyalar = set(glob.glob(os.path.join(d, "em_records_*.csv.gz")))
    if alt:
        haric = os.path.abspath(CIKTI_DIZINI)
        dosyalar |= {f for f in glob.glob(os.path.join(d, "**", "em_records_*.csv.gz"), recursive=True)
                     if not os.path.abspath(f).startswith(haric + os.sep) or os.path.abspath(d).startswith(haric)}
    dosyalar = sorted(dosyalar)
    if not dosyalar:
        print(f"  HATA: '{d}' altında em_records_*.csv.gz bulunamadı. Eski kayıtları bu klasöre koyun; "
              f"kayıtlar kaybolduysa CALISMA_MODU = 'hub_yeniden_uret' ile Hub'daki eski modellerden yeniden üretin.")
        return {}
    print(f"  {len(dosyalar)} kayıt dosyası bulundu: " + ", ".join(os.path.basename(f) for f in dosyalar[:8])
          + (" ..." if len(dosyalar) > 8 else ""))
    parcalar = []
    for f in dosyalar:
        p = kayit_oku(f)
        p["_kaynak_dosya"] = os.path.basename(f)
        parcalar.append(p)
    df = pd.concat(parcalar, ignore_index=True)
    # Eski biçimde bölünme tohumu yoktur → REANALIZ_VERI'den (yeni biçim satırlarına dokunulmaz)
    yedek_split = df["dataset"].map(lambda x: REANALIZ_VERI.get(x, (None, None, SPLIT_SEED))[2]).astype(int)
    mevcut = pd.to_numeric(df["split_seed"], errors="coerce") if "split_seed" in df.columns else \
        pd.Series(np.nan, index=df.index)
    df["split_seed"] = mevcut.fillna(yedek_split).astype(int)
    df = kayitlari_df(df)
    _aile_denetimi(df)   # tekilleştirmeden ÖNCE: sürüm bilgisi taşıyan satırlar düşmeden denetlenir
    anahtar = ["dataset", "split_seed", "seed", "model", "sent_idx", "word_idx"]
    cift = df.duplicated(anahtar, keep=False)
    if cift.any():
        celiski = df[cift].groupby(anahtar)[["true_label", "pred_label", "is_em"]].nunique(dropna=False).max(axis=1) > 1
        if celiski.any():
            ornek = df[cift].merge(celiski[celiski].reset_index()[anahtar], on=anahtar)["_kaynak_dosya"].unique()[:6]
            raise ValueError(f"Aynı belirteç için ÇELİŞEN kayıtlar var (farklı koşumlar karışmış). İlgili dosyalar: "
                             f"{list(ornek)}. Klasörü tek bir koşum ailesiyle sınırlayın.")
        df = df.drop_duplicates(subset=anahtar)
        print(f"  Birebir aynı {int(cift.sum())} yinelenen satır tekilleştirildi.")
    eski = None
    if REANALIZ_ESKI_SONUC_JSON and os.path.exists(REANALIZ_ESKI_SONUC_JSON):
        with open(REANALIZ_ESKI_SONUC_JSON, encoding="utf-8") as f:
            eski = json.load(f)
        print(f"  E0 karşılaştırması için '{REANALIZ_ESKI_SONUC_JSON}' okundu.")
    baglamlar = _reanaliz_baglamlari(df)
    out = reanaliz_calistir(df, baglamlar, alt_dizin="reanaliz", eski_json=eski, e9=E9_TOKENIZASYON_ANALIZI)
    return {"ciktilar": out, "baglamlar": baglamlar}


def eski_modeli_yukle(yerel_dizin: str, m_cfg: Dict, label_mappings: Dict[str, Dict[str, int]]):
    """Eski koddaki save_pretrained çıktısını (config.json + *.safetensors) yükler."""
    from safetensors.torch import load_file
    cfg = MultiHeadConfig.from_pretrained(yerel_dizin)
    for h in MORPH_HEADS:
        if getattr(cfg, f"num_{h}_labels") != len(label_mappings[h]):
            raise ValueError(f"[{h}] etiket sayısı uyuşmuyor: model {getattr(cfg, f'num_{h}_labels')} ≠ "
                             f"veri {len(label_mappings[h])} (yanlış UD sürümü/bölünme?)")
    model = MultiHeadMorphModel(cfg)
    sd = {}
    for f in sorted(glob.glob(os.path.join(yerel_dizin, "*.safetensors"))):
        sd.update(load_file(f))
    if not sd:
        raise FileNotFoundError(f"{yerel_dizin} içinde .safetensors yok")
    eksik, beklenmeyen = model.load_state_dict(sd, strict=False)
    eksik = [k for k in eksik if ".pooler." not in k]
    if eksik or beklenmeyen:
        raise RuntimeError(f"Ağırlıklar tam yüklenemedi. Eksik: {eksik[:5]}  Beklenmeyen: {list(beklenmeyen)[:5]}")
    onceki = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()   # config.json'daki özel model_type uyarısını bastır
    try:
        tok = AutoTokenizer.from_pretrained(yerel_dizin)
    except Exception:
        tok = AutoTokenizer.from_pretrained(m_cfg["hf_id"], revision=_rev(m_cfg["hf_id"]))
    finally:
        transformers.logging.set_verbosity(onceki)
    return model, tok


def hub_yeniden_uret_modu() -> Dict:
    """em_records kaybolduysa: Hub'daki eski modellerle test tahminlerini yeniden üretir."""
    from huggingface_hub import snapshot_download
    hf_oturum_kontrol()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hedef = os.path.dirname(cikti_yolu("hub_kayitlari", "x"))
    tum = []
    for ds, (tag, anot, sp) in REANALIZ_VERI.items():
        ham = veri_seti_yukle(ds, _eski_spec(ds, tag, anot))
        bag = bolunme_baglami_kur(ds, ham, sp, sessiz=True)
        for seed in SEEDS:
            frames = []
            for m in MODELS:
                repo = HUB_ESKI_DEPO_SABLONU.format(user=HF_USERNAME, short=m["short"], ds=ds.lower(), seed=seed)
                try:
                    yerel = snapshot_download(repo_id=repo)
                    model, tok = eski_modeli_yukle(yerel, m, bag.label_mappings)
                except Exception as e:
                    print(f"  ⚠️  {repo} yüklenemedi: {e}")
                    continue
                model.to(device)
                tr_ = batch_predict(model, tok, bag.splits["test"], bag.label_mappings, device=device)
                recs = token_kayitlari(tr_, bag.splits["test"], bag.held_out, bag.train_bundles, bag.train_forms)
                frames.append(_kayit_meta(pd.DataFrame(recs), ds, bag, seed, m))
                print(f"  {repo}: {len(recs):,} belirteç tahmini üretildi.")
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if frames:
                d = pd.concat(frames, ignore_index=True)
                d.to_csv(os.path.join(hedef, f"em_records_{ds.lower()}_seed{seed}.csv.gz"), index=False, compression="gzip")
                tum.append(d)
    if not tum:
        print("  Hiçbir model indirilemedi (Hub girişi: from huggingface_hub import login; login()).")
        return {}
    artimli_yedek("v5: Hub modellerinden em_records yeniden üretildi")
    return reanaliz_modu(hedef, alt_klasorler=False)


def main():
    hizli_mesaj = _hizli_test_ayarlari()
    _drive_dizini_coz()
    _log_kur()
    with konsolu_dosyaya_kopyala(cikti_yolu("konsol_ciktisi.log")):
        print(f"\n{'#' * 72}\n  TEZ DENEY KODU v5 — mod: {CALISMA_MODU}  |  çıktı: {os.path.abspath(CIKTI_DIZINI)}"
              f"  |  Drive yedeği: {DRIVE_YEDEK_DIZINI or 'kapalı'}\n{'#' * 72}")
        if hizli_mesaj:
            print(hizli_mesaj)
        determinizm_uyarisi()
        if CALISMA_MODU == "deney":
            return run_thesis_experiments()
        if CALISMA_MODU == "veri":
            return veri_modu()
        if CALISMA_MODU == "reanaliz":
            return reanaliz_modu()
        if CALISMA_MODU == "hub_yeniden_uret":
            return hub_yeniden_uret_modu()
        raise ValueError(f"Bilinmeyen CALISMA_MODU: {CALISMA_MODU}")


if __name__ == "__main__":
    main()