# -*- coding: utf-8 -*-
"""Düzeltme 6.4 (BULGU-029): Şekil 4.3'ü noktaları çizgiyle BİRLEŞTİRMEDEN yeniden çizer.

Bölünmeler (0, 1, 2, 4) sıralı bir değişkenin değerleri değil, birbirinden bağımsız held-out
seçimleridir; noktaları çizgiyle bağlamak aralarında bir eğilim varmış izlenimi verir. Bu betik
aynı verilerle (üç tohumun ortalaması ± örneklem standart sapması) yalnızca nokta ve hata çubuğu çizer.

Girdi: analizler/imst_r2.18_ham_split{0,1,2,4}/tohum_em.csv (sonuç deposunda hazır)
Çıktı: sekil_4_3.png (300 dpi) ve sekil_4_3.pdf (vektör; Word'e PDF/EMF olarak da eklenebilir)

Kullanım:
  python sekil_4_3.py --sonuc final-thesis-results-v5 --cikti tez_ek_ciktilar
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

MODELLER = [("BERTurk", "#1f4e79", "o"), ("mBERT", "#d9822b", "s"), ("XLM-R", "#4c9a5f", "^")]
BOLUNMELER = [0, 1, 2, 4]


def tr_sayi(x, ond=1):
    s = f"{x:.{ond}f}".replace(".", ",")
    return s[:-2] if s.endswith(",0") else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sonuc", required=True)
    ap.add_argument("--cikti", default="tez_ek_ciktilar")
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(os.path.join(a.sonuc, "analizler", f"imst_r2.18_ham_split{sp}", "tohum_em.csv"))
                    for sp in BOLUNMELER])
    oz = df.groupby(["split_seed", "model"]).agg(seen_ort=("seen_em", "mean"), seen_s=("seen_em", "std"),
                                                 unseen_ort=("unseen_em", "mean"), unseen_s=("unseen_em", "std")).reset_index()

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                         "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
    kaydirma = {"BERTurk": -0.12, "mBERT": 0.0, "XLM-R": 0.12}
    for ax, (anahtar, baslik, ylim) in zip(axes, [("seen", "(a) Seen EM", (75, 95)), ("unseen", "(b) Unseen EM", (20, 95))]):
        for m, renk, isaret in MODELLER:
            d = oz[oz.model == m].set_index("split_seed").reindex(BOLUNMELER)
            x = [i + kaydirma[m] for i in range(len(BOLUNMELER))]
            ax.errorbar(x, 100 * d[f"{anahtar}_ort"], yerr=100 * d[f"{anahtar}_s"],
                        fmt=isaret, linestyle="none",            # ← çizgi yok: yalnız nokta + hata çubuğu
                        color=renk, markersize=5.5, capsize=3, elinewidth=1.2, label=m)
        ax.set_xticks(range(len(BOLUNMELER)))
        ax.set_xticklabels(["0 (ana)", "1", "2", "4"])
        ax.set_xlim(-0.5, len(BOLUNMELER) - 0.5)
        ax.set_xlabel("Bölünme (SPLIT_SEED)")
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: tr_sayi(y)))
        ax.set_title(baslik)
        ax.grid(axis="y", alpha=0.35, linestyle=":")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("EM (%)")
    axes[1].legend(loc="lower left", frameon=True, fontsize=9)
    fig.tight_layout()
    os.makedirs(a.cikti, exist_ok=True)
    for uz in ("png", "pdf"):
        yol = os.path.join(a.cikti, f"sekil_4_3.{uz}")
        fig.savefig(yol, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  → {yol}")
    print(oz.assign(**{c: (100 * oz[c]).round(2) for c in ["seen_ort", "seen_s", "unseen_ort", "unseen_s"]}).to_string(index=False))


if __name__ == "__main__":
    main()
