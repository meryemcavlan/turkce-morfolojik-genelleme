# -*- coding: utf-8 -*-
"""Ana deney kodunun üretmediği ya da yalnızca kısmen ürettiği çizelgeleri belirteç kayıtlarından yeniden üretir.
(Çizelge 4.8 ve 4.15–4.16 ana kodda da üretilir; burada çapraz denetim için yeniden hesaplanır.)

Üretilen dosyalar (cikti klasörüne, CSV):
  cizelge_3_4_atom_korunma.csv   Çizelge 3.4'e eklenecek "en düşük korunma oranı" satırı
  cizelge_4_8_bicim.csv          Çizelge 4.8 (IV/OOV EM)
  cizelge_4_9_seen_aday.csv      Çizelge 4.9 (Seen-aday karşılaştırması)
  cizelge_4_15_16_baslik.csv     Çizelge 4.15–4.16 (başlık başına Seen/Unseen doğruluk)
  cizelge_4_17_hata.csv          Çizelge 4.17 (seçilmiş hata örüntüleri)
  cizelge_4_18_upos_payi.csv     Çizelge 4.18
  cizelge_4_19_voice_cau.csv     Çizelge 4.19
  cizelge_4_20_bicimler.csv      Çizelge 4.20
  cizelge_4_21_dat_loc.csv       Çizelge 4.21
  cizelge_4_22_birlesim.csv      Çizelge 4.22 ve metindeki referans oranları
  ek_2_2_mikro_f1.csv            Çizelge Ek 2.2
  ek_5_1_duyarlilik.csv          Çizelge Ek 5.1

Kullanım:
  python ek_analizler.py --sonuc final-thesis-results-v5 --ud ud_data --cikti tez_ek_ciktilar
"""
import argparse
import re
from collections import Counter

import numpy as np
import pandas as pd

from ortak import (BASLIKLAR, KOSULLAR, MODELLER, TOHUMLAR, atom_sayimi, atomlar, baglamlar, demet,
                   kayitlari_yukle, kaydet, tohum_ort, upos_demet_kumesi)

FK = {"case": "Case", "number": "Number", "person": "Person", "tense": "Tense", "aspect": "Aspect",
      "mood": "Mood", "voice": "Voice", "polarity": "Polarity", "verbform": "VerbForm"}
DIR_SONU = re.compile(r"(?:dır|dir|dur|dür|tır|tir|tur|tür)$")
DIR_HERHANGI = re.compile(r"(?:dır|dir|dur|dür|tır|tir|tur|tür)")
NS = "Case=Nom|Number=Sing|Person=3"
AS = "Case=Acc|Number=Sing|Person=3"


def c34_atom_korunma(B):
    """Held-out atomlarının ham eğitimden kontrollü eğitime korunma oranı; her koşulda en düşüğü."""
    rows = []
    for k, b in B.items():
        ham = atom_sayimi(b["ud"]["train"])
        kon = atom_sayimi(b["egitim"])
        ho_atom = sorted({a for d in b["held_out"] for a in atomlar(d)})
        oran = {a: kon[a] / ham[a] for a in ho_atom if ham[a]}
        en = min(oran, key=oran.get)
        rows.append({"kosul": k, "atom": en, "kontrollu": kon[en], "ham": ham[en],
                     "korunma_yuzde": round(100 * oran[en], 1),
                     "tez_hucresi": f"{en} {round(100 * oran[en]):d}"})
    return pd.DataFrame(rows)


def c48_bicim(D):
    rows = []
    for (k, m), g in D.groupby(["kosul", "model"], sort=False):
        s, iv = g.is_seen, g.form_in_train
        v = [tohum_ort(g, s & iv), tohum_ort(g, s & ~iv), tohum_ort(g, ~s & iv), tohum_ort(g, ~s & ~iv)]
        rows.append({"kosul": k, "model": m, "seen_iv": v[0], "seen_oov": v[1], "unseen_iv": v[2], "unseen_oov": v[3],
                     "unseenoov_eksi_seenoov": v[3] - v[1],
                     "n_seen_iv": (s & iv).sum() // 3, "n_seen_oov": (s & ~iv).sum() // 3,
                     "n_unseen_iv": (~s & iv).sum() // 3, "n_unseen_oov": (~s & ~iv).sum() // 3})
    return pd.DataFrame(rows).round(2)


def c49_seen_aday(D, B):
    rows = []
    for (k, m), g in D.groupby(["kosul", "model"], sort=False):
        b = B[k]
        s, nn, oov = g.is_seen, g.true_bundle != "NONE", ~g.form_in_train
        ad = s & nn & g.true_bundle.isin(b["aday"]) & ~g.true_bundle.isin(b["held_out"])
        v = [tohum_ort(g, s), tohum_ort(g, s & nn), tohum_ort(g, ad), tohum_ort(g, ad & oov),
             tohum_ort(g, ~s), tohum_ort(g, ~s & oov)]
        rows.append({"kosul": k, "model": m, "seen": v[0], "seen_none_haric": v[1], "seen_aday": v[2],
                     "seen_aday_oov": v[3], "unseen": v[4], "unseen_oov": v[5],
                     "unseen_eksi_seenaday": v[4] - v[2], "unseenoov_eksi_seenadayoov": v[5] - v[3],
                     "n_seen_none_haric": (s & nn).sum() // 3, "n_seen_aday": ad.sum() // 3,
                     "n_seen_aday_oov": (ad & oov).sum() // 3, "n_unseen_oov": (~s & oov).sum() // 3})
    return pd.DataFrame(rows).round(2)


def c4151_baslik(D):
    rows = []
    for (k, m), g in D[D.split_seed == 0].groupby(["kosul", "model"], sort=False):
        for h in BASLIKLAR:
            f = lambda z, h=h: (z["true_" + h] == z["pred_" + h]).mean()
            rows.append({"kosul": k, "model": m, "baslik": h,
                         "seen": tohum_ort(g, g.is_seen, f), "unseen": tohum_ort(g, ~g.is_seen, f)})
    return pd.DataFrame(rows).round(1)


def _ekle(b, a):
    return "|".join(sorted(atomlar(b) + a.split("|")))


def c417_hata(D):
    oruntuler = {
        "NONE → Nom|Sing|3": lambda z: (z.true_bundle == "NONE") & (z.pred_bundle == NS),
        "Nom|Sing|3 → NONE": lambda z: (z.true_bundle == NS) & (z.pred_bundle == "NONE"),
        "Nom|Sing|3 → Acc|Sing|3": lambda z: (z.true_bundle == NS) & (z.pred_bundle == AS),
        "Acc|Sing|3 → Nom|Sing|3": lambda z: (z.true_bundle == AS) & (z.pred_bundle == NS),
        "(U) +Voice=Cau": lambda z: ~z.is_seen & (z.pred_bundle == z.true_bundle.map(lambda b: _ekle(b, "Voice=Cau"))),
        "(U) +Nom|Sing|3": lambda z: ~z.is_seen & (z.pred_bundle == z.true_bundle.map(lambda b: _ekle(b, NS))),
    }
    rows = []
    for k in ("IMST 0", "BOUN 0"):
        g = D[D.kosul == k]
        for ad, f in oruntuler.items():
            x = g[f(g)]
            # (U) satırları gerçek demete göre ayrılır (tezde demet başına satır var)
            gruplar = x.groupby("true_bundle") if ad.startswith("(U)") else [("", x)]
            for tb, y in gruplar:
                r = {"kosul": k, "oruntu": ad, "gercek_demet": tb}
                for m in MODELLER:
                    r[m] = " / ".join(str(int(((y.model == m) & (y.seed == t)).sum())) for t in TOHUMLAR)
                rows.append(r)
    return pd.DataFrame(rows)


def c418_upos(D):
    rows = []
    for k in ("IMST 0", "BOUN 0"):
        g = D[D.kosul == k]
        for ad, f in [("Nom|Sing|3 → NONE", lambda z: (z.true_bundle == NS) & (z.pred_bundle == "NONE")),
                      ("NONE → Nom|Sing|3", lambda z: (z.true_bundle == "NONE") & (z.pred_bundle == NS))]:
            r = {"kosul": k, "yon": ad}
            for m in MODELLER:
                x = g[(g.model == m) & f(g)]
                c = x.groupby("seed").size()
                r[m + "_hata"] = f"{c.min()}–{c.max()}"
                r[m + "_upos_yanlis_yuzde"] = round(100 * (x.true_upos != x.pred_upos).mean(), 1)
            rows.append(r)
    return pd.DataFrame(rows)


def c419_voice(D, B):
    rows = []
    for k, b in B.items():
        g = D[D.kosul == k]
        r = {"kosul": k}
        for m in MODELLER:
            fz = [int(((g.model == m) & (g.seed == t) & (g.true_voice == "NONE") & (g.pred_voice == "Cau")).sum())
                  for t in TOHUMLAR]
            r[m + "_fazladan_42_23_56"] = " / ".join(map(str, fz))
        dmask = g.token.str.lower().str.contains(DIR_SONU) & (g.true_voice == "NONE")
        r["dir_atanan"] = int((g[dmask].pred_voice == "Cau").sum())
        r["dir_karar"] = int(dmask.sum())
        r["dir_belirtec"] = int(dmask.sum() // 9)
        r["egitimde_voice_cau"] = atom_sayimi(b["egitim"])["Voice=Cau"]
        for m in MODELLER:
            x = g[(g.model == m) & g.is_seen & (g.true_voice == "Cau")]
            r[m + "_seen_duyarlilik"] = round(100 * x.groupby("seed").apply(lambda z: (z.pred_voice == "Cau").mean()).mean(), 1)
            pc = sorted(int(((g.model == m) & (g.seed == t) & (g.pred_voice == "Cau")).sum()) for t in TOHUMLAR)
            r[m + "_kosu_basina_tahmin"] = f"{pc[0]}–{pc[-1]}"
        rows.append(r)
    return pd.DataFrame(rows)


def c420_bicimler(D, n=12):
    g = D[(D.kosul == "IMST 0") & (D.model == "BERTurk") & (D.true_voice == "NONE") & (D.pred_voice == "Cau")]
    c = Counter(g.token.str.lower())
    df = pd.DataFrame(sorted(c.items(), key=lambda x: (-x[1], x[0])), columns=["bicim", "hata_uc_tohum"])
    print(f"    Çiz. 4.20: üç tohumda tam 3 hata alan biçim sayısı = {(df.hata_uc_tohum == 3).sum()}")
    return df


def c421_dat_loc(D):
    rows = []
    for k in D.kosul.unique():
        g = D[D.kosul == k]
        r = {"kosul": k}
        for m in MODELLER:
            x = g[g.model == m]
            dl = [int(((x.seed == t) & (x.true_case == "Dat") & (x.pred_case == "Loc")).sum()) for t in TOHUMLAR]
            ld = [int(((x.seed == t) & (x.true_case == "Loc") & (x.pred_case == "Dat")).sum()) for t in TOHUMLAR]
            r[m + "_Dat→Loc"] = f"{min(dl)}–{max(dl)}"
            r[m + "_Loc→Dat"] = f"{min(ld)}–{max(ld)}"
            if k == "IMST 4":
                for ad, mk in (("seen", x.is_seen), ("unseen", ~x.is_seen)):
                    v = [int(((x.seed == t) & mk & (x.true_case == "Loc") & (x.pred_case == "Dat")).sum()) for t in TOHUMLAR]
                    r[f"{m}_Loc→Dat_{ad}"] = f"{min(v)}–{max(v)}"
        rows.append(r)
    return pd.DataFrame(rows)


def c422_birlesim(D, B):
    rows = []
    for k, b in B.items():
        tr = b["egitim"]
        TR = upos_demet_kumesi(tr)
        TB = upos_demet_kumesi(b["ud"]["train"] + b["ud"]["dev"] + b["ud"]["test"])
        ciftler = {(u, f) for s in tr for u, fd in zip(s["upos"], s["feats"]) for f in fd if f in FK.values()}
        g = D[D.kosul == k]
        # Referans: Unseen altın etiketlerinde tek başlık değişimi (değer kümeleri kontrollü eğitimden)
        deger = {"upos": Counter(u for s in tr for u in s["upos"])}
        for h, f in FK.items():
            deger[h] = Counter(fd.get(f, "NONE") for s in tr for fd in s["feats"])
        U = g[(g.model == MODELLER[0]) & (g.seed == TOHUMLAR[0]) & ~g.is_seen]
        tb_tok, duz, agirlik_pay, agirlik_payda = [], [], 0.0, 0.0
        for _, r in U.iterrows():
            komsu = set()
            for h in BASLIKLAR:
                alt = [v for v in deger[h] if v != r["true_" + h]]
                top = sum(deger[h][v] for v in alt)
                for v in alt:
                    up = v if h == "upos" else r.true_upos
                    fd = {FK[x]: (v if x == h else r["true_" + x]) for x in FK if (v if x == h else r["true_" + x]) != "NONE"}
                    lab = up + "+" + demet(fd)
                    duz.append(lab in TR)
                    agirlik_pay += (lab in TR) * deger[h][v] / top / len(BASLIKLAR)
                    agirlik_payda += deger[h][v] / top / len(BASLIKLAR)
                    if lab in TB:
                        komsu.add(lab)
            if komsu:
                tb_tok.append(np.mean([c in TR for c in komsu]))
        ref_tb, ref_duz, ref_agr = 100 * np.mean(tb_tok), 100 * np.mean(duz), 100 * agirlik_pay / agirlik_payda
        w = g[~g.is_em]
        nh = sum((w["true_" + h] != w["pred_" + h]).astype(int) for h in BASLIKLAR)
        wu = w[~w.is_seen]
        nhu = nh[~w.is_seen]
        for m in MODELLER:
            per = []
            for t, z in g[g.model == m].groupby("seed"):
                gec = z.apply(lambda r: any((r.pred_upos, FK[h]) not in ciftler for h in FK if r["pred_" + h] != "NONE"), axis=1).mean()
                u = z[~z.is_seen]
                hu, hs, hz = u[~u.is_em], z[z.is_seen & ~z.is_em], z[~z.is_em]
                per.append([gec, (~z.pl.isin(TB)).mean(), (~u.pl.isin(TR)).mean(), hu.pl.isin(TR).mean(),
                            hs.pl.isin(TR).mean(), (~hz.pl.isin(TB)).mean()])
            P = 100 * np.array(per)
            rows.append({"kosul": k, "model": m, "gecersiz_upos_ozellik": round(P[:, 0].mean(), 2),
                         "agac_bankasinda_yok": round(P[:, 1].mean(), 2),
                         "unseen_egitimde_gorulmemis_tahmin": round(P[:, 2].mean(), 1),
                         "hatali_unseen_egitimde_gorulmus": f"{P[:, 3].mean():.1f} ({P[:, 3].min():.1f}–{P[:, 3].max():.1f})",
                         "hatali_seen_egitimde_gorulmus": f"{P[:, 4].mean():.1f} ({P[:, 4].min():.1f}–{P[:, 4].max():.1f})",
                         "agacta_yok_hatalarin_payi": round(P[:, 5].mean(), 1),
                         "ref_agac_bankasiyla_sinirli": round(ref_tb, 1),
                         "ref_tekduze": round(ref_duz, 1), "ref_siklik_agirlikli": round(ref_agr, 1),
                         "havuz_hatali_unseen_egitimde": round(100 * wu.pl.isin(TR).mean(), 1),
                         "havuz_hatali_seen_egitimde": round(100 * w[w.is_seen].pl.isin(TR).mean(), 1),
                         "havuz_hatali_unseen_agacta": round(100 * wu.pl.isin(TB).mean(), 1),
                         "tek_baslik_hata_payi_unseen": round(100 * (nhu == 1).mean(), 1),
                         "tek_baslik_hata_sayi": f"{int((nhu == 1).sum())}/{len(wu)}"})
    return pd.DataFrame(rows)


def ek22_mikro_f1(D):
    rows = []
    for (k, m), g in D[D.split_seed == 0].groupby(["kosul", "model"], sort=False):
        v = []
        for t, z in g.groupby("seed"):
            sat = []
            for h in BASLIKLAR:
                gt, pr = z["true_" + h], z["pred_" + h]
                tp = ((gt == pr) & (gt != "NONE")).sum()
                fp = ((pr != "NONE") & (pr != gt)).sum()
                fn = ((gt != "NONE") & (pr != gt)).sum()
                sat.append(2 * tp / (2 * tp + fp + fn))
            v.append(sat)
        v = 100 * np.mean(v, axis=0)
        r = {"kosul": k, "model": m, **{h: round(x, 1) for h, x in zip(BASLIKLAR, v)}, "ortalama": round(v.mean(), 2)}
        rows.append(r)
    return pd.DataFrame(rows)


def ek51_duyarlilik(D, B, esik=85.0):
    rows = []
    for k, b in B.items():
        kon = atom_sayimi(b["egitim"])
        g = D[(D.kosul == k) & D.is_seen]
        ho_atom = sorted({a for d in b["held_out"] for a in atomlar(d)})
        for a in ho_atom:
            f, v = a.split("=")
            col = [h for h, x in FK.items() if x == f][0]
            x = g[g["true_" + col] == v]
            s = {m: 100 * x[x.model == m].groupby("seed").apply(lambda z: (z["pred_" + col] == v).mean()).mean() for m in MODELLER}
            rows.append({"kosul": k, "atom": a, "kontrollu_egitimde": kon[a],
                         "seen_test": int(len(x) // 9), **{m: round(s[m], 1) for m in MODELLER},
                         "cizelgede": min(s.values()) < esik})
    df = pd.DataFrame(rows)
    for k, g in df.groupby("kosul", sort=False):
        print(f"    Ek 5.1 {k}: {int((~g.cizelgede).sum())}/{len(g)} atom üç modelde de ≥ %{esik:.0f}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sonuc", required=True, help="final-thesis-results-v5 klasörü")
    ap.add_argument("--ud", required=True, help="UD kök klasörü (ana kodun ud_data klasörü)")
    ap.add_argument("--cikti", default="tez_ek_ciktilar")
    a = ap.parse_args()
    print("Belirteç kayıtları okunuyor…")
    D = kayitlari_yukle(a.sonuc)
    print(f"  {len(D):,} satır, {D.groupby(['kosul', 'model', 'seed']).ngroups} koşu")
    print("Kontrollü bölünmeler yeniden kuruluyor…")
    B = baglamlar(a.ud, a.sonuc)
    adimlar = [("cizelge_3_4_atom_korunma.csv", lambda: c34_atom_korunma(B)),
               ("cizelge_4_8_bicim.csv", lambda: c48_bicim(D)),
               ("cizelge_4_9_seen_aday.csv", lambda: c49_seen_aday(D, B)),
               ("cizelge_4_15_16_baslik.csv", lambda: c4151_baslik(D)),
               ("cizelge_4_17_hata.csv", lambda: c417_hata(D)),
               ("cizelge_4_18_upos_payi.csv", lambda: c418_upos(D)),
               ("cizelge_4_19_voice_cau.csv", lambda: c419_voice(D, B)),
               ("cizelge_4_20_bicimler.csv", lambda: c420_bicimler(D)),
               ("cizelge_4_21_dat_loc.csv", lambda: c421_dat_loc(D)),
               ("cizelge_4_22_birlesim.csv", lambda: c422_birlesim(D, B)),
               ("ek_2_2_mikro_f1.csv", lambda: ek22_mikro_f1(D)),
               ("ek_5_1_duyarlilik.csv", lambda: ek51_duyarlilik(D, B))]
    for ad, f in adimlar:
        print(ad)
        kaydet(f(), a.cikti, ad)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()
