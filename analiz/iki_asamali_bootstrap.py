# -*- coding: utf-8 -*-
"""BULGU-031: Çizelge 4.3'ün son sütunu için iki aşamalı (demet × tohum) bootstrap.

Sorun: Çizelge 4.3'teki "Unseen demet bootstrap %95 GA" sütunu, her tohum için ayrı hesaplanan
demet bootstrap aralıklarının tohumlar üzerinden ORTALAMASIDIR. Aralıkların ortalaması bir güven
aralığı değildir: ne demet seçiminden ne de eğitim tohumundan gelen belirsizliği doğru birleştirir.

Çözüm: Her bootstrap örnekleminde
  1) held-out demetler yerine koyarak yeniden çekilir (K demetten K tane),
  2) eğitim tohumları yerine koyarak yeniden çekilir (3 tohumdan 3 tane),
  3) Unseen EM = seçilen demetlerin seçilen tohumlardaki doğru belirteç sayısı / belirteç sayısı.
Bu, "üç tohumun ortalaması olan Unseen EM"nin hem demet hem tohum belirsizliğini yansıtan tek bir
yüzdelik aralığını verir. Aynı örneklemlerle model farkları için de aralık hesaplanır (demetler iki
modelde ortak çekilir; tohumlar her model için bağımsız çekilir, çünkü farklı modellerin tohumları eşli değildir).

Not: 3 tohumla tohum düzeyi yeniden örnekleme kaba bir yaklaşımdır ve 16–18 demetle yüzdelik
bootstrap'ın kapsamı nominal %95'in altında kalabilir (Bölüm 3.12). Aralık betimsel yorumlanmalıdır.

Kullanım:
  python iki_asamali_bootstrap.py --sonuc final-thesis-results-v5 --cikti tez_ek_ciktilar [--n 10000]
"""
import argparse

import numpy as np
import pandas as pd

from ortak import MODELLER, kayitlari_yukle, kaydet


def sayim_matrisi(g):
    """Unseen belirteçler: demet × tohum doğru sayısı ve demet büyüklüğü."""
    u = g[~g.is_seen]
    k = u.pivot_table(index="true_bundle", columns="seed", values="is_em", aggfunc="sum").sort_index()
    n = u[u.seed == u.seed.iloc[0]].groupby("true_bundle").size().reindex(k.index)
    return k.values.astype(float), n.values.astype(float), list(k.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sonuc", required=True)
    ap.add_argument("--cikti", default="tez_ek_ciktilar")
    ap.add_argument("--n", type=int, default=10000, help="bootstrap örneklem sayısı")
    ap.add_argument("--tohum", type=int, default=0, help="bootstrap rastgele sayı tohumu")
    a = ap.parse_args()
    D = kayitlari_yukle(a.sonuc)
    satirlar, farklar = [], []
    for kosul in ["IMST 0", "IMST 1", "IMST 2", "IMST 4", "BOUN 0"]:
        g = D[D.kosul == kosul]
        M = {m: sayim_matrisi(g[g.model == m]) for m in MODELLER}
        K = len(M[MODELLER[0]][2])
        T = M[MODELLER[0]][0].shape[1]
        assert all(M[m][2] == M[MODELLER[0]][2] for m in MODELLER), "demet kümeleri modeller arasında farklı"
        rng = np.random.RandomState(a.tohum)
        dem = rng.multinomial(K, np.full(K, 1 / K), size=a.n)              # (n, K) demet ağırlıkları
        toh = {m: rng.multinomial(T, np.full(T, 1 / T), size=a.n) for m in MODELLER}  # (n, T) tohum ağırlıkları
        ornek = {}
        for m in MODELLER:
            k, n, _ = M[m]
            # her örneklem: Σ_b w_b Σ_s v_s k_bs / (T Σ_b w_b n_b)
            pay = np.einsum("ib,bs,is->i", dem, k, toh[m])
            payda = T * (dem @ n)
            ornek[m] = pay / payda
            nokta = k.sum() / (T * n.sum())
            # karşılaştırma için: yalnız demet bootstrapı (tohum ortalaması sabit)
            yalniz_demet = (dem @ k.mean(axis=1)) / (dem @ n)
            satirlar.append({"kosul": kosul, "model": m, "unseen_em": round(100 * nokta, 2),
                             "iki_asamali_ga_alt": round(100 * np.percentile(ornek[m], 2.5), 1),
                             "iki_asamali_ga_ust": round(100 * np.percentile(ornek[m], 97.5), 1),
                             "yalniz_demet_ga_alt": round(100 * np.percentile(yalniz_demet, 2.5), 1),
                             "yalniz_demet_ga_ust": round(100 * np.percentile(yalniz_demet, 97.5), 1)})
        for i in range(len(MODELLER)):
            for j in range(i + 1, len(MODELLER)):
                ma, mb = MODELLER[i], MODELLER[j]
                d = ornek[ma] - ornek[mb]
                nokta = (M[ma][0].sum() - M[mb][0].sum()) / (T * M[ma][1].sum())
                farklar.append({"kosul": kosul, "karsilastirma": f"{ma} − {mb}", "fark_puan": round(100 * nokta, 2),
                                "ga_alt": round(100 * np.percentile(d, 2.5), 1),
                                "ga_ust": round(100 * np.percentile(d, 97.5), 1),
                                "ga_sifiri_dislar": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)})
    kaydet(pd.DataFrame(satirlar), a.cikti, "bulgu031_iki_asamali_bootstrap.csv")
    kaydet(pd.DataFrame(farklar), a.cikti, "bulgu031_iki_asamali_bootstrap_farklar.csv")
    print(pd.DataFrame(satirlar).to_string(index=False))
    print(pd.DataFrame(farklar).to_string(index=False))


if __name__ == "__main__":
    main()
