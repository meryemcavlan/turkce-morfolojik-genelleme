# Türkçe Morfolojik Etiketlemede Görülmemiş Özellik Demetlerine Genelleme

Bu depo, Meryem Çavlan'ın yüksek lisans tezinin (Afyon Kocatepe Üniversitesi, Fen Bilimleri Enstitüsü, 2026; danışman Dr. Öğr. Üyesi Özkan Aslan) deney ve analiz kodunu içerir.

- **Sonuçlar ve belirteç düzeyi tahmin kayıtları:** https://huggingface.co/datasets/meryemmm22/final-thesis-results-v5
- **İnce ayarlı modeller (ana bölünmeler, 18 depo):** https://huggingface.co/meryemmm22 altında `{berturk,mbert,xlmr}-{imst-r2.18,boun-r2.10}-ham-split0-seed{42,23,56}-v5`
- **Kalıcı arşiv (DOI):** [https://doi.org/10.5281/zenodo.23461408]

## Klasör yapısı

```
tez_deney_kodu_v5.py          Ana deney kodu (bölünme, eğitim, değerlendirme, analizler E0–E10)
analiz/ortak.py               Analiz betiklerinin ortak yardımcıları (UD okuma, kontrollü bölünme)
analiz/ek_analizler.py        Ana kodun üretmediği ya da yalnızca kısmen ürettiği çizelgeler: 3.4 (atom
                              korunma satırı), 4.9, 4.17–4.22, Ek 2.2, Ek 5.1; karşılaştırma için 4.8, 4.15–4.16
analiz/iki_asamali_bootstrap.py  Çizelge 4.3 son sütunu: demet × tohum bootstrap GA
analiz/sekil_4_3.py           Şekil 4.3
analiz/model_surumleri.py     Ön eğitimli modellerin Hub commit kimlikleri ve ağırlık parmak izleri
analiz/hf_herkese_ac.py       Sonuç ve model depolarının görünürlüğünü denetler / herkese açar
requirements_analiz.txt       Analiz betikleri için paketler
manifest/                     pip_freeze.txt, calisma_manifesti.json, determinizm_sinamasi.json,
                              model_surumleri.json
```

## Ortam

Deneyler Google Colab'da tek bir NVIDIA A100 40 GB GPU üzerinde, 13–14 Ekim 2026'da çalıştırıldı.
Tüm paket sürümleri `manifest/pip_freeze.txt` dosyasındadır. Temel sürümler: Python 3.13.15,
PyTorch 2.11.0 (CUDA 13.0), Transformers 5.19.0, Tokenizers 0.23.2.

| Model | Hub kimliği | Revision (commit) |
|---|---|---|
| BERTurk | dbmdz/bert-base-turkish-cased | ["b6e1de16c983e0f2c70664591ea3f22810072608"] |
| mBERT | bert-base-multilingual-cased | ["3f076fdb1ab68d5b2880cb87a0886f315b8146f8"] |
| XLM-R | xlm-roberta-base | ["e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"] |

| Ağaç bankası | Etiket | Commit |
|---|---|---|
| UD_Turkish-IMST | r2.18 | 64da444f835fdb32630ebd9c3f1f00760dbf3c53 |
| UD_Turkish-BOUN | r2.10 | 17ff1b9f5645f2feb4b68b829c0841dcb7304687 |

## Yeniden üretim

Ana koddaki `CALISMA_MODU` değişkeni ne yapılacağını belirler:

1. **Yalnızca bölünmeler ve veri analizleri** (GPU gerekmez, birkaç dakika): `CALISMA_MODU = "veri"`.
   UD depoları yukarıdaki etiketlerle `ud_data/` altına klonlanır; held-out listeleri ve Çizelge 3.4'ün
   değerleri `veri/` klasörüne yazılır.
2. **Tam deney** (45 koşu; A100'de yaklaşık 3,2 saat eğitim): `CALISMA_MODU = "deney"`.
3. **Analizleri kayıtlardan yeniden üretmek** (GPU gerekmez): sonuç deposunu indirin, `em_records_*.csv.gz`
   dosyalarını `REANALIZ_KAYIT_DIZINI` klasörüne koyun ve `CALISMA_MODU = "reanaliz"`.

Ek analizler (ana kod `veri` modunda bir kez çalıştırılıp `ud_data/` oluşturulduktan sonra):

```bash
pip install pandas numpy matplotlib huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('meryemmm22/final-thesis-results-v5', repo_type='dataset', local_dir='final-thesis-results-v5')"
cd analiz
python ek_analizler.py          --sonuc ../final-thesis-results-v5 --ud ../ud_data --cikti ../tez_ek_ciktilar
python iki_asamali_bootstrap.py --sonuc ../final-thesis-results-v5 --cikti ../tez_ek_ciktilar
python sekil_4_3.py             --sonuc ../final-thesis-results-v5 --cikti ../tez_ek_ciktilar
```

`ek_analizler.py` held-out demet listelerini sonuç deposundaki `veri/veri_ozeti_*.json` dosyalarıyla
karşılaştırır; UD sürümü yanlışsa çalışmayı durdurur.

## Atıf

Çavlan M, 2026, Türkçe Morfolojik Etiketlemede Transformer Tabanlı Modellerin Görülmemiş Özellik
Demetlerine Genellemesinin Kontrollü Bölünmelerle Ölçülmesi, Yüksek Lisans Tezi, Afyon Kocatepe
Üniversitesi Fen Bilimleri Enstitüsü, Afyonkarahisar.

## Lisans

Kod: MIT. UD verileri kendi lisanslarıyla (IMST: CC BY-NC-SA 4.0; BOUN: CC BY-SA 4.0) dağıtılır ve bu depoya dahil edilmemiştir.
