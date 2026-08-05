# Data

Ovaj direktorijum sadrži podatke korišćene u projektu.

## Struktura
- `raw/` — originalni preuzeti podaci (ignorisani u git-u)
- `processed/` — obrađeni podaci spremni za trening (ignorisani u git-u)

## Izvor
Dataset: **ShanghaiTech** (Part A and Part B)

Link: https://www.kaggle.com/datasets/tthien/shanghaitech

## Preuzimanje

```bash
# One-liner:
pip install kagglehub
python scripts/download_data.py
```

## Sadržaj dataseta

| Stavka | Vrednost |
|--------|----------|
| Part A | 482 slike (300 train, 182 test), guste gužve |
| Part B | 716 slika (400 train, 316 test), ređe scene |
| Anotacije | (x,y) koordinate glava u .mat fajlovima |
| Veličina | ~333 MB |

### Očekivana struktura
```
data/raw/
  ShanghaiTech/
    part_A/
      train_data/
        images/     (IMG_*.jpg)
        ground-truth/  (GT_IMG_*.mat)
      test_data/
        images/
        ground-truth/
    part_B/
      train_data/
        images/
        ground-truth/
      test_data/
        images/
        ground-truth/
```

## Preuzimanje

## Napomena
Dataset ne ide u git (ignorisan u `.gitignore`). Samo ovaj README fajl se čuva u repozitorijumu.

## Sledeći koraci
- [ ] preuzeti ShanghaiTech dataset
- [ ] proveriti .mat anotacije (head annotations)
- [ ] napisati skriptu za generisanje density mapa
- [ ] napraviti train/val/test split
