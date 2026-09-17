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
# Pokrenuti iz korena repozitorijuma nakon `uv sync --locked`:
uv run --locked python scripts/download_data.py
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

## Napomena
Dataset ne ide u git (ignorisan u `.gitignore`). U repozitorijumu se čuvaju
ovaj README i `.gitkeep` fajlovi koji zadržavaju prazne `raw/` i `processed/`
direktorijume.

## Status i oprez

Pipeline, pregled anotacija i split su završeni; važeći status je u
[PLAN.md](../PLAN.md). Finalni eksperimenti su samo Part A: 270/30 trening/validacija
iz zvaničnih 300 trening slika i 182 test slike. Pregled Part B podataka u
notebook-u nije dokaz završenog treninga na Part B.

`download_data.py` zamenjuje postojeći `data/raw/ShanghaiTech/`; ne pokretati ga
nad jedinom kopijom lokalno izmenjenih podataka. `precompute_density_maps.py --force`
briše ceo izabrani direktorijum keša, ne samo izabrane modele/splitove. Za rutinski
rad koristiti postojeće neizmenjene podatke i v2 keš bez `--force`.
