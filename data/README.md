# Data

Ovaj direktorijum sadrži podatke korišćene u projektu.

## Struktura
- `raw/` — originalni preuzeti podaci
- `processed/` — obrađeni podaci spremni za trening

## Izvor
Dataset:
https://www.kaggle.com/datasets/fmena14/crowd-counting

## Napomena
Veliki fajlovi i dataset se po pravilu ne čuvaju direktno u Git repozitorijumu.
U ovom folderu treba ostaviti samo:
- uputstvo za preuzimanje,
- eventualne skripte za pripremu podataka,
- manje metadata fajlove.

## Sledeći koraci
- preuzeti dataset u `data/raw/`
- proveriti format anotacija
- dokumentovati transformacije potrebne za density map pipeline
