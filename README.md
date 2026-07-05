# Komparativna analiza modela CSRNet i MCNN za prebrojavanje osoba na slikama

## Članovi tima
- Uroš Dimitrijević
- Miloš Kutlešić

## Opis projekta
Ovaj projekat se bavi problemom **prebrojavanja osoba na slikama** (*crowd counting*), sa fokusom na uporednu analizu dve neuronske arhitekture:
- **MCNN (Multi-Column Convolutional Neural Network)**
- **CSRNet (Congested Scene Recognition Network)**

Cilj je da se ispita kako se ove dve arhitekture ponašaju na zadatku procene broja ljudi na fotografijama sa gužvom, koristeći odgovarajući skup podataka za crowd counting.

## Motivacija
Za razliku od klasične detekcije objekata, gde se svaka osoba lokalizuje bounding box-om, crowd counting pristup je pogodniji za scene sa velikom gustinom ljudi, delimičnim preklapanjem i otežanom detekcijom pojedinačnih objekata.
Zbog toga su modeli kao što su MCNN i CSRNet pogodniji od standardnih YOLO pristupa za ovaj problem.

## Skup podataka
Korišćen dataset:
- Kaggle: https://www.kaggle.com/datasets/fmena14/crowd-counting

Napomena: u okviru projekta biće analiziran format anotacija i provereno da li su potrebne dodatne transformacije za generisanje density mapa.

## Cilj rada
Glavni cilj projekta je:
1. analizirati dataset za crowd counting,
2. implementirati ili prilagoditi modele **MCNN** i **CSRNet**,
3. trenirati modele pod istim uslovima,
4. uporediti njihove performanse pomoću standardnih metrika,
5. izvesti zaključak o tome koji model daje bolje rezultate za posmatrani problem.

## Istraživačko pitanje
Koji od modela, **MCNN** ili **CSRNet**, daje bolje rezultate na zadatku prebrojavanja osoba na slikama, posmatrano kroz tačnost procene i stabilnost rada na izabranom skupu podataka?

## Modeli
### MCNN
MCNN koristi više paralelnih konvolucionih grana sa različitim receptive field-ovima kako bi obradio scene sa različitim gustinama ljudi.

### CSRNet
CSRNet koristi dublju arhitekturu sa dilatiranim konvolucijama i poznat je po dobrim rezultatima na crowd counting zadacima, posebno u scenama sa velikom gustinom ljudi.

## Metrike evaluacije
Za poređenje modela koristiće se sledeće metrike:
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**

Po potrebi će dodatno biti razmotreno:
- vreme treniranja,
- vreme inferencije,
- broj parametara modela.

## Plan rada
1. Preuzimanje i analiza dataseta  
2. Priprema pipeline-a za učitavanje podataka i anotacija  
3. Implementacija/prilagođavanje MCNN modela  
4. Implementacija/prilagođavanje CSRNet modela  
5. Trening i evaluacija modela  
6. Uporedna analiza rezultata  
7. Pisanje izveštaja i priprema prezentacije  

## Struktura projekta
```text
data/       - skup podataka i priprema podataka
notebooks/  - istraživačke analize i vizualizacije
src/        - glavni kod projekta
reports/    - beleške, slike, rezultati i skice izveštaja
```

## Pokretanje projekta
### 1. Kloniranje repozitorijuma
```bash
git clone <LINK_DO_REPOA>
cd crowd-counting-csrnet-mcnn
```

### 2. Kreiranje virtuelnog okruženja
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalacija zavisnosti
```bash
pip install -r requirements.txt
```

## Trenutni status
Projekat je u početnoj fazi:
- [x] definisana tema
- [x] izabran skup podataka
- [ ] analiza anotacija i priprema podataka
- [ ] implementacija MCNN
- [ ] implementacija CSRNet
- [ ] trening i evaluacija
- [ ] završni izveštaj

## Očekivani rezultat
Na kraju projekta očekuje se:
- funkcionalna implementacija oba modela,
- eksperimentalno poređenje performansi,
- jasan zaključak o prednostima i manama MCNN i CSRNet pristupa na izabranom datasetu.

## Napomena
Ovaj repozitorijum predstavlja studentski projekat iz mašinskog učenja i služi za eksperimentalnu uporednu analizu modela za crowd counting.
