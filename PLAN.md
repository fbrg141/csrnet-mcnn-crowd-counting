# PLAN — Crowd Counting: status projekta i preostali obuhvat

Ovaj dokument prati stvarno stanje projekta, ne plan implementacije čišćenja.
Važeći zaključci odnose se na **ShanghaiTech Part A**. Izvršen eksperiment,
reprodukcija iz sačuvanih dokaza i nova nezavisna replikacija nisu isto.

## 1. Podaci i pipeline — završeno

- [x] Izabrani ShanghaiTech, MCNN i CSRNet; pregled anotacija i podataka u
  `notebooks/01_dataset_inspection.ipynb` i `notebooks/02_density_maps.ipynb`.
- [x] Loader, skaliranje slike/koordinata, stride 4/8 i ImageNet normalizacija za CSRNet.
- [x] Ispravljena akumulacija preklopljenih anotacija (`+=`, ne prepisivanje),
  regresioni testovi i invalidacija starih mapa preko `CACHE_VERSION=2`.
- [x] Ispravljen adaptivni single-head slučaj; keš i precompute skripta rade.
- [x] Fiksiran split: 270 trening / 30 validacija (leksikografski poslednjih 10%),
  182 zvanične test slike. Finalni treninzi koriste fiksnu sigma=15, ne adaptivne mape.

Eksplorativni notebook-i su sačuvane studijske beleške, ne izvor finalnih metrika.

## 2. Modeli i ponovljivi eksperimenti — završeno u navedenom obuhvatu

- [x] MCNN i CSRNet-B implementacije, zajednički trening/evaluacija, MAE i RMSE.
- [x] Konfiguracije, pune istorije, najbolji checkpoint, optimizer/RNG resume,
  manifesti integriteta, validacioni izbor i eksplicitno odvojena test evaluacija.
- [x] Kanonski Colab notebook: `notebooks/03_hyperparameters.ipynb`;
  uparen allowlist izvorni ZIP sa SHA256, bez oslanjanja na neobjavljene Git izmene.
- [x] MCNN Part A: svih **12 grid pokušaja**, seed 42, po 50 epoha;
  izbor `(lr=1e-5, momentum=.95, weight_decay=1e-4)` samo po validacionom MAE.
- [x] MCNN: **4 potvrde**, početna/izabrana konfiguracija × seed 123/2026,
  po 50 epoha; šest test zapisa uključuje seed 42 iz pretrage, bez ponovnog treninga.
- [x] CSRNet Part A: **seed 42, svih 50 epoha**, najbolja epoha 31;
  lokalni best/last checkpoint-i, istorija, log i test metrike sačuvani.

**Dokazi:** [originalni laki JSON/CSV i granice integriteta](reports/evidence/README.md).
MCNN težine **nisu u dostavljenoj arhivi**. Njihov zabeleženi hash ne znači da su
težine nezavisno proverene; za ponovnu inferenciju/resume potrebni su originalni
Drive checkpoint-i ili novi trening. Lokalni CSRNet checkpoint-i ostaju ignorisani
u Gitu; kurirani mali originali su pod `reports/evidence/csrnet/`.

## 3. Poređenje i završni materijali — izrađeno

- [x] Seed-42 poređenje na istom Part A test skupu: podešeni MCNN MAE/RMSE
  **316.3083/431.9880**, CSRNet **233.4080/376.9634**.
- [x] Odvojena troseed MCNN analiza sa uzoračkom SD (n−1), ne CSRNet SD.
- [x] [Finalni izveštaj](reports/report.md), `reports/results_summary.json` i
  četiri grafikona iz originalnih istorija/metrika, uz proveru hash-eva.
- [x] [Prezentacija](presentation/presentation.pdf): **16 slajdova**, TeX izvor,
  [uputstvo za izradu](presentation/README.md) i govorne beleške.
- [x] Sačuvane istorijske pilot beleške 19/20, jasno odvojene od finalnih rezultata.

Svi MCNN grid minimumi su u epohi 50: konvergencija nije dokazana.
Poređenje ne izoluje arhitekturu (pretreniranje, normalizacija, stride i budžet
pretrage se razlikuju). Pilot test je ranije viđen. Jedan CSRNet seed ne dokazuje
statističku značajnost, opštu superiornost ili veću stabilnost.

## 4. Obavezno pri predaji / prenosu

- [ ] Potvrditi zahteve nastavnika, konačno pregledati materijale i izvršiti predaju.
  Izrađen PDF nije dokaz izvršene predaje ili ocene.
- [ ] Potvrditi trajni backup originalnih MCNN Drive težina i lokalnih CSRNet
  checkpoint-a pre uklanjanja runtime-a ili deljenja odgovornosti za reprodukciju.
  Nepostojanje MCNN težina ne blokira obnovu tabela, ali blokira novu inferenciju.

Za proveru izvornog repozitorijuma: `uv run --locked python -m pytest tests/`.
Za obnovu grafikona: `uv run --locked python scripts/plot_training_curves.py`.
Za PDF i izvorni Colab paket koristiti prateća uputstva; ne pokretati dugi trening
samo radi provere dokumentacije.

## 5. Opciona proširenja — nisu završena niti uslov za postojeći Part A izveštaj

- [ ] CSRNet seed-ovi 123 i 2026 i simetrična višeseed analiza oba modela.
- [ ] Trening/evaluacija na Part B (eksploracija podataka nije rezultat modela).
- [ ] Zasebno merenje inferencije i kontrolisano poređenje vremena na istom hardveru.
- [ ] Duži trening, augmentacija (issue #14), raspored stope učenja, adaptivne mape
  i reprezentativnija validacija, uz novi unapred definisan protokol/izlazni direktorijum.

Ako nastavnik zahteva neko proširenje, prvo ga prebaciti u obavezni obuhvat;
ne prikazivati ga kao završeno bez stvarnih dokaza.
