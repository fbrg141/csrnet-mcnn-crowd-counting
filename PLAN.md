# PLAN — Crowd Counting projekat

## Phase 1: Analiza problema i priprema podataka (Active)
**Goal:** razumeti dataset i postaviti osnovu za prve eksperimente.
- [x] potvrđena tema projekta
- [x] izabrani modeli: MCNN i CSRNet
- [x] kreiran README
- [ ] preuzeti dataset lokalno
- [ ] proveriti strukturu fajlova
- [ ] analizirati anotacije
- [ ] utvrditi da li anotacije mogu da se koriste za generisanje density mapa
- [ ] napraviti prvi notebook za vizualizaciju primera
- [ ] definisati train/validation/test split ako nije već dat

**Deliverable:** dokumentovan dataset pipeline i prvi notebook sa analizom podataka.
**Timeline:** 1–2 dana

---

## Phase 2: Implementacija baseline modela — MCNN (⏳ PENDING)
**Goal:** pokrenuti prvi model i proveriti ceo trening/evaluacioni pipeline.
- [ ] implementirati ili prilagoditi MCNN
- [ ] napraviti dataset loader
- [ ] definisati loss funkciju
- [ ] pokrenuti prvi trening
- [ ] sačuvati checkpoint i rezultate
- [ ] izračunati MAE i RMSE

**Deliverable:** prvi funkcionalan baseline sa rezultatima.
**Timeline:** 2–4 dana

---

## Phase 3: Implementacija CSRNet (⏳ PENDING)
**Goal:** dodati drugi model za fer poređenje.
- [ ] implementirati ili prilagoditi CSRNet
- [ ] koristiti isti dataset pipeline i iste metrike
- [ ] pokrenuti trening CSRNet modela
- [ ] sačuvati rezultate evaluacije

**Deliverable:** drugi funkcionalan model sa rezultatima.
**Timeline:** 2–4 dana

---

## Phase 4: Uporedna analiza (⏳ PENDING)
**Goal:** uporediti modele pod istim uslovima.
- [ ] uporediti MAE
- [ ] uporediti RMSE
- [ ] uporediti vreme treninga
- [ ] uporediti vreme inferencije
- [ ] diskutovati prednosti i mane arhitektura
- [ ] napraviti tabelu rezultata

**Deliverable:** tabela rezultata i radni zaključci.
**Timeline:** 1–2 dana

---

## Phase 5: Izveštaj i prezentacija (⏳ PENDING)
**Goal:** pripremiti završnu fakultetsku predaju.
- [ ] napisati uvod i motivaciju
- [ ] opisati dataset
- [ ] opisati MCNN i CSRNet
- [ ] prikazati eksperimente i rezultate
- [ ] napisati zaključak
- [ ] pripremiti prezentaciju

**Deliverable:** završni izveštaj i prezentacija spremni za predaju.
**Timeline:** 2–3 dana

---

## Notes
- Glavni rizik projekta je format anotacija i priprema density mapa.
- Prvo završiti dataset inspection pre ozbiljne implementacije modela.
- Držati iste uslove treniranja za oba modela da poređenje bude fer.
