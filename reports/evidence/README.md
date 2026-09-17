# Originalni laki dokazi

`mcnn/` sadrži neizmenjene JSON/CSV fajlove iz korisničke arhive
`crowd-hyperparameters-results.zip`: protokol, mrežu, izbor, rezultate,
12 pokušaja i četiri potvrde. Izostavljeni su stari grafikon i prazni planovi
za druge modele. Arhiva ne sadrži MCNN težine.

`csrnet/` sadrži neizmenjene config/history/summary/complete/test JSON fajlove
iz lokalnog treninga `csrnet_local_partA_seed42_e50_b4_lr1e-5_v2_20260917T133646`.
Lokalni potpuni manifest (uključujući best i last težine) proveren je pri uvozu;
težine se ne kopiraju u ovaj laki paket i ostaju git-ignored.

`manifest.json` navodi SHA256 originalne MCNN arhive i svakog uključenog
izvornog fajla. Izvorni apsolutni putevi u JSON ostaju kao provenijencija,
ne kao prenosive putanje za izvršavanje. Originalni complete manifesti
referenciraju i težine koje nisu uključene: provera izveštaja nije provera
odsutnih MCNN težina. Test JSON nije obuhvaćen originalnim complete manifestom
(nastao je kasnije), ali je vezan za checkpoint hash i uključen u novi manifest.

Iz korena projekta:

```bash
uv run --locked python scripts/plot_training_curves.py
```

Skripta proverava sadržaj, 50 epoha svake istorije, minimum validacije i
saglasnost šest MCNN test zapisa, pa računa uzoračku SD pomoću `statistics.stdev`.
Generiše četiri grafikona i `reports/results_summary.json` bez GPU-a, težina
ili dataseta. Hash-evi dokazuju integritet sačuvanih fajlova, ne nezavisnu
replikaciju treninga. [Finalna analiza](../report.md) razdvaja seed-42 poređenje
od MCNN troseed statistike. Za inference/resume potrebne su originalne težine.
