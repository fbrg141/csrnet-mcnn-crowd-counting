# Izrada prezentacije

Grafikoni imaju jedan izvor: `../reports/figures/`. TeX ih koristi direktno;
ne praviti duplikate u `presentation/figures/`. Za prenos izvora prezentacije
poneti i `reports/figures/` uz očuvanu relativnu strukturu. PDF je samostalan.

Iz korena projekta:

```bash
uv run --locked python scripts/plot_training_curves.py
cd presentation
tectonic --keep-logs presentation.tex
```

Tectonic nije Python zavisnost; instalirati ga zasebno i dodati na PATH.
Prvo pokretanje preuzima TeX resurse. Može se koristiti i potpuna XeLaTeX
instalacija (dva prolaza). Sistemski XeLaTeX na mašini pripreme nije imao
`xelatex.fmt`; PDF je stvarno izgrađen postojećim Tectonic binarnim fajlom:

```
/home/gamzatore/.codex/.tmp/bundled-marketplaces/openai-bundled/plugins/latex/bin/tectonic
```

Finalni PDF ima 16 slajdova. Svi su rasterizovani i vizuelno pregledani;
provereni su dijakritici, ključne metrike i granice tekstualnih blokova.
Finalni TeX log nema overfull/underfull ili missing-character upozorenja.
Fontconfig upozorenja starijeg Tectonic binarnog fajla nisu uticala na prikaz.
Govorne beleške: `speaker-notes.md`. Podaci i ograničenja: `../reports/report.md`.
