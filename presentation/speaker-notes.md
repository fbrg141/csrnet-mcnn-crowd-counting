# Govorne beleške — finalni Part A eksperimenti

Beleške prate 16 slajdova u `presentation.tex`. Ne koristiti stare pilot brojke.

1. **Naslov.** Predstaviti autore i predmet; ne tvrditi da je rad već predat ili ocenjen.
2. **Problem.** Motivisati regresiju gustine okluzijom. Pitanje je poređenje izvedenih konfiguracija, ne dokaz najbolje moguće arhitekture. Part B nije završen.
3. **Mapa gustine.** Suma mape daje broj. Sigma je fiksna 15, keš v2; ilustracija nije predikcija finalnog checkpoint-a. Adaptivna sigma postoji u kodu ali nije eksperiment ovog izveštaja.
4. **Modeli.** Naglasiti tri MCNN grane naspram pretreniranog VGG frontend-a. Lokalni MCNN koristi po dve konvolucije u grani; nije tačna reprodukcija svih slojeva/protokola originalnog rada. Dilatacija 2 daje efektivni kernel 5×5, ne 7×7. Razlike uključuju normalizaciju i stride, pa ne izolujemo samo arhitekturu.
5. **Protokol.** Objasniti 270/30 leksikografski split i 182 test slike. Broj 300 nije broj slika koje ulaze u trening. MAE i RMSE su greške broja; RMSE nije varijansa među seed-ovima.
6. **Pretraga.** Nabrojati tri ose mreže i objasniti validation-only izbor. Seed 42 se koristi iz pretrage, ne trenira ponovo pri potvrdi. Test ranijih pilota već je viđen, zato ne reći „potpuno netaknut test“.
7. **Mreža.** Svaki pokušaj završio je 50 epoha i bira poslednju: budžet ograničava zaključak. Pobeda weight decay od 0.0135 nije dokaz korisne regularizacije. Ne tvrditi statističku značajnost.
8. **Glavna tabela.** Porediti seed 42 sa seed-om 42. CSRNet 233.4080/376.9634 prema podešenom MCNN-u 316.3083/431.9880. Smanjenje 26.21%/12.74% je opisno. Početni MCNN je undertrained, ne najbolji mogući MCNN.
9. **Grafikon testa.** Velika početna MCNN greška vizuelno dominira; u tabeli su precizne vrednosti. Nema error-bar tvrdnje za CSRNet sa jednim seed-om.
10. **Tri seed-a.** Srednja vrednost nije „novi seed“. SD je uzoračka, imenilac n−1, n=3; nije interval poverenja. CSRNet SD je nepoznata, ne nula.
11. **Varijabilnost.** Narandžaste tačke prikazuju sve seed-ove, uključujući nepovoljne. Nije biran najbolji test seed; prikazana je prethodno fiksirana lista.
12. **Krive.** Originalne JSON istorije zamenjuju stare prepisane nizove. CSRNet najbolja epoha 31, ne stara pilot epoha 34. Oscilacije same ne dokazuju precizan uzrok ili konačnu konvergenciju.
13. **Dokazi.** Razdvojiti verifikaciju izveštajnih hash-eva od odsutnih MCNN težina. CSRNet vreme 1583.2287 s je trening na lokalnom RTX 4070, ne brzina inferencije niti fer međuhardversko poređenje.
14. **Ograničenja.** Pretraga daje MCNN-u dodatni budžet. Nema Part B, CSRNet tri seed-a ili inferencijskog benchmark-a. Razlike od literature su moguća objašnjenja, ne izmereni uzroci.
15. **Zaključak.** Ograničiti tvrdnju na izvedeni protokol. Dodatni seed-ovi i ablacione studije potrebni su za jače tvrdnje; ne obećavati rezultat budućeg rada.
16. **Izvori.** Pokazati finalni izveštaj, protokol, originalne lake dokaze i komandu za ponovnu izradu grafikona. Originalni notebook prilog nije prepisivan ni ponovo izvršavan.

## Pitanja koja treba očekivati

- **Zašto ne objavljeni rezultati iz radova?** Naš budžet i protokol nisu replika radova; uticaj svake razlike nije izolovan.
- **Da li je CSRNet stabilniji?** Ne znamo; imamo jedan seed. MCNN varijabilnost je posebno izmerena.
- **Da li je weight decay pomogao?** Formalno je najmanji validacioni MAE niži za 0.0135; nema ubedljivog dokaza koristi.
- **Može li se sve reprodukovati iz repozitorijuma?** Tabele/grafikoni da, bez treninga. Ponovna inferencija MCNN zahteva originalne težine ili novi trening; dataset nije uključen.
