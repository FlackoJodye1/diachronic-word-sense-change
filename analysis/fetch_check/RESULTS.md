# Fetch-Check: Wie viele Sätze kostet der Grundform-Filter?

Stand: 2026-10-28 (Abruf mit DWDS-API-Key). Anlass: Nebenbefund in `analysis/dwug/INVENTORY.md` §5.
`target_matches_token` (`src/data/dwds/util/context_utils.py`) behält nur Sätze, in denen das
Zielwort **genau einmal** als exakte Grundform vorkommt (Regex, Groß-/Kleinschreibung egal).

## Kurzfassung

| Frage | Ergebnis (20 Stichproben-Lemmas) |
|---|---|
| **b/c**: ohne Grundform-Filter | **1,67×** gesamt (Median 1,52). Nomen 1,47, Verben 2,06, Adjektive 2,55 |
| **d/c**: zusätzlich alte Schreibungen abfragen | **1,67×**, also gleich b/c. Kein Stichproben-Lemma hat eine `oldOrthForm` in GermaNet. Bei den 4 Lemmas mit alter Schreibung zeigt sich: DWDS sucht *Schloß* schon unter *Schloss* mit. Die Sätze gehen im Filter verloren, nicht in der Suche (siehe 3c) |
| **e/c**: alle Treffer ohne limit 5000 | **17,8×** Rohtreffer (Median 6,5). Nach unseren Filtern geschätzt **~5,8×** (Median 1,7). Getrieben von wenigen häufigen Lemmas (*Tag* 78×, *glauben* 45×) |
| Liefert die Suche flektierte Formen? | **Ja.** 43 % der Rohtreffer enthalten die Grundform gar nicht (Nomen 35 %, Verben 60 %, Adjektive 74 %), z. B. *glaubte*, *Tage*, *schwarze* |
| Bindet das limit? | Bei 7 von 20 Lemmas (API: 5.000 Zeilen). Hochgerechnet auf alle 502: **~216 Lemmas (43 %)** |
| Reproduziert c unsere Daten? | Ja. c weicht im Mittel um 3,4 Sätze von `pilot_v3_coarse_stats.csv` ab (max. 26 bei *tief*, Zufallsstichprobe) |
| Rechenzeit Neulauf (nur Einbetten) | heute 3,9 h → ohne Grundform-Filter **~6,5 h** → mit allen Treffern **~22 h** (Obergrenze) |

**Wichtigster Befund:** Der Grundform-Filter kostet vor allem bei Verben, Adjektiven und
ß-Lemmas Daten. Die Suche selbst ist lemmatisiert und findet flektierte Formen und alte
Schreibungen. Der Filter wirft beides wieder weg.

## Zugang (2026-10-28)

Seit Juni 2026 braucht der automatisierte Abruf einen persönlichen API-Key (Mail von Franziska
Nojack, DWDS, 28.10.2026; siehe `documentation/DATA_SOURCES.md`). Unsere v2/v3-Abrufe
(Feb.–Mai 2026) lagen vor dieser Einschränkung.

- **Übergabe:** Eine offizielle Doku gibt es noch nicht. Getestet mit **einer** Abfrage
  (`glauben`, limit 5): Key als Header `Authorization: Bearer <key>` und `X-API-Key: <key>`,
  dazu der User-Agent `WSD-Research-TUWien/1.0 (Kontakt: Paul Schmitt; Zweck: Diachrone WSD,
  Kernkorpus)`. Ergebnis: HTTP 200 mit CSV statt 403. Welcher Header entscheidet, ist nicht
  getrennt geprüft. Der Key steht nie in der URL, im Log oder in Dateien.
- **Lauf:** 56 Anfragen für 24 Lemmas, dazu 2 Nachholanfragen (*Schloß*, siehe unten). Je
  3,5 s Pause, nur eine Anfrage gleichzeitig. Alle HTTP 200, kein 429. Ein Lese-Timeout bei
  *Schloß*: Es war keine Sperre, die Abfrage lief beim zweiten Versuch durch.
  Protokoll: `raw/requests.log`.
- **Gesamttreffer (e):** Der CSV-Export liefert keine Gesamtzahl. Die normale Trefferseite
  (`/r/?q=…&corpus=kern&limit=1`, HTML) zeigt sie: „1–1 von **32 698** Treffern (**40503**
  insgesamt)“. Die Zahl „insgesamt“ ist bei 19 von 20 Lemmas **exakt** die Token-Frequenz in
  `kern.sqlite` (*Tag*: 76.699 statt 76.697). Die lokale Datei stammt also sehr
  wahrscheinlich aus dem DWDS-Kernkorpus (Hinweis zur offenen Herkunftsfrage in
  `DATA_SOURCES.md`). „Treffer“ sind im Median 82 % davon, wohl Sätze statt Tokens. Für e
  nehmen wir „Treffer“, denn das ist die abrufbare Menge (bei Lemmas unter 5.000 gilt
  „Treffer“ = a).

## 3. Abruf a–e (Ergebnis)

Definitionen (Einstellungen wie v3, siehe 1): **a** Zeilen im CSV-Export; **b** nach Länge,
1 pro Dokument und Duplikaten, *ohne* Grundform-Filter; **c** dasselbe *mit* Grundform-Filter
(= heutige Pipeline); **d** wie b, plus Rohzeilen der alten Schreibung; **e** Treffer ohne limit.
b und c entstehen aus *demselben* Abruf.

### 3a. Je Wortart (Summen über die Lemmas)

| Wortart | n | a | b | c | b/c | Median b/c | e | e/c | e_est/c* | ohne Grundform |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Nomen | 15 | 39.873 | 11.253 | 7.647 | **1,47** | 1,28 | 109.293 | 14,3 | 4,8 | 35 % |
| Verben | 3 | 10.168 | 3.206 | 1.560 | **2,06** | 2,41 | 37.866 | 24,3 | 8,3 | 60 % |
| Adjektive | 2 | 10.000 | 2.736 | 1.071 | **2,55** | 2,71 | 35.587 | 33,2 | 9,5 | 74 % |
| **gesamt** | 20 | 60.041 | 17.195 | 10.278 | **1,67** | 1,52 | 182.746 | **17,8** | **5,8** | 43 % |

d/c ist in der Stichprobe gleich b/c (keine alten Schreibungen).
\* e_est = e × b/a: So viele Sätze blieben übrig, wenn man alle Treffer abriefe und die
Filter ohne Grundform-Filter anwendete. Das ist eine **Obergrenze**: Bei mehr Treffern pro
Dokument entfernt „1 pro Dokument“ anteilig mehr. Median e_est/c: 1,7.
Verben und Adjektive sind mit 3 bzw. 2 Lemmas nur grob geschätzt.

### 3b. Je Lemma

| Lemma | Wortart | a | b | c | bisher | e (Treffer) | b/c | e/c | ohne Grundform |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tag | NOUN | 5.000 | 1.908 | 813 | 801 | 63.083 | 2,35 | 77,6 | 64 % |
| glauben | VERB | 5.000 | 1.745 | 725 | 726 | 32.698 | 2,41 | 45,1 | 72 % |
| tief | ADJ | 5.000 | 1.579 | 708 | 734 | 22.610 | 2,23 | 31,9 | 66 % |
| schwarz | ADJ | 5.000 | 1.157 | 363 | 374 | 12.977 | 3,19 | 35,8 | 82 % |
| Minute | NOUN | 5.000 | 1.220 | 442 | 451 | 9.211 | 2,76 | 20,8 | 78 % |
| Jugend | NOUN | 5.000 | 1.421 | 1.397 | 1.401 | 8.721 | 1,02 | 6,2 | 0 % |
| Morgen | NOUN | 5.000 | 1.166 | 1.145 | 1.141 | 8.405 | 1,02 | 7,3 | 0 % |
| Blume | NOUN | 4.531 | 881 | 332 | 332 | 4.531 | 2,65 | 13,6 | 74 % |
| Stuhl | NOUN | 4.519 | 935 | 683 | 683 | 4.519 | 1,37 | 6,6 | 34 % |
| Ecke | NOUN | 4.132 | 1.047 | 899 | 899 | 4.132 | 1,16 | 4,6 | 23 % |
| malen | VERB | 3.619 | 743 | 285 | 284 | 3.619 | 2,61 | 12,7 | 74 % |
| Heft | NOUN | 2.109 | 796 | 620 | 620 | 2.109 | 1,28 | 3,4 | 30 % |
| Block | NOUN | 2.100 | 840 | 502 | 502 | 2.100 | 1,67 | 4,2 | 45 % |
| verfahren | VERB | 1.549 | 718 | 550 | 550 | 1.549 | 1,31 | 2,8 | 34 % |
| Blei | NOUN | 738 | 321 | 313 | 313 | 738 | 1,03 | 2,4 | 2 % |
| Frosch | NOUN | 696 | 207 | 109 | 109 | 696 | 1,90 | 6,4 | 58 % |
| Heizung | NOUN | 367 | 188 | 182 | 182 | 367 | 1,03 | 2,0 | 5 % |
| Schal | NOUN | 290 | 108 | 91 | 91 | 290 | 1,19 | 3,2 | 23 % |
| Juwel | NOUN | 241 | 131 | 37 | 37 | 241 | 3,54 | 6,5 | 81 % |
| Werfen | NOUN | 150 | 84 | 82 | 82 | 150 | 1,02 | 1,8 | 5 % |

„bisher“ = Summe `n_contexts` in `pilot_v3_coarse_stats.csv`. Nomen ohne Pluralendung im
Singular-Stamm (*Jugend*, *Morgen*, *Blei*) verlieren fast nichts. Nomen mit -e/-en/-er-Plural
(*Minute*, *Blume*, *Juwel*, *Tag*) verlieren viel.

### 3c. Alte Schreibungen (d): die 4 Lemmas mit `oldOrthForm` in GermaNet

Nur 4 der 502 Lemmas haben in GermaNet 19.0 eine alte Schreibung, keines davon in der
Stichprobe. Deshalb wurden sie getrennt abgerufen (`old_orth_check.py`).

| Lemma | alt | a | e (neu / alt) | neue Rohzeilen durch alte Abfrage | Rohzeilen mit alter Form | b | c | c, Filter kennt alte Form | d | bisher |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bass | Baß | 823 | 823 / 664 | 0 | 61 % | 173 | 25 | 145 | 173 | 25 |
| Schluss | Schluß | 5.000 | 8.762 / 8.762 | 2.106† | 81 % | 1.898 | 72 | 1.699 | 2.350† | 87 |
| Schloss | Schloß | 5.000 | 5.046 / 5.046 | 45† | 75 % | 1.005 | 41 | 835 | 1.008 | 41 |
| Foto | Photo | 686 | 686 / 303 | 298 | 0 % | 168 | 91 | 91 | 323 | 91 |

† Gleiche Treffermenge: Die zweite Abfrage ist nur eine zweite Zufallsstichprobe daraus. Der
Zuwachs kommt allein vom limit, nicht von der Schreibung.

- **Bass, Schluss, Schloss:** DWDS ordnet *Baß/Schluß/Schloß* demselben Lemma zu. Die Suche
  findet sie also schon, eine Zusatzabfrage bringt nichts. Verloren gehen die Sätze im
  Grundform-Filter, weil „Schluss“ nicht auf „Schluß“ passt. Wenn der Filter die alte Form
  kennt, gibt es 6–24× mehr Sätze. Ohne Grundform-Filter (b) gibt es sie ohnehin. Die
  Zahlen in 4b (41 bzw. 87 Sätze) sind damit erklärt.
- **Foto:** *Photo* ist bei DWDS ein eigenes Lemma. Hier bringt die Zusatzabfrage echte
  neue Sätze: d/b = 1,9.
- **Nicht abgedeckt:** alte Schreibungen in flektierten Formen (*faßte*, *mußte*,
  *vergaß*). GermaNet führt sie nicht. Bei lemmatisierter Suche bringt der Wegfall des
  Grundform-Filters sie automatisch mit.

## 3d. Hochrechnung der Rechenzeit für einen Neulauf

Einbetten mit ~44 s pro 1.000 Sätze (Benchmark, siehe 6). Faktoren aus der Stichprobe,
gepoolt über die Lemmas, angewendet auf 318.611 Sätze:

| Szenario | Faktor | Sätze | Einbetten |
|---|---:|---:|---:|
| heute (c) | 1,00 | 318.611 | ~3,9 h |
| ohne Grundform-Filter (b) | 1,67 | ~533.000 | **~6,5 h** |
| b + alte Schreibungen (d) | ≈ 1,67 | ~533.000 | ~6,5 h (nur *Foto* bringt zusätzlich etwas) |
| alle Treffer, gefiltert (e_est, Obergrenze) | 5,8 | ~1,84 Mio. | ~22 h |
| alle Treffer, ungefiltert (e) | 17,8 | ~5,7 Mio. | ~69 h (nicht sinnvoll) |

Bei Median-Faktoren (1,52 bzw. 1,74) läge der Neulauf darunter. Die gepoolten Werte sind
hier aussagekräftiger, weil häufige Lemmas den Hauptteil der Sätze stellen.

**Abruf:**
- Eine Seite je Lemma (wie v3): 502 Anfragen × (3,5 s Pause + 1–4 s Download) ≈ **~1 h**.
  Dazu kommen die lokalen Filter. Der Duplikat-Filter vergleicht paarweise und braucht bei
  5.000 Zeilen grob 0,5–1 min je Lemma, bei 216 Lemmas mit vollem limit also **~2–3 h**.
- Alle Seiten (für e): geschätzt **~1.330 Anfragen** (Treffer ≈ 0,82 × Kern-Frequenz) ≈
  ~2,5 h Download. Der Duplikat-Filter in seiner heutigen Form (quadratisch) ist dafür zu
  langsam (*Tag*: 63.000 Zeilen). Dafür bräuchte man einen schnelleren Ansatz, z. B. MinHash.

## Früherer Stand: Blocker (2026-10-27, überholt)

Am 27.10.2026 war `/r/` mit ehrlichem User-Agent gesperrt (HTTP 403, „als Bot identifiziert“;
`robots.txt`: `Disallow: /r`). Ich habe nichts umgangen. Die DWDS-Antwort vom 28.10.2026 hat
das gelöst (API-Key, siehe oben). Die Vorschläge für `DWDSClient` bleiben gültig. Punkt 1
(UA) und 2 (Exportweg) kommen jetzt um den Key-Header als Env-Variable `DWDS_API_KEY` dazu:

### Vorschlag: DWDSClient für einen späteren Neulauf (nicht umgesetzt)

0. **API-Key:** aus `DWDS_API_KEY` lesen und nur als Header senden, nie in URL oder Log.
   `fetch_check.py` zeigt, wie das geht.

1. **User-Agent:** Statt `Mozilla/5.0` einen benannten, konfigurierbaren Research-UA
   verwenden, z. B. `WSD-Research-TUWien/1.0 (Kontakt: Paul Schmitt; Zweck: Diachrone WSD)`,
   als Konstruktor-Argument `user_agent` oder über die Env-Variable `DWDS_USER_AGENT`. Keine
   Browser-Kennung als Default.
2. **Exportweg:** Den Endpunkt nicht im Code festschreiben (`BASE_URL` + `/r/`), sondern
   konfigurierbar machen und den Weg nehmen, den DWDS auf Nachfrage freigibt: entweder `/r/`
   mit freigeschaltetem UA oder die dstar-Schnittstelle für `kern`. Bei dstar käme die
   Gesamttrefferzahl gleich mit, was Frage 4 klären würde.
3. **Bot-Sperre erkennen:** Ist `Content-Type` nicht `text/csv` oder kommt HTTP 403, sofort
   mit klarer Meldung abbrechen. Bisher würde `pd.read_csv` die HTML-Seite still als
   kaputtes CSV parsen oder `None` liefern, und das Lemma fiele ohne Warnung als
   `fetch_error` weg.
4. **Rate-Limit:** Den Default von `delay` deutlich anheben (≥ 3–5 s zwischen Requests), bei
   HTTP 429/503 mit exponentiellem Backoff warten und `Retry-After` beachten. Nur ein Request
   gleichzeitig, Abrufe nachts oder am Wochenende.
5. **robots.txt:** Vor dem Lauf mit `urllib.robotparser` prüfen und nur weitermachen, wenn
   der UA erlaubt ist oder eine schriftliche Freigabe vorliegt. Die Freigabe über ein Flag
   wie `--dwds-consent 2026-03-23` dokumentieren.
6. **Protokoll:** Rohantworten (CSV) je Lemma mit Zeitstempel, URL und UA ablegen
   (gitignored), damit ein Neulauf a/b/c ohne erneuten Abruf nachrechnen kann.
7. **Grundform-Filter:** `target_matches_token` hinter eine Option `require_base_form`
   setzen, damit b und c aus *einem* Abruf entstehen. Das Modell bettet ganze Sätze ein und
   braucht keine Zielposition. Offen ist nur, wie Sätze mit mehrfachem Vorkommen behandelt
   werden: ein Satz, ein Label.

## 1. Einstellungen des v3-Abrufs

Quelle: `documentation/research_log.md` (2026-03-23), `scripts/data/fetch_dwds_contexts.py`,
`src/data/dwds/fetcher.py`. Laut `roadmap.md` lief der erweiterte Abruf (412 Lemmas) mit
denselben Einstellungen.

| Parameter | Wert | Herkunft |
|---|---|---|
| Korpus | `kern` | CLI |
| `limit` | 5000 (eine Seite, keine Pagination) | CLI |
| `sort` | `random` | CLI |
| `min_tokens` / `max_tokens` | 6 / 60 (Leerzeichen-Tokens) | CLI |
| `max_per_doc` | 1 (Dokumentschlüssel aus `Bibl`, Autor:Titel) | Default |
| `near_dup_threshold` | 0,7 (Jaccard auf Token-Mengen) | Default |
| `target_n` / `auto_select` | 100 / aus → alle gefilterten Sätze behalten | Default |
| `delay` | 1,0 s | Default |

Reihenfolge in `apply_filters`: Länge → Grundform-Filter → 1 pro Dokument → Duplikate.
Weil der Grundform-Filter **vor** der Dokument-Deduplizierung läuft, kann ein Wegfall des
Filters auch die Auswahl je Dokument ändern. `fetch_check.py` bildet das ab.

## 2. Stichprobe (fertig)

`sample_lemmas.py`, Seed 20261027, proportional geschichtet nach GermaNet-Wortart
(`senses_combined.jsonl`). Gesamtset: 366 Nomen / 84 Verben / 52 Adjektive →
Stichprobe 15 / 3 / 2. Ergebnis in `sample_lemmas.csv`.

| Wortart | Lemmas |
|---|---|
| Nomen | Minute, Heizung, Morgen, Blume, Blei, Stuhl, Heft, Juwel, Schal, Werfen, Ecke, Jugend, Block, Frosch, Tag |
| Verben | glauben, verfahren, malen |
| Adjektive | schwarz, tief |

## 3. (alt) Erwartung vor dem Abruf

Die Ergebnisse stehen oben in §3. Die Erwartung war, dass Verben und Adjektive stärker
flektieren und mehr verlieren. Das hat sich bestätigt.

## 4. Bindet das limit? Ja, bei rund 43 % (offline gemessen, mit API bestätigt)

**Nachtrag 2026-10-28:** Mit API-Daten bindet das limit bei 7 von 20 Stichproben-Lemmas
(a = 5.000). Die Offline-Schätzung unten (10/20) zählt Tokens. *Blume*, *Stuhl* und *Ecke*
liegen knapp über 5.000 Tokens, haben aber weniger als 5.000 Treffer (Treffer ≈ 0,82 ×
Tokens). Mit diesem Faktor auf alle 502 hochgerechnet: **~216 Lemmas (43 %)**. Vorbehalt (2)
unten ist geklärt: Die Suche liefert alle Formen des Lemmas.

Der CSV-Export liefert keine Gesamttrefferzahl. Die Lemma-Frequenzen im Kernkorpus stehen
aber schon in `data/raw/dlexdb/kern.sqlite` (Tabelle `lex`, Summe `f` je Lemma `l`;
121,5 Mio. Tokens). Skript: `limit_check.py`, Ausgabe `limit_check_sample.csv` und
`limit_check_all.csv`.

**Ergebnis:** Bei **245 der 502 Lemmas (49 %)** liegt die Kern-Frequenz über 5.000. Für sie
ist unser Abruf eine Zufallsstichprobe von 5.000 Treffern, nicht der ganze Bestand. Je
Wortart: Nomen 156/366 (43 %), Verben 57/84 (68 %), Adjektive 32/52 (62 %). Median-Frequenz
über alle 502: 4.722.

Stichprobe (Kern-Frequenz → unsere Sätze):

| Lemma | Wortart | Kern-Freq. | unsere Sätze | limit bindet |
|---|---|---:|---:|---|
| Tag | NOUN | 76.697 | 801 | ja |
| glauben | VERB | 40.503 | 726 | ja |
| tief | ADJ | 27.270 | 734 | ja |
| schwarz | ADJ | 17.567 | 374 | ja |
| Minute | NOUN | 11.914 | 451 | ja |
| Morgen | NOUN | 10.797 | 1.141 | ja |
| Jugend | NOUN | 9.613 | 1.401 | ja |
| Stuhl | NOUN | 6.134 | 683 | ja |
| Ecke | NOUN | 5.685 | 899 | ja |
| Blume | NOUN | 5.423 | 332 | ja |
| malen | VERB | 4.286 | 284 | nein |
| Heft | NOUN | 2.597 | 620 | nein |
| Block | NOUN | 2.589 | 502 | nein |
| verfahren | VERB | 1.712 | 550 | nein |
| Blei | NOUN | 904 | 313 | nein |
| Frosch | NOUN | 793 | 109 | nein |
| Schal | NOUN | 430 | 91 | nein |
| Heizung | NOUN | 429 | 182 | nein |
| Juwel | NOUN | 261 | 37 | nein |
| Werfen | NOUN | 168 | 82 | nein |

**Obergrenze für den Gewinn.** Für die 256 Lemmas mit ≤ 5.000 Treffern kann man den
Bestand mit unseren Sätzen vergleichen: Wir behalten **17 %** der Kern-Treffer (Nomen 19 %,
Verben 11 %, Adjektive 13 %; in der Stichprobe 7 % bei *malen* bis 49 % bei *Werfen*).
Der Rest fällt durch *alle* Filter zusammen: Länge, 1 Satz pro Dokument, Duplikate und
Grundform. Wie viel davon auf den Grundform-Filter entfällt, misst erst `fetch_check.py`.
Dass Verben und Adjektive am wenigsten behalten, passt zur stärkeren Flexion.

Vorbehalte: (1) `f` zählt Tokens. Ein Satz mit zwei Vorkommen zählt doppelt. (2) Ob die
DWDS-Suche nach `Tag` alle Formen des Lemmas liefert (*Tage*, *Tages*) oder nur die
Oberflächenform, ist weiter offen. Davon hängt ab, ob die 5.000 Zeilen aus allen
76.697 Treffern gezogen werden. (3) Die Herkunft von `kern.sqlite` ist nicht dokumentiert,
siehe `documentation/DATA_SOURCES.md`.

Folge für einen Neulauf: Bei der Hälfte der Lemmas hilft der Wegfall des Grundform-Filters
nur innerhalb der 5.000er-Stichprobe. Mehr Daten gibt es dort erst mit Pagination
(deterministische Sortierung, da `sort=random` je Seite neu zieht).

### 4b. Nebenbefund: alte Rechtschreibung im Kernkorpus

Ein Lemma (*Schloss*) fehlte in `kern.sqlite`. Ursache: Das Kernkorpus (1900–1999) schreibt
fast durchgehend in alter Rechtschreibung. Wir suchen nach der neuen Schreibung:

| Lemma (unsere Suche) | Kern-Freq. neu | Kern-Freq. alt | unsere Sätze |
|---|---:|---:|---:|
| Schloss | – | 6.073 (*Schloß*) | 41 |
| Schluss | 5 | 10.025 (*Schluß*) | 87 |
| Bass | 159 | 758 (*Baß*) | 25 |

Median aller 502 Lemmas: 542 Sätze. Diese drei sind also nicht von Natur aus selten. Bisher
galt *Bass* im Memory und im research_log als „genuinely sparse“. Das stimmt nicht.
Andere ß/ss-Lemmas der 502 (*fassen*, *messen*, *passen*, *vergessen*, *Wasser* …) sind
in der Grundform nicht betroffen, wohl aber flektierte Formen (*faßte*, *vergaß*, *mußte*).
Bei einem Neulauf sollte die Abfrage beide Schreibungen abdecken.

## 5. Abgleich mit pilot_v3_coarse_stats.csv

Die Summe von `n_contexts` je Lemma in `pilot_v3_coarse_stats.csv` stimmt für alle 20
Stichproben-Lemmas exakt mit der Zeilenzahl in `combined_contexts_v3.jsonl` überein
(Spalten `n_ours_stats`, `n_ours_contexts` in `sample_lemmas.csv`). Diese Zahlen sind der
Vergleichswert für c. Hinweis: Wegen `sort=random` wird ein neuer Abruf c nicht exakt
reproduzieren, sondern nur in der Größenordnung, sofern das limit bindet.

## 6. Rechenzeit

`bench_embedding.py`: 1.000 zufällige Sätze aus `combined_contexts_v3.jsonl`, Präfix
`query: `, `batch_size=64`, `intfloat/multilingual-e5-large`, Apple M1 (16 GB), MPS, Env
`de-wsd`.

| Lauf | Zeit |
|---|---|
| Modell laden | 12,7 s |
| 1.000 Sätze, Lauf 1 | 42,2 s |
| 1.000 Sätze, Lauf 2 | 46,2 s |

**~44 s pro 1.000 Sätze ≈ 23 Sätze/s.** Hochrechnung (nur Kontext-Einbettung; Sense-Embeddings
und Ähnlichkeitsberechnung fallen kaum ins Gewicht):

| Szenario | Sätze | Rechenzeit |
|---|---|---|
| heutiger Umfang | 318.611 | ~3,9 h |
| Faktor 1,5 | ~478.000 | ~5,8 h |
| Faktor 2 | ~637.000 | ~7,8 h |
| Faktor 3 | ~956.000 | ~11,7 h |

Dazu kommt der Abruf selbst, siehe §3d.
Vorbehalt: Der Benchmark lief eventuell parallel zur DWUG-Sitzung, die Werte können daher
etwas zu pessimistisch sein.

## Dateien

- `sample_lemmas.py`: Stichprobe (ausgeführt), Ausgabe `sample_lemmas.csv`
- `bench_embedding.py`: Benchmark (ausgeführt)
- `limit_check.py`: Schritt 4 offline (ausgeführt), Ausgabe `limit_check_sample.csv` / `limit_check_all.csv`
- `fetch_check.py`: Abruf a–e (ausgeführt 2026-10-28), Ausgabe `fetch_counts.csv`, `fetch_meta.json`
- `old_orth_check.py`: d für die 4 Lemmas mit alter Schreibung, Ausgabe `old_orth_counts.csv`
- `summarize_fetch.py`: Faktoren und Hochrechnung aus `fetch_counts.csv`
- `raw/` (gitignored): Roh-CSV und Trefferseite je Abfrage, `requests.log`. Dateinamen sind
  prozentkodiert, weil macOS *Baß.csv* und *Bass.csv* als dieselbe Datei behandelt.

CSV-Ausgaben unter `analysis/` sind laut `.gitignore` nicht versioniert. Die Zahlen stehen hier.
