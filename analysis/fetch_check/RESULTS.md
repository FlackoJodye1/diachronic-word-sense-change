# Fetch-Check: Wie viele Sätze kostet der Grundform-Filter?

Stand: 2026-10-27. Anlass: Nebenbefund in `analysis/dwug/INVENTORY.md` §5.
`target_matches_token` (`src/data/dwds/util/context_utils.py`) behält nur Sätze, in denen das
Zielwort **genau einmal** als exakte Grundform vorkommt (Regex, Groß-/Kleinschreibung egal).

## Kurzfassung

| Frage | Ergebnis |
|---|---|
| Faktor b/c (gesamt, je Wortart) | **Nicht gemessen.** Der DWDS-Abruf wurde nicht ausgeführt, siehe „Blocker“. |
| Bindet das `limit`? | **Ja, bei 245 von 502 Lemmas (49 %)**, Stichprobe 10/20. Gemessen offline über die Kern-Frequenzen in `kern.sqlite` (siehe 4). |
| Obergrenze für den Gewinn (Lemmas ≤ 5.000 Treffer) | Wir behalten **17 %** aller Kern-Treffer (Nomen 19 %, Verben 11 %, Adjektive 13 %). Darin stecken *alle* Filter, nicht nur der Grundform-Filter. |
| **Neu: Rechtschreibung** | *Schloss*, *Schluss*, *Bass* sind nur scheinbar selten. Das Korpus schreibt *Schloß/Schluß/Baß* (siehe 4b). |
| Rechenzeit Einbetten | **~44 s pro 1.000 Sätze** (M1, MPS). Ein Neulauf mit dem heutigen Umfang (318.611 Sätze) dauert **~3,9 h**, bei Faktor 2 **~7,8 h**. |

## Blocker: DWDS-Zugang (inhaltlich inzwischen geklärt, technisch offen, siehe „DWDS-Zustimmung“)

Der Abruf über `/r/?…&view=csv` ist für automatisierte Zugriffe nicht freigegeben:

- `robots.txt` enthält für `User-agent: *` die Zeile `Disallow: /r`.
- Die TDM-Policy (`/dwds_static/tdm-policy.json`) erlaubt Text-Mining nur mit Einwilligung
  (`"duty": [{"action": "obtainConsent"}]`).
- Mit einem ehrlichen User-Agent liefert der Server die Seite „Auf diese Seite ist kein
  Zugriff gestattet … als Bot identifiziert“ statt CSV.
- `DWDSClient` (`src/data/dwds/client.py`) kommt nur durch, weil er `User-Agent: Mozilla/5.0`
  sendet. Diesen Weg habe ich nicht genutzt. Das betrifft auch die bisherigen v2/v3-Abrufe
  und sollte vor einem Neulauf (und fürs Paper) geklärt werden: Einwilligung bei DWDS einholen
  oder einen offiziell freigegebenen Zugang nutzen (z. B. den Account, der für ZEIT/FAZ
  ohnehin beantragt ist).

Das Skript für Schritt 3–5 liegt fertig in `fetch_check.py`. Es wurde **nicht ausgeführt**.

## DWDS-Zustimmung (Nachtrag 2026-10-27)

Paul hat eine schriftliche Zustimmung von Franziska Nojack (BBAW, DWDS). Übersicht aller Quellen:
`documentation/DATA_SOURCES.md`.

- **Mail 21.11.2025:** Wörterbuchdaten (`/wb/`: Bedeutungen, Verwendungsbeispiele) dürfen
  nicht nachgenutzt werden. Korpusergebnisse sind „exportierbar und nachnutzbar“. Unser Code
  greift nicht auf `/wb/` zu (per grep über `src/` und `scripts/` geprüft).
- **Mail 23.03.2026:** Export „im recht großen Stil“ ist erlaubt, entweder über den Button
  „Treffer exportieren“ oder nach der API-Seite `https://www.dwds.de/d/api#export`.
  Einschränkungen gelten nur für ZEIT, FAZ und Spiegel. Wir nutzen das Kernkorpus.

Damit ist der Inhalt geklärt: Wir dürfen Kernkorpus-Treffer exportieren und nutzen.
**Technisch** ist der automatisierte Weg aber weiterhin gesperrt. Geprüft am 2026-10-27:

| Weg | Ergebnis |
|---|---|
| `https://www.dwds.de/d/api#export` | Der Abschnitt existiert nicht. Die Seite hat keine Stelle zu „Export“ oder `/r/`. Für Korpora verweist sie nur auf dstar. |
| `/r/?q=glauben&view=csv&corpus=kern&limit=5&sort=random` mit User-Agent `WSD-Research-TUWien (Kontakt: Paul Schmitt)` | **HTTP 403**, HTML-Seite „als Bot identifiziert“. Das ist derselbe Export wie hinter dem Button. |
| `www.dwds.de/robots.txt` | `Disallow: /r` für `*` |
| dstar (`ddc.dwds.de/dstar/`) | `robots.txt`: `Disallow /` für alle. Die Übersichtsseite dokumentiert keine Export-Parameter. Nicht abgefragt. |

**Folge:** Ich habe nichts umgangen und `fetch_check.py` **nicht** ausgeführt. Faktor b/c und
„bindet das limit“ bleiben offen, bis DWDS einen Weg für automatisierte Abrufe nennt oder
einen Research-User-Agent freischaltet.

Nützlich für Frage 4, sobald freigegeben: `/api/frequency/?q=<lemma>` liefert laut API-Seite
`hits` (Trefferzahl zum Lemma). Allerdings gilt das über *alle* DWDS-Korpora, nicht nur für
`kern`. Die Gesamtzahl im Kern bekäme man nur über dstar.

### Vorschlag: DWDSClient für einen späteren Neulauf (nicht umgesetzt)

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

## 3. Abruf a/b/c: offen

Nicht ausgeführt (Blocker). Indirekter Hinweis aus `analysis/dwug/INVENTORY.md` §5: Auf den
DWUG-Usages bestehen nur 54 % (1946–1990) bzw. 41 % (19. Jh.) den Grundform-Filter. Für die
DWUG-Wörter entspräche das einem Faktor von ~1,8–2,5. Das ist **nur ein Anhaltspunkt**:
andere Wörter, anderes Korpus, und die DWUG-Daten enthalten sicher flektierte Formen.

Erwartung je Wortart (ungeprüft): Verben und Adjektive flektieren stärker (*glaubte*,
*geglaubt*, *schwarze*, *tiefen*) als Nomen (*Tage*, *Minuten*). Der Faktor dürfte also bei
Verben/Adjektiven höher liegen als bei Nomen.

Ob die API überhaupt flektierte Formen liefert, misst `fetch_check.py` als Anteil der
Rohtreffer ohne exakte Grundform (`share_raw_without_baseform`). Ist der Anteil ≈ 0, sucht
die API nur die Oberflächenform, und der Filter kostet nur die Sätze mit mehrfachem Vorkommen.

Nebenbefund: Der Filter ignoriert Groß-/Kleinschreibung. Bei *Morgen* (Adverb *morgen*) und
*Werfen* (Verb *werfen*) wäre deshalb eine Vermischung denkbar. In den gespeicherten Daten
kommt das nicht vor (Morgen: 1.140× „Morgen“, 1× „MORGEN“; Werfen: 82× „Werfen“).

## 4. Bindet das limit? Ja, bei rund der Hälfte (offline gemessen)

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

Dazu kommt der Abruf selbst: 502 Lemmas × (1 s Pause + Download), grob unter 1 h.
Vorbehalt: Der Benchmark lief eventuell parallel zur DWUG-Sitzung, die Werte können daher
etwas zu pessimistisch sein.

## Dateien

- `sample_lemmas.py` / `sample_lemmas.csv`: Stichprobe (ausgeführt)
- `bench_embedding.py`: Benchmark (ausgeführt)
- `fetch_check.py`: Schritte 3–5 (vorbereitet, **nicht** ausgeführt)
- `limit_check.py` / `limit_check_sample.csv` / `limit_check_all.csv`: Schritt 4 offline (ausgeführt)
