# Datenquellen und Nutzungsbedingungen

Stand: 2026-10-27. Rohdaten aller Quellen liegen gitignored unter `data/raw/` bzw.
`data/external/`. Sie werden **nicht** ins Repo oder in Releases weitergegeben.

| Quelle | Wofür | Bedingungen | Beleg |
|---|---|---|---|
| **DWDS-Kernkorpus** (`kern`, 1900–1999, BBAW) | Diachrone Kontexte (`data/processed/dwds/*_contexts_v3.jsonl`) | Korpustreffer dürfen exportiert und nachgenutzt werden (siehe Mails unten). Wörterbuchdaten der `/wb/`-Seiten dürfen **nicht** nachgenutzt werden. Unser Code nutzt keine `/wb/`-Daten. Sinninventar kommt aus GermaNet. | Mails 21.11.2025 und 23.03.2026 |
| DWDS-Zeitungskorpora (ZEIT, FAZ, Spiegel) | Derzeit **nicht** genutzt | Nur Zitate in wissenschaftlichen Arbeiten, kein Massenexport | Mail 23.03.2026 |
| **GermaNet 19.0** (Univ. Tübingen, Nov 2024) | Sinninventar (Synsets), Basis für `senses_*.jsonl` und Cluster | Nutzung über die Lizenz von **Prof. Andreas Baumann (Universität Wien)** als Lizenznehmer. Für spätere Releases ist laut GermaNet-Team keine neue Lizenz nötig. Rohdaten nicht weitergeben. **Offen: bei Andreas angefragt**, ob wir abgeleitete Daten veröffentlichen dürfen (Cluster-Zuordnungen mit Synset-IDs, LLM-Texte, Core Glosses). | Mail des GermaNet-Teams (Univ. Tübingen), 24.11.2025; `data/raw/GN_V190/GN_V190-README.txt` |
| **DWUG DE 3.0.0 / DWUG DE Sense 1.0.1** (Zenodo, 2024-11-01) | Evaluation (Reviewer-Anforderung) | **CC BY-ND 4.0**: zitieren (Schlechtweg et al. 2021, 2024; LRE 2024), keine veränderten Fassungen weitergeben. Eigene Vorhersagen und Scores getrennt veröffentlichen ist erlaubt. | `analysis/dwug/INVENTORY.md` §1 |
| **SemEval-2020 Task 1, Deutsch** (IMS Stuttgart; Korpora aus DTA 1800–1899 und BZ/ND 1946–1990) | Gold für Graded Change (geplant): `truth/graded.txt`, `truth/binary.txt`, `targets.txt` (48 Lemmas) | Keine benannte Lizenz. Die README sagt: „frei verfügbar für Lehre, Forschung sowie andere nicht-kommerzielle Zwecke“, also **nur nicht-kommerziell**. Nicht weitergeben: Wir verlinken das Original-Archiv. Zitieren: Schlechtweg et al. 2020 (SemEval@COLING) sowie die Quellen DTA (BBAW 2017), Berliner Zeitung und Neues Deutschland (Staatsbibliothek zu Berlin 2018). | `README.md` im Archiv `data/external/semeval2020_ulscd_ger/semeval2020_ulscd_ger.zip` (825 MB, vollständig) |
| **dlexDB** (Frequenzen des DWDS-Kernkorpus, 20. Jh.) | Frequenzkontrolle (`log_freq_dlexdb`), limit-Prüfung (`analysis/fetch_check/`) | Offizielle Fassung auf Zenodo: **CC BY-SA 4.0** (v2, 2025-04-22, DOI 10.5281/zenodo.15264903). Zitieren mit Lizenzhinweis. ShareAlike gilt für weitergegebene Bearbeitungen der Daten, nicht für unsere Analyseergebnisse. **Offen: Herkunft unserer `kern.sqlite` noch nicht geklärt**, siehe unten. | Zenodo-Record; `data/raw/dlexdb/kern.sqlite` |

## GermaNet: Lizenz über Andreas Baumann

- Laut Mail des GermaNet-Teams (Univ. Tübingen) vom **24.11.2025** ist **Prof. Andreas
  Baumann (Universität Wien)** Lizenznehmer. Paul nutzt GermaNet über diese Lizenz.
- Für spätere GermaNet-Releases ist laut derselben Mail keine neue Lizenz nötig.
- **Offen: bei Andreas angefragt.** Die genauen Lizenzbedingungen, insbesondere ob wir
  abgeleitete Daten veröffentlichen dürfen: Cluster-Zuordnungen mit Synset-IDs
  (`sense_clusters_reviewed*.json`), LLM-Texte (`llm_definition`, `llm_examples`) und
  Core Glosses. Wichtig für ein Benchmark-Release.

## dlexDB: Herkunft der lokalen Datei offen

- Die Zenodo-Fassung (`dlex.zip`, 2,2 GB) enthält laut Beschreibung **TSV-Dateien** und eine
  Markdown-Doku. Projektleitung: Kliegl, Hanneforth, Geyken. Die Zitierangabe auf Zenodo
  ist leer; zitiert wird über die DOI.
- Unsere `data/raw/dlexdb/kern.sqlite` (645 MB, 2026-05-19) ist eine **SQLite-Datei** mit
  den Tabellen `lex/w/p/l` (Spalten `w`, `p`, `l`, `f`; Kommentare wie „transliterated text
  (==unicruft(u))“). Das sieht nach einer dstar-Lexikondatenbank aus, nicht nach dem
  Zenodo-Archiv. Summe `f`: 121,5 Mio. Tokens.
- **Offen, noch nicht geklärt:** Woher stammt die Datei? Kommt sie aus dem Zenodo-Archiv, gilt CC BY-SA 4.0.
  Ist sie von dstar/DWDS, deckt das vermutlich die DWDS-Zustimmung zu „Korpusergebnissen“
  ab, das sollte aber in der Nachfrage beim DWDS mit erwähnt werden.

## Zustimmung des DWDS

Absenderin: Franziska Nojack (BBAW, DWDS). Die Mails liegen bei Paul; Mailadressen werden hier
bewusst nicht aufgeführt.

- **21.11.2025:** Wörterbuchdaten (Bedeutungen, Verwendungsbeispiele der `/wb/`-Seiten)
  dürfen nicht nachgenutzt werden. Aber: „Sie finden allerdings zahlreiche Textbelege zu
  Lemmata (oder Ihrer jeweiligen Suchabfrage) in unseren Korpora. Diese Korpusergebnisse
  sind exportierbar und nachnutzbar.“
- **23.03.2026** (Antwort auf eine Anfrage zum API-Zugang für größere Datenmengen): „Die
  Rechercheergebnisse lassen sich auch im recht großen Stil exportieren. Das können Sie
  entweder auf der Abfrageseite tun (Button ‚Treffer exportieren‘) oder Sie folgen den
  Anweisungen auf unserer API-Seite: https://www.dwds.de/d/api#export“. Einschränkungen
  gelten nur für die Zeitungskorpora (ZEIT, FAZ, Spiegel): dort nur Zitate in
  wissenschaftlichen Arbeiten.

### Offener Punkt: technischer Zugang (Stand 2026-10-27)

Die Erlaubnis ist inhaltlich da. Der automatisierte Abruf ist aber technisch noch nicht
geklärt, Details in `analysis/fetch_check/RESULTS.md`:

- `https://www.dwds.de/d/api` hat derzeit **keinen** Abschnitt `#export`. Für Korpusabfragen
  verweist die Seite nur auf die Plattform dstar (`ddc.dwds.de/dstar/`).
- `www.dwds.de/robots.txt` sperrt `/r` (den Exportweg hinter „Treffer exportieren“) für alle
  Bots. `ddc.dwds.de/robots.txt` sperrt alles.
- Mit ehrlichem User-Agent antwortet `/r/?…&view=csv` mit HTTP 403 („als Bot identifiziert“).
  Die bisherigen v2/v3-Abrufe liefen nur, weil `DWDSClient` sich als Browser ausgibt.

Nachfrage beim DWDS steht aus: Welcher automatisierte Exportweg ist gemeint, und kann ein
benannter Research-User-Agent freigeschaltet werden?
