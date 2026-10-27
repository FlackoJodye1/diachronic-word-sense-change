# DWUG DE – Bestandsaufnahme (Schritt 0)

Stand 2026-10-27. Nur Inventur, keine Evaluation gerechnet.
Anlass: Reviewer verlangen eine Evaluation auf DWUG DE / SemEval-2020 Task 1 (Deutsch).
Geplant: (1) ARI unserer Zuordnungen gegen die DWUG-Cluster, (2) Spearman zwischen
unserer JSD (2 Perioden) und `change_graded`; später XL-LEXEME/XL-DURel als Baselines.

## Kurzfassung

- **Geladen:** DWUG DE **3.0.0** (2024-11-01, aktuelle Version) und DWUG DE Sense
  **1.0.1** (2024-11-01). Lizenz: **CC BY-ND 4.0**.
- **50 Zielwörter, 9.125 Usages.** Periode 1 = 1800–1899 (DTA), Periode 2 = 1946–1990.
  Periode 2 hat immer 100 Usages, Periode 1 hat 39–100 (Median 100).
- **GermaNet deckt alle 50 Wörter ab.** Zwei davon nur über die alte Schreibung
  (`Engpaß`, `Mißklang`), und die liest unser Parser nicht.
- **Problem:** **24 der 50 Wörter haben nur ein einziges Synset** in GermaNet. Mit unserer
  Methode kommt für sie zwangsläufig JSD = 0 und eine einzige Klasse heraus. Darunter sind
  Wörter mit starkem Gold-Wandel: abgebrüht 0,83, Armenhaus 0,72, artikulieren 0,71,
  Knotenpunkt 0,70. Der fehlende Sinn ist meist der historische, den GermaNet nicht kennt.
- **Überschneidung mit unseren 502 Lemmas: nur 3 Wörter** (Fuß, Mut, packen). Für die übrigen
  23 Wörter mit ≥ 2 Synsets fehlen LLM-Texte, Cluster und manuelle Prüfung.
- **Unseren DWDS-Satzfilter auf DWUG nicht anwenden.** Er würde 47 % der Usages aus
  Periode 2 und 65 % aus Periode 1 verwerfen, vor allem wegen der Oberflächenform-Prüfung.
  ARI und `change_graded` beziehen sich auf die vollständigen Usage-Mengen.

## 1. Datenquelle und Lizenz

| Datensatz | Version | Zenodo | Datei | SHA-256 |
|---|---|---|---|---|
| DWUG DE | 3.0.0 (2024-11-01) | [10.5281/zenodo.14028509](https://zenodo.org/records/14028509) | `dwug_de.zip` (19,4 MB) | `541213c2…94779` |
| DWUG DE Sense | 1.0.1 (2024-11-01) | [10.5281/zenodo.14041715](https://zenodo.org/records/14041715) | `dwug_de_sense.zip` (1,4 MB) | `9a12b425…4254b` |

Ältere Versionen gibt es ebenfalls (DWUG DE 1.0.0–2.3.0, DWUG DE Sense 1.0.0) sowie
„DWUG DE Resampled“ 1.0.0. Laut Zenodo-Suche ist 3.0.0 die neueste Version.

- **Ablage:** `data/external/dwug_de/{dwug_de,dwug_de_sense}/` mit den Zips daneben.
  Durch `.gitignore` Z. 89 (`data/external/`) ist der Ordner ignoriert, das habe ich mit
  `git check-ignore` geprüft.
- **Lizenz CC BY-ND 4.0:** Zitieren ist Pflicht. **NoDerivatives** heißt: Wir dürfen keine
  veränderten Fassungen des Datensatzes weitergeben, also z. B. kein DWUG mit unseren Labels
  ins Repo oder in ein Release. Eigene Vorhersagen und Scores getrennt zu veröffentlichen
  ist unproblematisch.
- **Zitieren:** Schlechtweg et al. 2021 (EMNLP, „DWUG“) und Schlechtweg et al. 2024 (EMNLP,
  „More DWUGs“). Für DWUG DE Sense außerdem Schlechtweg et al. 2024 (LRE, „Sense Through
  Time“).

### DWUG DE 3.0.0 vs. SemEval-2020 Task 1

Die Reviewer nennen beides. Der SemEval-Gold-Standard ist eine **ältere Fassung**. DWUG DE
3.0.0 hat eine weitere Annotationsrunde, neu berechnete Clusterings und korrigierte Usages,
deshalb weichen die `change_graded`-Werte von SemEval ab. Die meisten veröffentlichten
Baselines (u. a. XL-LEXEME) berichten auf SemEval. *Noch zu prüfen:* die SemEval-Zielwortliste
(meines Wissens 48 Wörter) und die Frage, welche DWUG-Version dem SemEval-Gold entspricht.
SemEval-Daten habe ich nicht geladen.

## 2. Struktur

```
dwug_de/
  data/<lemma>/uses.csv          # Usages (TSV)
  data/<lemma>/judgments.csv     # paarweise DURel-Urteile 1–4 (TSV)
  clusters/opt/<lemma>.csv       # identifier \t cluster
  graphs/opt/<lemma>             # WUG (networkx-Pickle)
  stats/opt/stats_groupings.csv  # Gold-Change pro Lemma  ← hier steht change_graded
  stats/opt/stats.csv, stats/stats_agreement.csv
dwug_de_sense/
  data/<lemma>/{uses,senses,judgments_senses}.csv
  labels/<lemma>/{maj_2,maj_3}/{labels_senses,labels_proximity}.csv
  stats/{maj_2,maj_3}/stats_groupings.csv
```

**`uses.csv`** hat folgende Spalten: `lemma, pos, date, grouping (1|2), identifier, context,
indexes_target_token (Zeichen-Offset "a:b"), indexes_target_sentence, context_tokenized,
context_lemmatized, context_pos, context_normalized`. `context` ist immer genau ein Satz,
denn `indexes_target_sentence` deckt den ganzen Kontext ab. Wortarten: 6.244 NN, 2.540 Verbformen,
335 Adjektivformen, 5 NE.

**Cluster-Labels** (`clusters/opt`) sind ganze Zahlen ab 0, pro Lemma beliebig vergeben,
und **−1 steht für Rauschen bzw. ausgeschlossene Knoten** (251 von 9.125 Usages). Die Cluster
sind aus dem Usage-Graphen induziert, es sind keine Definitionen dahinter. Verteilung über alle
Lemmas: Cluster 0 = 6.898, 1 = 1.295, 2 = 344, danach ein langer Schwanz bis über 10.

**Gold-Change** steht in `stats/opt/stats_groupings.csv`, eine Zeile pro Lemma,
`grouping = 1_2`:
- `change_graded` ist die JSD der Cluster-Wahrscheinlichkeitsverteilungen zwischen Periode 1
  und 2. Spannweite 0,00–0,93, fünf Wörter liegen genau bei 0.
- `change_binary` (20 von 50 = 1) sowie `_gain` und `_loss`, Schwellen `k1/n1/k2/n2`.
- `EARLIER`, `LATER`, `COMPARE` sind mittlere paarweise Urteile (alternatives Graded-Maß).
- `cluster_prob_dist1/2` sind die Gold-Verteilungen selbst. Darauf ließe sich zusätzlich
  prüfen, ob wir die einzelnen Verteilungen treffen, nicht nur den Change-Score.

**DWUG DE Sense 1.0.1**: 24 der 50 Wörter, je 50 Usages (25 pro Periode, 1.200 insgesamt).
`senses.csv` enthält kurze Definitionen (3–8 pro Wort, `sense0` = „andere“). Die
`labels_senses.csv` mit maj_2 enthalten 184 Usages mit „andere“ und 83 mit `-1`, wo keine
Mehrheit zustande kam. Die Change-Scores dort sind aus den Sense-Labels abgeleitet, nicht
direkt annotiert. Das kommt unserem Aufbau (Sinn-Inventar → Zuordnung) **näher** als die
Graph-Cluster und wäre ein naheliegendes zweites ARI-Ziel.

## 3. Abdeckung

„GN-Synsets“ zählt Synsets über `orthForm`/`orthVar`, bei „(alt)“ nur über `oldOrthForm`.
„in 502“ heißt: steht in `sense_clusters_reviewed.json` und in `senses_combined.jsonl`.
„Filter ok“ zählt die Usages pro Periode, die unseren DWDS-Filter bestehen (siehe §5).
Rohdaten: `analysis/dwug/coverage.csv` (gitignored).

| Lemma | GN-Synsets | in 502 | DWUG Sense | Usages P1 / P2 | change_graded | binary | Filter ok P1 / P2 |
|---|---|---|---|---|---|---|---|
| abbauen | 5 | – | ja | 61 / 100 | 0.787 | 1 | 6 / 5 |
| Fuß | 5 | ja | ja | 100 / 100 | 0.606 | 1 | 33 / 44 |
| Dynamik | 5 | – | ja | 61 / 100 | 0.605 | 0 | 46 / 96 |
| verbauen | 5 | – | – | 68 / 100 | 0.577 | 1 | 7 / 36 |
| ausspannen | 4 | – | ja | 100 / 100 | 0.730 | 1 | 11 / 50 |
| abdecken | 4 | – | ja | 52 / 100 | 0.627 | 1 | 5 / 12 |
| packen | 4 | ja | ja | 100 / 100 | 0.480 | 1 | 19 / 19 |
| Schmiere | 4 | – | ja | 100 / 100 | 0.415 | 1 | 40 / 78 |
| Behandlung | 4 | – | – | 100 / 100 | 0.338 | 1 | 77 / 97 |
| Spielball | 4 | – | ja | 59 / 100 | 0.271 | 0 | 41 / 93 |
| Titel | 3 | – | ja | 100 / 100 | 0.360 | 0 | 83 / 88 |
| Manschette | 3 | – | ja | 78 / 100 | 0.359 | 1 | 7 / 18 |
| überspannen | 3 | – | ja | 100 / 100 | 0.357 | 1 | 19 / 25 |
| Entscheidung | 3 | – | – | 100 / 100 | 0.287 | 0 | 78 / 69 |
| Seminar | 3 | – | ja | 100 / 100 | 0.265 | 0 | 48 / 52 |
| Mut | 3 | ja | – | 100 / 100 | 0.211 | 1 | 4 / 91 |
| Engpaß | 2 (alt) | – | ja | 82 / 100 | 0.927 | 1 | 29 / 51 |
| Ohrwurm | 2 | – | ja | 39 / 100 | 0.887 | 1 | 20 / 39 |
| Abgesang | 2 | – | ja | 100 / 100 | 0.800 | 1 | 55 / 95 |
| Eintagsfliege | 2 | – | ja | 59 / 100 | 0.691 | 0 | 21 / 73 |
| Rezeption | 2 | – | ja | 100 / 100 | 0.513 | 1 | 12 / 98 |
| zersetzen | 2 | – | ja | 100 / 100 | 0.502 | 0 | 12 / 49 |
| vorliegen | 2 | – | – | 100 / 100 | 0.175 | 0 | 23 / 44 |
| vergönnen | 2 | – | – | 100 / 100 | 0.124 | 0 | 15 / 1 |
| vorweisen | 2 | – | – | 68 / 100 | 0.112 | 0 | 17 / 52 |
| Frechheit | 2 | – | – | 100 / 100 | 0.100 | 0 | 78 / 89 |
| abgebrüht | 1 | – | ja | 64 / 100 | 0.835 | 0 | 38 / 33 |
| Armenhaus | 1 | – | ja | 100 / 100 | 0.716 | 1 | 25 / 88 |
| artikulieren | 1 | – | ja | 59 / 100 | 0.709 | 1 | 0 / 36 |
| Knotenpunkt | 1 | – | ja | 100 / 100 | 0.700 | 0 | 19 / 56 |
| Mißklang | 1 (alt) | – | ja | 54 / 100 | 0.473 | 1 | 27 / 88 |
| Sensation | 1 | – | ja | 100 / 100 | 0.468 | 1 | 63 / 74 |
| Festspiel | 1 | – | – | 50 / 100 | 0.241 | 0 | 13 / 1 |
| voranstellen | 1 | – | – | 100 / 100 | 0.226 | 0 | 12 / 7 |
| aufrechterhalten | 1 | – | – | 40 / 100 | 0.188 | 0 | 22 / 52 |
| beimischen | 1 | – | – | 100 / 100 | 0.182 | 0 | 11 / 11 |
| Tragfähigkeit | 1 | – | – | 82 / 100 | 0.178 | 1 | 71 / 94 |
| Naturschönheit | 1 | – | – | 98 / 100 | 0.159 | 0 | 48 / 16 |
| Gesichtsausdruck | 1 | – | – | 100 / 100 | 0.151 | 0 | 86 / 96 |
| Tier | 1 | – | – | 100 / 100 | 0.149 | 0 | 2 / 22 |
| Lyzeum | 1 | – | – | 100 / 100 | 0.144 | 0 | 1 / 63 |
| Mulatte | 1 | – | – | 100 / 100 | 0.143 | 0 | 10 / 44 |
| Truppenteil | 1 | – | – | 100 / 100 | 0.142 | 0 | 3 / 23 |
| Ausnahmegesetz | 1 | – | – | 59 / 100 | 0.093 | 0 | 7 / 26 |
| Einreichung | 1 | – | – | 60 / 100 | 0.072 | 0 | 45 / 93 |
| Ackergerät | 1 | – | – | 53 / 100 | 0.000 | 0 | 6 / 22 |
| Kubikmeter | 1 | – | – | 100 / 100 | 0.000 | 0 | 28 / 86 |
| Pachtzins | 1 | – | – | 39 / 100 | 0.000 | 0 | 23 / 61 |
| Unentschlossenheit | 1 | – | – | 76 / 100 | 0.000 | 0 | 63 / 98 |
| weitgreifend | 1 | – | – | 64 / 100 | 0.000 | 0 | 2 / 4 |

**Zusammenfassung:**

| | n | mittl. change_graded | change_binary = 1 |
|---|---|---|---|
| ≥ 2 GN-Synsets | 26 | 0,466 | 15 |
| 1 GN-Synset | 24 | 0,249 | 5 |

- Alle 50 Wörter sind in GermaNet V19.0 vorhanden. 2 davon findet man nur über `oldOrthForm`,
  das unser `XMLParser` (`src/data/germanet/parsing/xml_parser.py:133–141`, liest nur
  `orthForm` und `orthVar`) nicht ausliest.
- Davon sind **3** schon in unseren 502 Lemmas: Fuß (4 Coarse-Cluster), Mut (2), packen (3).
- **23** Wörter mit ≥ 2 Synsets bräuchten die komplette Vorverarbeitung.
- **24** Wörter mit 1 Synset lassen sich mit unserer Methode nicht sinnvoll auswerten. Unsere
  Pipeline nimmt grundsätzlich nur polyseme Lemmas auf (≥ 2 Senses).
- Die Ein-Synset-Wörter sind nicht zufällig verteilt: Sie haben im Mittel weniger Wandel, aber
  6 von ihnen liegen über 0,45. Lässt man sie weg, beschönigt das die Spearman-Korrelation.
  Nimmt man sie mit vorhergesagtem 0 auf, drückt das sie. Beides muss im Paper offen gesagt
  werden.
- 18 der 26 Wörter mit ≥ 2 Synsets haben zusätzlich DWUG-Sense-Labels.

## 4. Was es für die übrigen Wörter bräuchte (grobe Schätzung)

Für die **23 Wörter** mit ≥ 2 Synsets, die nicht unter den 502 sind (≈ 65 Synsets):

| Schritt | Werkzeug | Aufwand |
|---|---|---|
| Alte Schreibungen (`Engpaß` → `Engpass`, `Mißklang` → `Missklang`) | entweder Alias-Mapping für die DWUG-Lemmas oder `oldOrthForm` im Parser ergänzen | < 1 h |
| LLM-Anreicherung (`core_gloss`, `llm_definition`, 5 Beispiele) | `scripts/data/generate_expanded_senses.py` (wie bei den 413) | < 1 h Laufzeit, Kosten vernachlässigbar |
| Clustering t = 0,10 | `scripts/clustering/compute_sense_clusters.py --merge-into …` | Minuten |
| Manuelle Cluster-Prüfung | `review_app.py` | ca. 1–2 h für 23 Lemmas (Erfahrung: 136 Lemmas in einem Annotationsblock) |
| DWUG → unser Kontextformat | neuer Adapter: Satz + `indexes_target_token`, **ohne** Satzfilter | 2–3 h |
| WSD-Vorhersage für alle Usages | `predict_dwds_senses.py` (weighted) | einige Stunden lokal, ~9.100 Usages; für die 26 auswertbaren Wörter ~4.700 |
| JSD über 2 Perioden + ARI | vorhandene Metrik-Skripte / `sklearn.adjusted_rand_score` | 1–2 h |

**Zusammen etwa 1,5–2 Arbeitstage** bis zu ersten Zahlen, ohne die XL-LEXEME-Baselines.

Für die **24 Ein-Synset-Wörter** gäbe es drei Wege, die *vor* dem Rechnen festgelegt werden
müssen:
- (a) als vorhergesagtes JSD = 0 aufnehmen, ehrlich, aber pessimistisch
- (b) ausschließen und n = 26 berichten, sauber begrenzt, aber selektiv
- (c) für die 18 bzw. 24 DWUG-Sense-Wörter das DWUG-Sense-Inventar statt GermaNet verwenden.
  Das trennt „Inventar fehlt“ von „Zuordnung falsch“, kostet zusätzlich LLM-Anreicherung der
  DWUG-Definitionen und ca. 0,5 Tag.

Wahrscheinlich ist es am sinnvollsten, (a) und (b) beide zu berichten und (c) als Diagnose
anzuhängen. Das entscheidest du.

## 5. Stichprobe: passt unsere Satz-Vorverarbeitung?

Ich habe unseren DWDS-Filter (`src/data/dwds/fetcher.py:apply_filters` mit `clean_context`,
`count_words` und `target_matches_token`) mit 6–60 Tokens auf alle 9.125 Usages angewendet.
Rohdaten: `analysis/dwug/filter_check.csv` (gitignored).

| | P1 (1800–1899) | P2 (1946–1990) |
|---|---|---|
| Tokens/Satz, Median (IQR) | 33 (22–51) | 23 (16–32) |
| Länge 6–60 ok | 83,7 % | 97,2 % |
| Zielwort genau einmal als Grundform (`target_matches_token`) | 40,8 % | 54,3 % |
| beides ok | **34,7 %** | **53,2 %** |

**Was ich daraus ablese:**

1. **Die Länge ist ein kleineres Problem.** DTA-Sätze sind länger: 16 % liegen über 60
   Tokens, das Maximum bei 1.643. Für das Modell reicht es, lange Sätze um das Zielwort herum
   zu kürzen (Fenster), statt sie zu verwerfen.
2. **Die Oberflächenform-Prüfung ist das eigentliche Problem.** `target_matches_token` sucht
   die Grundform wörtlich. Damit fallen alle flektierten Formen weg (*Abgesangs*, *packte*,
   *Mulatten*, *vergönnt*). Im DWDS-Kern ist das ein (bekannter?) Auswahleffekt. Bei DWUG ist
   es unnötig, weil die Zielposition über `indexes_target_token` gegeben ist. **Für die
   Evaluation darf nicht gefiltert werden**, sonst stimmen die ARI-Mengen nicht mehr mit dem
   Gold überein und `change_graded` gilt für eine andere Stichprobe. Bei einigen Wörtern
   bliebe fast nichts übrig (abbauen 6/5, artikulieren 0/36, weitgreifend 2/4).
   **Nebenbefund:** Für unsere eigene Diachronie-Pipeline heißt das, dass die DWDS-Kontexte
   systematisch auf Grundformen beschränkt sind. Das sollte man im Paper zumindest erwähnen.
3. **Die Orthographie von P1 ist historisch, aber gut lesbar.** Stichproben (1814–1882)
   zeigen *Gemüths*, *Jrrungen*, *mußten*, *nothwendig*. Langes ſ kommt in `context` nicht vor
   (0 %), es ist bereits aufgelöst. `context_normalized` liefert für P1 eine modernisierte
   Fassung (*Gemüts*, *Irrungen*, *notwendig*), für P2 ist die Spalte leer. Unser WSD-Modell
   ist nur auf 1901–1999 validiert (Median 1932). **19.-Jh.-Text liegt außerhalb dieser
   Validierung.** Zu entscheiden ist, ob P1 als `context` oder als `context_normalized`
   eingegeben wird. Am saubersten wäre es, P1 mit beiden Varianten zu rechnen und die
   Differenz als Robustheitscheck zu berichten.
4. **Die Zeitachse passt nicht zu unserer.** Unser Korpus deckt 1900–1999 ab, DWUG 1800–1899
   gegen 1946–1990. Wir messen also auf einem anderen Zeitfenster als in der Haupt-Analyse.
   Für den Methodentest ist das in Ordnung, die Werte sind aber nicht direkt mit unseren
   Kern-JSDs vergleichbar.

## Entscheidungen (27.10.2026)

1. **Wortmenge.** Die Hauptzahl gilt über alle 50 Wörter. Die 24 Wörter mit nur einer
   GermaNet-Bedeutung gehen mit JSD = 0 und einer einzigen Klasse ein. Zusätzlich wird nur
   über die 26 Wörter mit ≥ 2 Bedeutungen ausgewertet.
2. **Gold.** Hauptreferenz ist SemEval-2020 Task 1 (Deutsch), zusätzlich DWUG DE 3.0.0
   (`change_graded`). Herkunft und Abdeckung des SemEval-Golds: siehe unten.
3. **ARI** wird gegen zwei Ziele berechnet: die DWUG-Graph-Cluster (`clusters/opt`, Usages mit
   −1 ausgeschlossen) und die DWUG-DE-Sense-Labels.
4. **Periode 1.** Hauptanalyse auf dem Originaltext (`context`), `context_normalized` als
   Robustheitscheck.
5. **Satzvorverarbeitung.** Kein Grundform-Filter. Die Zielwort-Position wird aus DWUG
   übernommen (`indexes_target_token`), lange Sätze werden um das Zielwort herum gekürzt.

### SemEval-2020-Gold: Herkunft und Abdeckung

*Noch offen.* Die offizielle Quelle ist das Archiv `semeval2020_ulscd_ger.zip` der Organisatoren
(IMS Stuttgart, `www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/`), das Korpora, `targets.txt`
und `truth/{binary,graded}.txt` enthält. Der Download (~ 400+ MB, die Gold-Dateien stehen am
Ende des Archivs) brach am 27.10. zweimal ab und läuft mit Fortsetzung weiter. Der Abgleich
der Zielwörter mit DWUG DE 3.0.0 wird hier nachgetragen.

### Vorbereitung der 23 neuen Wörter (Stand 27.10.2026)

Die automatischen Cluster liegen bereit; die **manuelle Prüfung steht noch aus** (Paul).

**Wörter.** 23 DWUG-Wörter mit ≥ 2 GermaNet-Bedeutungen, die nicht unter den 502 sind, mit
70 Synsets: Abgesang, Behandlung, Dynamik, Eintagsfliege, Engpass (DWUG: *Engpaß*),
Entscheidung, Frechheit, Manschette, Ohrwurm, Rezeption, Schmiere, Seminar, Spielball, Titel,
abbauen, abdecken, ausspannen, verbauen, vergönnen, vorliegen, vorweisen, zersetzen,
überspannen. Fuß, Mut und packen haben schon geprüfte Cluster aus den 502.

**Alte Schreibungen.** `XMLParser` liest jetzt zusätzlich `oldOrthForm`/`oldOrthVar` (neues
Feld `old_orth_forms`). Darüber wird *Engpaß* → *Engpass* und *Mißklang* → *Missklang*
aufgelöst. Missklang hat nur eine Bedeutung und gehört damit zur JSD-=-0-Gruppe.

**Gleiche Einstellungen wie im Paper:**
- **Basistexte:** Das Skript, das `senses_base.csv` erzeugt hat, war nie im Git. Das Template
  ist in `src/data/germanet/transforms/baseline_text_builder.py` rekonstruiert und
  reproduziert alle 2.022 Zeilen von `senses_base.csv` exakt (Text, Reihenfolge,
  `sense_frequency`, `has_examples`). Prüfbar mit `--verify-baseline`; die Prüfung läuft
  auch bei jedem normalen Aufruf vorab.
- **Definition und 5 Beispiele:** dieselben Prompts (`config/prompts/definition.txt`,
  `examples_multiple.txt`) und derselbe Parser wie in `enrich_senses_with_llm.py`. Prompts und
  Code sind seit der Generierung von `senses_full.jsonl` (08.02.2026) unverändert.
- **Core-Gloss:** derselbe Prompt mit den anderen Bedeutungen desselben Worts als Kontext wie in
  `generate_expanded_senses.py` (`build_sibling_map`, `clean_gloss` werden direkt importiert).
- **Modell:** gpt-4o-mini, Temperatur 0,7. Die Texte sind wegen der Temperatur nicht
  bit-identisch reproduzierbar; das gilt für die Paper-Texte genauso.
- **Clustering:** `compute_sense_clusters.py` mit τ = 0,10, `min_clusters` = 2, E5-large
  auf `core_gloss`. Ergebnis: im Mittel 3,0 → 2,4 Cluster, bei 8 von 23 Wörtern hat die
  Mindestaufteilung gegriffen (in der App mit „C“ markiert).

**Unabhängigkeit der Prüfung.** Aus DWUG wird für die Vorbereitung nur die Liste der Zielwörter
(Ordnernamen) gelesen. Die Review-App zeigt nur GermaNet-Klasse, Core-Gloss und
LLM-Definition, keine DWUG-Sätze, -Cluster oder Gold-Werte.

**Dateien** (alle neu, unter `data/processed/dwug/`, gitignored; bestehende Dateien sind
unverändert):

| Datei | Inhalt |
|---|---|
| `lemma_map.csv` | DWUG-Lemma → GermaNet-Lemma, Anzahl Bedeutungen, Status (`prepare` / `existing_502` / `single_sense`) |
| `senses_base_dwug.csv` | Basistexte im Format von `senses_base.csv` |
| `senses_dwug.jsonl` | angereicherte Bedeutungen im Format von `senses_expanded.jsonl` |
| `sense_clusters_dwug_0.10.json`, `t010/cluster_summary.csv` | automatische Cluster |
| `sense_clusters_dwug_reviewed.json` | Arbeitsdatei der manuellen Prüfung (aus den automatischen Clustern angelegt) |
| `review_lemmas.txt` | die 23 Lemmas für die App |

**Reproduzieren:**

```bash
python scripts/dwug/prepare_dwug_senses.py --yes      # Basistexte + LLM-Anreicherung
python scripts/clustering/compute_sense_clusters.py --threshold 0.10 \
    --senses-file data/processed/dwug/senses_dwug.jsonl \
    --output data/processed/dwug/sense_clusters_dwug_0.10.json \
    --summary-dir data/processed/dwug/t010
```

**Manuelle Prüfung starten:**

```bash
python scripts/clustering/review_app.py \
    --source-file data/processed/dwug/sense_clusters_dwug_0.10.json \
    --reviewed-file data/processed/dwug/sense_clusters_dwug_reviewed.json \
    --senses-file data/processed/dwug/senses_dwug.jsonl \
    --summary-file data/processed/dwug/t010/cluster_summary.csv \
    --lemmas-file data/processed/dwug/review_lemmas.txt
# → http://127.0.0.1:8001
```

Gleiche Aufgabe wie bei den 502 (siehe `scripts/clustering/ANNOTATION.md`): Zuordnung prüfen,
Bedeutungen verschieben oder abspalten, jedes Cluster mit einem kurzen deutschen Label
versehen, speichern.

## Offene Entscheidungen (Stand vor dem 27.10.2026, siehe oben)

1. Umgang mit den 24 Ein-Synset-Wörtern: (a), (b), (c) oder eine Kombination.
2. Gold-Version: nur DWUG DE 3.0.0, oder zusätzlich das SemEval-2020-Gold für die
   Vergleichbarkeit mit veröffentlichten Baselines.
3. ARI-Ziel: Graph-Cluster (`clusters/opt`, alle 50 Wörter), DWUG-Sense-Labels (maj_2,
   24 Wörter) oder beide. Umgang mit −1, „andere“ und Fine- vs. Coarse-Zuordnung.
4. P1-Eingabe: `context` oder `context_normalized`.
