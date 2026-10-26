# Robustheits-Analysen – Ergebnisse (16.10.2026)

Skripte (lesen nur, schreiben nur in diesen Ordner):
`analysis/robustness/robustness_analyses.py` (§1–4),
`analysis/robustness/binning_sensitivity.py` (§5–6).
Alle Originalzahlen aus dem Paper wurden exakt reproduziert (mean_jsd identisch bis 1e-14).
Null-Baseline: 1000 Permutationen pro Lemma, Seed 42.

## Kurzfassung

| Ergebnis im Paper | Nach Prüfung | Bewertung |
|---|---|---|
| H1: Survey ~ Cluster-Anzahl, r = .509 | .509 (alle), .394 ohne k=1 | hält |
| H2: Survey ~ Entropie, r = .436 | partiell für Cluster-Anzahl: **r = −.005** (p = .93) | Entropie trägt keine eigene Information |
| Innovation: Cluster-Anzahl ~ JSD, r = .774 | ohne k=1: .495; null-korrigiert: **r = .04** (p = .47) | verschwindet bei Dekaden — kommt bei gröberer Zeiteinteilung zurück (§5.1–5.2) und übersteht die Frequenzkontrolle (§5.3) |
| Conformity: Frequenz ~ JSD, r = −.278 / −.392 | null-korrigiert: **r = +.02 bis +.12** | verschwindet, in allen sechs Zeiteinteilungen (§5.2) |
| Survey-Reliabilität | Split-Half .86 [.80, .90] | gut |
| WSD schlechter auf altem Deutsch? | fine 48,4 % vor 1950 vs. 32,3 % ab 1950 (n.s.) | Richtung umgekehrt, n zu klein (§6, explorativ) |

## 1. Zwei Abweichungen zwischen Paper und Code

1. **Ein-Cluster-Lemmas sind enthalten.** 115 von 502 Lemmas haben nur einen Cluster (JSD = 0 und Entropie = 0 per Konstruktion). 109 davon stecken in den Hamilton-Analysen (n = 485), 77 in den Survey-Korrelationen (n = 353). Das Paper (§2.4) sagt, sie würden ausgeschlossen. Sie blähen alle Polysemie-Korrelationen auf.
2. **Conformity nutzt nicht dlexDB.** Die Paper-Zahlen (−.278 / −.392) entstehen mit `log_frequency` aus `baumann_predictors.csv` = log der Anzahl *abgerufener DWDS-Kontexte* (gedeckelt pro Lemma). Mit dlexDB, wie im Paper beschrieben: −.209 / −.331.

## 2. JSD-Null-Baseline

- Erwartete JSD ohne jeden Wandel (Sätze zufällig auf Dekaden verteilt) = **90 %** der beobachteten mittleren JSD (0,077 vs. 0,086).
- Nur 57 von 373 Lemmas (15 %) zeigen Wandel über Zufallsniveau (p < .05; erwartet wären 5 %). Nach FDR-Korrektur: 9.
- Über alle Lemmas gibt es aber systematisch etwas echten Wandel: Excess > 0 bei 68 % der Lemmas, Wilcoxon p = 1e−12.
- Die Null-JSD allein korreliert mit der Cluster-Anzahl genauso stark wie die beobachtete (r = .785 vs. .774) und mit der Kontextanzahl r = −.67. Beide Hamilton-Effekte sind also fast vollständig Sampling-Rauschen.
- *Präsident*: z = 1,36, p = .10, also nicht signifikant über Zufall.
- Stärkste Lemmas (z > 3,5): Druck, Schatz, Rat, Stift, Feuer, Preis, Geschichte, Ordnung, Radio, schön.

| Test (nur k > 1) | roh | null-korrigiert (Excess) | z-Score |
|---|---|---|---|
| Conformity, dlexDB | −.348 | +.023 (p = .66) | +.120 (p = .02) |
| Innovation, Cluster-Anzahl | +.495 | +.036 (p = .49) | +.037 (p = .47) |
| Innovation, Entropie | +.442 | +.119 (p = .02) | +.100 (p = .05) |
| Innovation, Survey | +.203 | +.058 (p = .34) | +.082 (p = .18) |

## 3. H2: partielle Korrelation

- Entropie ~ Cluster-Anzahl: r = .86 (alle), .70 (k > 1).
- Survey ~ Entropie | Cluster-Anzahl: r = −.005 (alle), −.004 (k > 1).
- Survey ~ Evenness (Entropie / log2 k): r = −.009.
- Regression (k > 1): Entropie zusätzlich zu Cluster-Anzahl β = .08, p = .31; R² .114 → .116.

## 4. Survey-Reliabilität

78 Teilnehmende, 544 Lemmas, ca. 24 Ratings pro Lemma. Split-Half (Spearman-Brown, 1000 Splits): **.86** [.80, .90].

## 5. Sensitivitätsanalyse: Zeiteinteilung

Skript: `analysis/robustness/binning_sensitivity.py`. Basis: Dekaden-Counts aus
`pilot_v3_coarse_sense_counts.csv`, zu Perioden aufsummiert. Nur Lemmas mit
n_clusters_coarse > 1 (bei k = 1 ist die JSD per Konstruktion 0). Mindestens 10 Sätze
pro Lemma-Periode; ein Übergang zählt nur, wenn beide Endpunkte den Filter passieren.
Null-Baseline wie in §2: 1000 Permutationen, Periodengrößen fix, Seed 42 — dieselben
Funktionen wie `robustness_analyses.py`, die Einteilung (a) reproduziert `mean_jsd`
exakt (Assertion im Skript).

Vorab festgelegte Einteilungen:

- **(a)** 10 Dekaden 1900–1990 (Referenz, wie eingereicht)
- **(b)** 5 Perioden à 20 Jahre: 1900–19, 1920–39, 1940–59, 1960–79, 1980–99
- **(c)** 2 Perioden: 1900–1949 vs. 1950–1999
- **(c′)** 2 Perioden ohne die 1940er: 1900–1939 vs. 1950–1999 (Tie-Break-Kontrolle)
- **(d)** 4 Perioden: 1900–1929, 1930–1949, 1950–1969, 1970–1999
- **(e)** 3 Perioden: 1900–1929, 1930–1959, 1960–1999

(d) und (e) wurden nach dem ersten Durchlauf ergänzt, aber vor ihrer Berechnung
festgelegt. Sie füllen die Lücke zwischen (b) und (c): bei (b) ist der
Innovation-Effekt noch nicht signifikant, bei (c) klar — die Frage ist, wo der
Übergang liegt und ob er graduell verläuft.

**Abweichung von der Vorab-Festlegung:** Vorgesehen war der Schnitt 1900–1944 vs.
1945–1999. Die aggregierten Counts liegen nur in Dekaden-Auflösung vor, die Dekade
1940 lässt sich also nicht bei 1945 teilen. (c) schlägt sie der frühen Periode zu
(nächstliegender realisierbarer Schnitt, nutzt alle Daten), (c′) lässt sie weg
(saubere Vor-/Nachkriegs-Trennung, kostet 9,9 % der Sätze). Beide werden berichtet;
sie stimmen bei 84 von 92 FDR-signifikanten Lemmas überein, die Zuordnung der 1940er
ist also unerheblich.

`mean_entropy` bleibt bewusst der dekadenbasierte Wert aus `per_lemma_summary.csv` —
er dient hier als fester Lemma-Prädiktor, nicht als Zielgröße, und wird nicht pro
Einteilung neu berechnet.

### 5.1 Rauschanteil und Zahl der signifikanten Lemmas

Zeilen nach Granularität sortiert (viele Bins → wenige Bins).

| Einteilung | Sätze/Lemma-Periode (Median) | JSD beob. | JSD null | JSD Excess | **Null-Anteil** | p_perm < .05 | BH-FDR < .05 | Excess > 0 |
|---|---|---|---|---|---|---|---|---|
| (a) 10 Dekaden | 54 | 0,110 | 0,099 | 0,011 | **90 %** | 57 / 373 | 9 | 68 % |
| (b) 5 Perioden | 102 | 0,092 | 0,074 | 0,018 | **80 %** | 81 / 378 | 28 | 69 % |
| (d) 4 Perioden | 123 | 0,089 | 0,069 | 0,020 | **78 %** | 83 / 377 | 26 | 68 % |
| (e) 3 Perioden | 179 | 0,078 | 0,055 | 0,023 | **71 %** | 104 / 377 | 50 | 65 % |
| (c) 2 Perioden | 253 | 0,091 | 0,047 | 0,045 | **51 %** | 146 / 378 | 92 | 75 % |
| (c′) 2 Per. o. 1940er | 244 | 0,094 | 0,047 | 0,046 | **51 %** | 139 / 376 | 92 | 75 % |

Die Spalte „Sätze/Lemma-Periode" ist der Median über alle gestützten
Lemma-Perioden-Zellen (≥ 10 Sätze) jener Lemmas, die eine auswertbare JSD liefern,
und kommt jetzt direkt aus dem Skript (`median_n_contexts_supported`). Für (b) und
(c) ersetzt das die früher hier eingetragenen 108 bzw. 270; (a) und (c′) sind
unverändert. Die Referenz-Einteilung (a) reproduziert alle Zahlen des ersten
Durchlaufs exakt — die Erweiterung hat an ihr nichts verschoben.

Der Rauschboden fällt wie erwartet mit ~1/√n: viermal mehr Sätze pro Bin halbieren
die Null-JSD (0,099 → 0,047), während die beobachtete JSD fast konstant bleibt
(0,110 → 0,091). Entsprechend steigt der mittlere Excess von 0,011 auf 0,045.
Mit (d) und (e) ist zu sehen, dass das **graduell** verläuft und nicht an einem
Schwellenwert kippt: Null-Anteil 90 → 80 → 78 → 71 → 51 %, Excess monoton
0,011 → 0,018 → 0,020 → 0,023 → 0,045.

**Die Dekaden-Analyse war unterpowert, nicht nur rauschdominiert.** Bei
2 Perioden zeigen 92 von 378 polysemen Lemmas (24 %) nach FDR-Korrektur echten
Wandel, bei Dekaden nur 9 (2 %); (d) liefert 26, (e) 50. Das ist kein freies
Mittagessen: 4 der 9 Dekaden-Treffer (Druck, Ordnung, Preis, Stift) fallen bei (e)
heraus, bei (d) zusätzlich Radio — Wandel, der in einer einzelnen Dekade
konzentriert oder nicht-monoton ist, wird innerhalb einer langen Periode
weggemittelt. Umgekehrt kommen bei (d) 22 und bei (e) 45 Lemmas neu dazu.
Die 50 Treffer von (e) sind fast eine Teilmenge der 92 von (c) (40 von 50 überlappen).

### 5.2 Korrelationen pro Einteilung (Spearman)

jsd_obs = rohe beobachtete JSD, jsd_excess = beobachtet − Null-Mittel,
jsd_z = Excess / Null-SD.

| Test | Prädiktor | Zielgröße | (a) 10 Dek. | (b) 5 Per. | (d) 4 Per. | (e) 3 Per. | (c) 2 Per. | (c′) o. 1940er |
|---|---|---|---|---|---|---|---|---|
| Conformity | log_freq_dlexdb | jsd_obs    | −.348*** | −.326*** | −.348*** | −.197*** | −.184*** | −.148** |
| Conformity | log_freq_dlexdb | jsd_excess | +.021 | +.065 | −.031 | +.124* | +.033 | +.053 |
| Conformity | log_freq_dlexdb | jsd_z      | +.120* | +.187*** | +.102* | +.212*** | +.177*** | +.184*** |
| Innovation | n_clusters_coarse | jsd_obs    | +.495*** | +.438*** | +.421*** | +.426*** | +.365*** | +.380*** |
| Innovation | n_clusters_coarse | jsd_excess | +.043 | +.083 | **+.143**  | **+.166**  | **+.213***  | **+.221*** |
| Innovation | n_clusters_coarse | jsd_z      | +.042 | +.100 | **+.149**  | **+.163**  | **+.235***  | **+.236*** |
| Innovation | mean_entropy | jsd_obs    | +.442*** | +.340*** | +.358*** | +.343*** | +.316*** | +.331*** |
| Innovation | mean_entropy | jsd_excess | +.129* | +.108* | **+.160**  | **+.167**  | **+.205***  | **+.213*** |
| Innovation | mean_entropy | jsd_z      | +.106* | +.124* | **+.165**  | **+.163**  | **+.209***  | **+.218*** |
| Innovation | subjective_polysemy_mean | jsd_obs    | +.203*** | +.185** | +.225*** | +.265*** | +.254*** | +.270*** |
| Innovation | subjective_polysemy_mean | jsd_excess | +.062 | +.097 | +.118 | **+.190**  | **+.214***  | **+.232*** |
| Innovation | subjective_polysemy_mean | jsd_z      | +.082 | +.113 | +.124* | **+.184**  | **+.191**   | **+.220*** |

\* p < .05, \*\* p < .01, \*\*\* p < .001. n = 373–382 (Survey-Zeilen: n = 271–274).
Exakte p-Werte in `binning_sensitivity_correlations.csv`.

**Innovation hält, sobald der Rauschboden niedrig ist.** Die *rohe* Korrelation
schrumpft mit gröberen Bins (+.495 → +.365), die *null-korrigierte* wächst monoton
mit (+.04 → +.08 → +.14 → +.17 → +.21). Signifikant wird sie ab (d), also sobald
der Null-Anteil unter ~80 % fällt. Das gilt für alle drei Polysemie-Maße, auch für
die unabhängig erhobenen Survey-Ratings (dort erst ab (e)). Der Befund aus §2
("verschwindet") war also ein Power-Problem der Dekaden-Einteilung, kein
Null-Befund. Die Größenordnung bleibt aber halbiert: etwa die Hälfte des
eingereichten Effekts (+.37 roh vs. +.21 korrigiert) ist Sampling-Rauschen.

**Conformity hält in keiner Einteilung.** Die rohe negative Korrelation schrumpft
genau so, wie es ein Rauschartefakt tun muss (−.348 → −.184; häufigere Lemmas haben
mehr Sätze, also weniger Rauschen), und der Excess liegt in allen sechs Einteilungen
bei 0 oder sogar leicht positiv ((e): +.124, also mit *falschem* Vorzeichen für
Conformity). (d) ist der einzige Wert mit dem erwarteten Vorzeichen (−.031) und
ist nicht signifikant.

Das positive jsd_z (+.10 bis +.21) ist **nicht** als umgekehrte Conformity zu lesen:
jsd_z skaliert mit der Stichprobengröße, und `jsd_null_sd` korreliert −.96 mit der
Kontextanzahl und −.75 mit log_freq_dlexdb. Häufige Lemmas bekommen bei gleichem
Excess mechanisch ein größeres z. Für Conformity ist daher der Excess die saubere
Größe — und der zeigt nichts. Für Innovation greift dieser Einwand nicht: die
Polysemie-Prädiktoren korrelieren nur +.13 bis +.19 mit der Kontextanzahl, und der
Effekt zeigt sich im Excess, der nicht SD-skaliert ist.

### 5.3 Partielle Korrelationen: ist Innovation ein Frequenz-Artefakt?

Dubossarsky et al. (2017) argumentieren, der Innovation-Effekt sei ein
Frequenz-Artefakt. Zielgröße ist hier durchgehend `jsd_excess` (die
null-korrigierte Größe aus §5.2); Ränge werden auf die Kontrollvariable
residualisiert und dann korreliert, p aus t mit df = n − 3 (dieselbe
`partial_spearman`-Implementierung wie bei H2 in §3).

Zwei Kontrollvariablen: `log_freq_dlexdb` (externe Korpusfrequenz, dlexDB) und
`log_n_contexts` = log der Anzahl Kontexte des Lemmas in der jeweiligen Einteilung
(die Größe, die den Rauschboden tatsächlich setzt).

| Prädiktor | Kontrolle | (a) 10 Dek. | (b) 5 Per. | (d) 4 Per. | (e) 3 Per. | (c) 2 Per. | (c′) o. 1940er |
|---|---|---|---|---|---|---|---|
| n_clusters_coarse | log_freq_dlexdb | +.040 | +.073 | **+.151**  | **+.147**  | **+.211***  | **+.215*** |
| n_clusters_coarse | log_n_contexts  | +.018 | +.044 | +.118* | +.112* | **+.191***  | **+.195*** |
| mean_entropy | log_freq_dlexdb | +.127* | +.101 | **+.164**  | **+.156**  | **+.203***  | **+.209*** |
| mean_entropy | log_n_contexts  | +.112* | +.075 | **+.139**  | +.126* | **+.188***  | **+.192*** |
| subjective_polysemy_mean | log_freq_dlexdb | +.062 | +.097 | +.117 | **+.191**  | **+.215***  | **+.233*** |
| subjective_polysemy_mean | log_n_contexts  | +.042 | +.062 | +.097 | +.148* | **+.192**   | **+.207*** |

n = 273–382 (Survey-Zeilen: n = 273–274).

**Der Innovation-Effekt überlebt die Frequenzkontrolle.** Mit `log_freq_dlexdb`
partialisiert ändern sich die Werte praktisch nicht (bei (c): +.213 → +.211 für die
Cluster-Anzahl, +.205 → +.203 für die Entropie, +.214 → +.215 für das Survey). Das
ist kein Zufall, sondern erwartbar: `jsd_excess` ist bereits null-korrigiert, und
die Permutations-Null absorbiert genau den stichprobengrößengetriebenen Anteil, über
den die Frequenz wirkt. Was Dubossarsky et al. als Artefakt beschreiben, ist hier
schon vor der Partialisierung entfernt — deshalb bleibt kaum etwas abzuziehen.

Mit `log_n_contexts` (dem direkten Stichprobenmaß) schrumpfen die Werte um etwa
10 %, bleiben aber signifikant: +.191 (Cluster), +.188 (Entropie), +.192 (Survey)
bei (c). Bei (d) und (e) überlebt die Kontrolle für die beiden modellbasierten
Prädiktoren, das Survey-Maß fällt bei (d) heraus (+.097, p = .11) und hält bei (e)
(+.148, p = .01).

**Einschränkung.** `n_clusters_coarse` und `mean_entropy` stammen aus derselben
Pipeline wie die JSD, teilen also Fehlerquellen mit der Zielgröße. Der saubere Test
ist das unabhängig erhobene Survey-Maß — und das liefert bei (c)/(c′) mit beiden
Kontrollen +.19 bis +.23. Die Innovation-Hypothese ist damit in der gröberen
Einteilung belastbar und nicht auf Frequenz reduzierbar; bei Dekaden bleibt sie
nicht nachweisbar.

### 5.4 Präsident

Das Fallbeispiel überlebt die gröbere Einteilung, und zwar monoton:

| Einteilung | jsd_obs | jsd_null | Excess | z | p_perm | p_BH |
|---|---|---|---|---|---|---|
| (a) 10 Dekaden | 0,057 | 0,041 | 0,017 | 1,36 | .100 | .414 |
| (b) 5 Perioden | 0,066 | 0,028 | 0,038 | 3,02 | .012 | .096 |
| (d) 4 Perioden | 0,088 | 0,025 | 0,063 | 5,13 | .001 | **.025** |
| (e) 3 Perioden | 0,105 | 0,023 | 0,082 | 6,12 | .001 | **.011** |
| (c) 2 Perioden | 0,131 | 0,018 | 0,112 | 8,16 | .001 | **.007** |
| (c′) o. 1940er | 0,210 | 0,022 | 0,188 | 11,42 | .001 | **.007** |

Stärkste Lemmas bei 2 Perioden (z): Rat (19,6), Markt (18,1), Schatz (14,9),
Spiegel (14,6), Weg (14,1), Schlüssel (12,7), Land (11,1), Start (11,1),
Ampel (10,0), Post (9,9). Bei (e): Schatz (22,3), Rat (14,3), Note (11,0),
Post (10,5), Markt (10,0), Feuer (7,7), Weg (7,4), Spiegel (7,2), Land (7,0),
Nummer (6,9) — die Spitzengruppe ist über die Einteilungen hinweg stabil.

### 5.5 Konsequenz für das Paper

Die Dekaden-Auflösung ist für die Wandel-*Detektion* zu feingranular: bei ~50 Sätzen
pro Lemma-Dekade sind 90 % der JSD Rauschen. Vorschlag: Wandel-Detektion und die
Hamilton-Korrelationen auf (d), (e) oder (c) umstellen und den Rauschanteil explizit
berichten; die Dekaden-Trajektorien bleiben für die Visualisierung und die
Fallstudien nützlich. (e) ist dabei ein vernünftiger Kompromiss — drei Perioden
erhalten noch eine Verlaufsrichtung (nicht bloß Vorher/Nachher), liefern 50
FDR-signifikante Lemmas und einen Innovation-Effekt, der beide Frequenzkontrollen
übersteht. Die Conformity-Claim ist in keiner Einteilung zu halten.

## 6. Explorativ: ist die WSD-Zuordnung bei älterem Deutsch schlechter?

**Klar als explorativ zu lesen.** Das Validierungsset hat 93 Instanzen über
10 Lemmas; nach einem Schnitt bei 1950 bleiben 31 Instanzen für die späte Gruppe.
Die Frage ist in der Dokumentation seit Längerem offen (`documentation/pipeline.md`
Z. 278, `research_log.md` Z. 454: „accuracy on 1900s data is unknown and likely
lower"), weil das 84,9-%-Maß implizit für alle Dekaden gilt.

Datenbasis: `data/evaluation/atemporal/results/predictions.csv` (eingereichte
`weighted`-Methode) verbunden mit den Datumsangaben aus
`data/evaluation/atemporal/input/contexts.csv`; coarse über
`sense_clusters_reviewed.json`, identisch zu `generate_predictions.py`. Die
Gesamtwerte reproduzieren die eingereichten 43,0 % / 84,9 % exakt.

| Teilmenge | Granularität | < 1950 | ≥ 1950 | Differenz | Fisher p |
|---|---|---|---|---|---|
| alle 10 Lemmas | fine   | **48,4 %** (30/62) | **32,3 %** (10/31) | −16,1 pp | .18 |
| alle 10 Lemmas | coarse | **88,7 %** (55/62) | **77,4 %** (24/31) | −11,3 pp | .22 |
| 9 Lemmas in beiden Hälften | fine   | 44,4 % (24/54) | 32,3 % (10/31) | −12,2 pp | .36 |
| 9 Lemmas in beiden Hälften | coarse | 90,7 % (49/54) | 77,4 % (24/31) | −13,3 pp | .11 |

**Antwort: nein — die Richtung ist umgekehrt.** Die Accuracy ist in der *frühen*
Hälfte höher, fine wie coarse, und kein Unterschied ist signifikant. Die
dokumentierte Sorge („auf 1900er-Daten wahrscheinlich schlechter") wird von diesen
Daten nicht gestützt.

Die zweite Teilmenge kontrolliert den naheliegenden Einwand: das Set umfasst nur
10 Lemmas, und *hören* hat keine Instanz nach 1950, der Era-Unterschied könnte also
Lemma-Zusammensetzung sein. Beschränkt auf die 9 Lemmas, die in beiden Hälften
vorkommen, bleibt die Richtung gleich.

**Was das trägt und was nicht.** Bei 31 späten Instanzen über 9 Lemmas sind
11–16 Prozentpunkte gut im Rauschen; der Check kann nur große Niveauunterschiede
ausschließen, keine Gleichheit zeigen. Relevant für die Diachronie ist ohnehin
nicht das Niveau, sondern eine *Drift* der Accuracy über die Zeit — ein konstanter
Fehler verzerrt alle Perioden gleich und erzeugt keinen Scheinwandel, eine Drift
schon. Gemessen wird hier keine signifikante Drift; der Punktschätzer zeigt
Richtung Gegenwart leicht *sinkende* Accuracy, also das Gegenteil der befürchteten
Asymmetrie. Das Set ist mit Median 1932 und 62 von 93 Instanzen vor 1950 deutlich
früh gewichtet — genau deshalb ist die späte Schätzung so dünn. Eine belastbare
Antwort bräuchte ein nach Dekaden stratifiziertes Gold-Set; die nicht annotierten
98 Kontexte des zweiten Annotationssets wären der nächste Schritt.

## Dateien

- `jsd_null_baseline.csv` – pro Lemma: jsd_obs, jsd_null_mean, jsd_null_sd, jsd_excess, jsd_z, p_perm
- `per_lemma_robustness.csv` – alles pro Lemma zusammengeführt
- `correlations.csv` – alle Korrelationen (alle Lemmas und nur k > 1)
- `run_log.txt` – Konsolenausgabe (§1–4)
- `binning_sensitivity.csv` – pro Lemma × Einteilung: Perioden-Bookkeeping, jsd_obs, Null-Baseline, Excess, z, p_perm, p_bh, Prädiktoren
- `binning_sensitivity_correlations.csv` – Korrelationen pro Einteilung × Test; Spalte `method` trennt `spearman` von `partial_spearman`, `control` nennt die Kontrollvariable
- `binning_sensitivity_wsd_era.csv` – §6: WSD-Accuracy fine/coarse vor vs. ab 1950, je für alle 10 Lemmas und für die 9 Lemmas in beiden Hälften
- `binning_sensitivity_log.txt` – Konsolenausgabe (§5–6)
