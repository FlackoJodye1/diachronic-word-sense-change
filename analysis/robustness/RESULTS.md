# Robustheits-Analysen – Ergebnisse (16.10.2026)

Skripte (lesen nur, schreiben nur in diesen Ordner):
`analysis/robustness/robustness_analyses.py` (§1–4),
`analysis/robustness/binning_sensitivity.py` (§5).
Alle Originalzahlen aus dem Paper wurden exakt reproduziert (mean_jsd identisch bis 1e-14).
Null-Baseline: 1000 Permutationen pro Lemma, Seed 42.

## Kurzfassung

| Ergebnis im Paper | Nach Prüfung | Bewertung |
|---|---|---|
| H1: Survey ~ Cluster-Anzahl, r = .509 | .509 (alle), .394 ohne k=1 | hält |
| H2: Survey ~ Entropie, r = .436 | partiell für Cluster-Anzahl: **r = −.005** (p = .93) | Entropie trägt keine eigene Information |
| Innovation: Cluster-Anzahl ~ JSD, r = .774 | ohne k=1: .495; null-korrigiert: **r = .04** (p = .47) | verschwindet bei Dekaden — kommt bei gröberer Zeiteinteilung zurück (§5) |
| Conformity: Frequenz ~ JSD, r = −.278 / −.392 | null-korrigiert: **r = +.02 bis +.12** | verschwindet, in jeder Zeiteinteilung (§5) |
| Survey-Reliabilität | Split-Half .86 [.80, .90] | gut |

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

| Einteilung | Sätze/Lemma-Periode (Median) | JSD beob. | JSD null | **Null-Anteil** | p_perm < .05 | BH-FDR < .05 | Excess > 0 |
|---|---|---|---|---|---|---|---|
| (a) 10 Dekaden | 54 | 0,110 | 0,099 | **90 %** | 57 / 373 | 9 | 68 % |
| (b) 5 Perioden | 108 | 0,092 | 0,074 | **80 %** | 81 / 378 | 28 | 69 % |
| (c) 2 Perioden | 270 | 0,091 | 0,047 | **51 %** | 146 / 378 | 92 | 75 % |
| (c′) 2 Per. o. 1940er | 244 | 0,094 | 0,047 | **51 %** | 139 / 376 | 92 | 75 % |

Der Rauschboden fällt wie erwartet mit ~1/√n: viermal mehr Sätze pro Bin halbieren
die Null-JSD (0,099 → 0,047), während die beobachtete JSD fast konstant bleibt
(0,110 → 0,091). Entsprechend steigt der mittlere Excess von 0,011 auf 0,045.

**Die Dekaden-Analyse war unterpowert, nicht nur rauschdominiert.** Bei
2 Perioden zeigen 92 von 378 polysemen Lemmas (24 %) nach FDR-Korrektur echten
Wandel, bei Dekaden nur 9 (2 %). Das ist kein freies Mittagessen: 4 der 9
Dekaden-Treffer (Druck, Feuer, Preis, Stift) fallen bei 2 Perioden heraus — Wandel,
der in einer einzelnen Dekade konzentriert oder nicht-monoton ist, wird innerhalb
einer 50-Jahres-Periode weggemittelt. 87 Lemmas kommen neu dazu.

### 5.2 Korrelationen pro Einteilung (Spearman)

jsd_obs = rohe beobachtete JSD, jsd_excess = beobachtet − Null-Mittel,
jsd_z = Excess / Null-SD.

| Test | Prädiktor | Zielgröße | (a) 10 Dek. | (b) 5 Per. | (c) 2 Per. | (c′) o. 1940er |
|---|---|---|---|---|---|---|
| Conformity | log_freq_dlexdb | jsd_obs    | −.348*** | −.326*** | −.184*** | −.148** |
| Conformity | log_freq_dlexdb | jsd_excess | +.021 | +.065 | +.033 | +.053 |
| Conformity | log_freq_dlexdb | jsd_z      | +.120* | +.187*** | +.177*** | +.184*** |
| Innovation | n_clusters_coarse | jsd_obs    | +.495*** | +.438*** | +.365*** | +.380*** |
| Innovation | n_clusters_coarse | jsd_excess | +.043 | +.083 | **+.213***  | **+.221*** |
| Innovation | n_clusters_coarse | jsd_z      | +.042 | +.100 | **+.235***  | **+.236*** |
| Innovation | mean_entropy | jsd_obs    | +.442*** | +.340*** | +.316*** | +.331*** |
| Innovation | mean_entropy | jsd_excess | +.129* | +.108* | **+.205***  | **+.213*** |
| Innovation | mean_entropy | jsd_z      | +.106* | +.124* | **+.209***  | **+.218*** |
| Innovation | subjective_polysemy_mean | jsd_obs    | +.203*** | +.185** | +.254*** | +.270*** |
| Innovation | subjective_polysemy_mean | jsd_excess | +.062 | +.097 | **+.214***  | **+.232*** |
| Innovation | subjective_polysemy_mean | jsd_z      | +.082 | +.113 | **+.191**   | **+.220*** |

\* p < .05, \*\* p < .01, \*\*\* p < .001. n = 373–382 (Survey-Zeilen: n = 271–274).
Exakte p-Werte in `binning_sensitivity_correlations.csv`.

**Innovation hält, sobald der Rauschboden niedrig ist.** Die *rohe* Korrelation
schrumpft mit gröberen Bins (+.495 → +.365), die *null-korrigierte* wächst
(+.04, n.s. → +.21, p < .001). Das gilt für alle drei Polysemie-Maße, auch für die
unabhängig erhobenen Survey-Ratings (+.21). Der Befund aus §2 ("verschwindet") war
also ein Power-Problem der Dekaden-Einteilung, kein Null-Befund. Die Größenordnung
bleibt aber halbiert: etwa die Hälfte des eingereichten Effekts (+.37 roh vs. +.21
korrigiert) ist Sampling-Rauschen.

**Conformity hält in keiner Einteilung.** Die rohe negative Korrelation schrumpft
genau so, wie es ein Rauschartefakt tun muss (−.348 → −.184; häufigere Lemmas haben
mehr Sätze, also weniger Rauschen), und der Excess liegt in allen vier Einteilungen
bei 0 (n.s.).

Das positive jsd_z (+.12 bis +.19) ist **nicht** als umgekehrte Conformity zu lesen:
jsd_z skaliert mit der Stichprobengröße, und `jsd_null_sd` korreliert −.96 mit der
Kontextanzahl und −.75 mit log_freq_dlexdb. Häufige Lemmas bekommen bei gleichem
Excess mechanisch ein größeres z. Für Conformity ist daher der Excess die saubere
Größe — und der zeigt nichts. Für Innovation greift dieser Einwand nicht: die
Polysemie-Prädiktoren korrelieren nur +.13 bis +.19 mit der Kontextanzahl, und der
Effekt zeigt sich im Excess, der nicht SD-skaliert ist.

### 5.3 Präsident

Das Fallbeispiel überlebt die gröbere Einteilung:

| Einteilung | jsd_obs | jsd_null | Excess | z | p_perm | p_BH |
|---|---|---|---|---|---|---|
| (a) 10 Dekaden | 0,057 | 0,041 | 0,017 | 1,36 | .100 | .414 |
| (b) 5 Perioden | 0,066 | 0,028 | 0,038 | 3,02 | .012 | .096 |
| (c) 2 Perioden | 0,131 | 0,018 | 0,112 | 8,16 | .001 | **.007** |
| (c′) o. 1940er | 0,210 | 0,022 | 0,188 | 11,42 | .001 | **.007** |

Stärkste Lemmas bei 2 Perioden (z): Rat (19,6), Markt (18,1), Schatz (14,9),
Spiegel (14,6), Weg (14,1), Schlüssel (12,7), Land (11,1), Start (11,1),
Ampel (10,0), Post (9,9).

### 5.4 Konsequenz für das Paper

Die Dekaden-Auflösung ist für die Wandel-*Detektion* zu feingranular: bei ~50 Sätzen
pro Lemma-Dekade sind 90 % der JSD Rauschen. Vorschlag: Wandel-Detektion und die
Hamilton-Korrelationen auf (b) oder (c) umstellen und den Rauschanteil explizit
berichten; die Dekaden-Trajektorien bleiben für die Visualisierung und die
Fallstudien nützlich. Die Conformity-Claim ist in keiner Einteilung zu halten.

## Dateien

- `jsd_null_baseline.csv` – pro Lemma: jsd_obs, jsd_null_mean, jsd_null_sd, jsd_excess, jsd_z, p_perm
- `per_lemma_robustness.csv` – alles pro Lemma zusammengeführt
- `correlations.csv` – alle Korrelationen (alle Lemmas und nur k > 1)
- `run_log.txt` – Konsolenausgabe (§1–4)
- `binning_sensitivity.csv` – pro Lemma × Einteilung: Perioden-Bookkeeping, jsd_obs, Null-Baseline, Excess, z, p_perm, p_bh, Prädiktoren
- `binning_sensitivity_correlations.csv` – Korrelationen pro Einteilung × Test
- `binning_sensitivity_log.txt` – Konsolenausgabe (§5)
