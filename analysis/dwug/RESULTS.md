# DWUG DE – Ergebnisse (Schritt 2 und 3)

Stand 2026-10-31. Grundlage: die Entscheidungen in `INVENTORY.md` (27.10. und 31.10.2026).
Skripte: `scripts/dwug/predict_dwug.py` (Vorhersagen), `scripts/dwug/evaluate_dwug.py`
(Auswertung). Rohzahlen: `analysis/dwug/{per_word,ari,spearman,error_analysis}.csv`
(gitignored wie in `analysis/robustness`), Log: `analysis/dwug/evaluate_log.txt`.
Gold-Werte pro Wort stehen hier nur für die Beispielwörter der Fehleranalyse (Abschnitt D).
Die vollständige Tabelle pro Wort (unsere Werte und Gold) erzeugt `evaluate_dwug.py` lokal in
`analysis/dwug/local/per_word_full.md` (gitignored, Repo ist öffentlich).

## Kurzfassung

- **Wandel-Rangfolge:** Unsere JSD korreliert mit dem SemEval-2020-Gold mit
  **ρ = 0,68** (48 Wörter, 95 %-KI 0,48–0,83) und mit DWUG 3.0.0 `change_graded` mit
  **ρ = 0,64** (50 Wörter). Auf den Wörtern mit ≥ 2 Clustern (Zusatz 2) sind es **ρ = 0,80**
  (19 Wörter) bzw. **0,72** (21 Wörter).
- **Die Zahl unserer Cluster allein erklärt einen großen Teil der Hauptzahl** (ρ = 0,61
  gegen SemEval). Wörter mit einem Cluster bekommen JSD = 0 und haben im Mittel auch weniger
  Gold-Wandel (0,17 gegenüber 0,52). Innerhalb von Zusatz 2 trägt die Cluster-Zahl dagegen
  nichts bei (ρ = 0,06), die JSD schon (ρ = 0,80).
- **Rauschen ist hier klein.** Auf Zusatz 2 macht die Null-JSD im Median 21 % der
  beobachteten JSD aus; 19 von 21 Wörtern liegen bei p_perm < 0,05. Zum Vergleich: auf
  DWDS-Dekaden waren es rund 90 % (`analysis/robustness/RESULTS.md`).
  Der Unterschied passt zu zwei großen Perioden mit je ~100 Usages.
- **Cluster-Übereinstimmung:** mittlerer ARI 0,30 gegen die Graph-Cluster und 0,34 gegen die
  Sense-Labels (Wörter mit ≥ 2 Clustern). Eine zufällige Zuordnung liegt bei 0,00.
- **Robust:** Mit P1 = `context_normalized` und ohne Seminar/vergönnen ändern sich alle
  Kennzahlen um höchstens 0,07.
- **Die größten Fehler sind fehlende Bedeutungen, keine falschen Zuordnungen.** Alle fünf
  Wörter mit der größten Rangabweichung haben in GermaNet nur eine Bedeutung.

## Aufbau

| | |
|---|---|
| Wörter | 50 (DWUG DE 3.0.0); SemEval-Gold für 48 |
| Usages | 9.125 (P1 1800–1899: 4.125, P2 1946–1990: 5.000) |
| Methode | wie im Paper: multilingual-e5-large, `weighted` (0,5 Definition + 0,5 Mittel der 5 Beispiele), Kosinus-Argmax über die GermaNet-Bedeutungen, dann geprüfte Cluster |
| Eingabe | ganzer Satz, kein Grundform-Filter; > 60 Tokens → 60-Token-Fenster um das Zielwort (`indexes_target_token`). Gekürzt: 8,9 % (`context`), 22,5 % (`context_normalized`, dort zählen Satzzeichen als Token) |
| P1-Varianten | `context` (Haupt), `context_normalized` (Check); 48 P1-Usages ohne normalisierten Text (42 davon Abgesang) laufen im Check mit `context` |
| Ein-Cluster-Wörter | 29: 24 mit einer GermaNet-Bedeutung + 5 nach der Prüfung (Entscheidung, Frechheit, Seminar, vergönnen, vorweisen). Alle Usages im selben Cluster, JSD = 0 |
| Zusatz 2 | Wörter mit ≥ 2 Clustern nach der Prüfung: 19 (SemEval), 21 (DWUG 3.0.0) |
| Unabhängigkeit | Vorhersagen committet (`a2b4368`), bevor eine Gold-Datei gelesen wurde |

Statistik: JSD mit Basis 2 zwischen den Cluster-Verteilungen von P1 und P2. Null-Baseline wie
in `analysis/robustness` (`jsd_null_baseline`, wiederverwendet): Periodenlabels pro Wort
1000× permutiert, Periodengrößen fest, Seed 42; `jsd_excess` = beobachtet − Null-Mittel.
Spearman mit p-Wert und 95 %-Perzentil-Bootstrap-KI (10.000 Ziehungen über Wörter, Seed 42).
ARI-Zufall: pro Wort gleichverteilte Labels mit unserer Cluster-Zahl, 1000×.

## A) Cluster-Übereinstimmung (ARI)

Mittelwert über Wörter. Graph-Cluster: Usages mit −1 ausgeschlossen (251). Sense-Labels:
DWUG DE Sense 1.0.1, `maj_2`, Usages mit −1 (keine Mehrheit) ausgeschlossen (83), 1.117 übrig.

| Ziel | Wörter | ARI | ARI zufällig | Wörter ≥ 2 Cluster | ARI | ARI zufällig |
|---|---|---|---|---|---|---|
| Graph-Cluster | 50 | 0,227 | 0,100 | 21 | **0,302** | 0,000 |
| Sense-Labels (maj_2) | 24 | 0,244 | 0,000 | 17 | **0,344** | 0,000 |

- **Der Mittelwert über alle 50 Wörter ist schwer zu deuten.** Bei Ein-Cluster-Wörtern ist
  ARI = 1, wenn auch das Gold nur eine Klasse hat (5 Wörter: Ackergerät, Kubikmeter,
  Pachtzins, Unentschlossenheit, weitgreifend), sonst 0. Daher kommt auch der Zufallswert
  0,100. Aussagekräftig ist die Spalte „≥ 2 Cluster“.
- **ARI auf den 21 Wörtern:** Median 0,37, Spannweite −0,04 (Rezeption, Behandlung) bis 0,86 (zersetzen).
  Bei 4 Wörtern liegt er bei ≤ 0: Fuß, Behandlung, Rezeption, vorliegen.
- **„andere“:** In den `maj_2`-Labels hat kein Usage die Mehrheit für „andere“. Eine Variante
  ohne „andere“ ist daher identisch und wird nicht eigens berichtet. (Korrektur zu
  `INVENTORY.md`: Dort waren alle `sense0` als „andere“ gezählt. „andere“ hat aber je nach Wort
  eine andere ID.)

## B) Wandel (Spearman gegen Gold)

| Gold | Teilmenge | n | JSD | jsd_excess | Cluster-Zahl (Baseline) |
|---|---|---|---|---|---|
| **SemEval-2020** (Hauptzahl) | alle | 48 | **0,678** [0,48; 0,83] | 0,609 [0,37; 0,80] | 0,609 [0,41; 0,76] |
| SemEval-2020 | Zusatz 2 | 19 | **0,795** [0,50; 0,93] | 0,765 [0,45; 0,93] | 0,059 [−0,40; 0,52] |
| DWUG 3.0.0 `change_graded` (Zusatz 1) | alle | 50 | **0,644** [0,43; 0,81] | 0,574 [0,33; 0,78] | 0,577 [0,37; 0,74] |
| DWUG 3.0.0 `change_graded` | Zusatz 2 | 21 | **0,717** [0,39; 0,92] | 0,678 [0,31; 0,92] | 0,075 [−0,36; 0,52] |

ρ [95 %-KI]. Alle JSD- und Excess-Korrelationen haben p < 0,001, in allen drei Läufen; die Cluster-Zahl ist nur
auf allen Wörtern signifikant (p < 0,001), auf Zusatz 2 nicht (p = 0,81 bzw. 0,75).

**So lese ich die Tabelle:**

1. **Auf allen Wörtern** liegt die JSD nur 0,07 über der Cluster-Zahl, und die KIs überlappen
   stark. Ein großer Teil der Hauptzahl kommt daher, dass 29 Wörter mit einem Cluster
   gemeinsam ganz unten stehen und im Gold tatsächlich eher wenig Wandel zeigen.
2. **Innerhalb von Zusatz 2** misst die JSD Wandel, den die Cluster-Zahl nicht vorhersagt.
   Das ist die eigentliche Prüfung der Zuordnung.
3. **`jsd_excess` ist etwas schwächer als die rohe JSD.** Die Null-JSD hängt an der
   Cluster-Zahl (0,05 bei 2 Clustern, ~0,10 bei 4), das zieht Wörter mit vielen Clustern
   nach unten. Bei P2 = 100 Usages und P1 = 39–100 ist das Rauschen ohnehin klein.

Signifikanz pro Wort (Zusatz 2, 21 Wörter): 19 mit p_perm < 0,05. Nicht signifikant sind
Fuß (JSD 0,092, p = 0,68) und Mut (0,062, p = 0,49).

## C) Robustheit

| Lauf | ARI Graph (≥ 2) | ARI Sense (≥ 2) | ρ JSD SemEval alle / Z2 | ρ JSD DWUG alle / Z2 | ρ Excess SemEval alle / Z2 |
|---|---|---|---|---|---|
| Haupt (`context`) | 0,302 | 0,344 | 0,678 / 0,795 | 0,644 / 0,717 | 0,609 / 0,765 |
| P1 = `context_normalized` | 0,314 | 0,358 | 0,678 / 0,809 | 0,647 / 0,732 | 0,674 / 0,765 |
| ohne Seminar, vergönnen | 0,302 | 0,344 | 0,673 / 0,795 | 0,639 / 0,717 | 0,606 / 0,765 |

- **Normalisierte Schreibung:** 97,7 % der P1-Usages bekommen denselben Cluster wie mit dem
  Originaltext. Alle Werte steigen leicht, höchstens um 0,07 (Excess, alle Wörter).
  Die historische Orthographie von P1 schadet also kaum.
- **Ohne Seminar und vergönnen:** Beide haben nach der Prüfung nur einen Cluster. Sie fallen
  damit nur aus den Mengen „alle“ heraus (n = 46 bzw. 48). Zusatz 2 bleibt gleich.
  Die bekannten Definitionsfehler (`review_notes.md`) haben auf das Ergebnis also keinen
  messbaren Einfluss.

## D) Fehleranalyse

Rangabweichung = Rang unserer JSD − Rang SemEval-Gold (Hauptlauf, 48 Wörter; die 29 Wörter
mit JSD = 0 teilen sich Rang 15). Gold-Verteilungen aus `cluster_prob_dist1/2`.

| Wort | unsere JSD | SemEval | Rang uns / Gold | Was die Daten zeigen |
|---|---|---|---|---|
| abgebrüht | 0 | 0,833 | 15 / 48 | Nur eine GermaNet-Bedeutung; im Gold wechselt der dominante Cluster fast vollständig (P1: 95 % Cluster 1, P2: 93 % Cluster 0). |
| Knotenpunkt | 0 | 0,648 | 15 / 42 | Nur eine GermaNet-Bedeutung; im Gold wechselt der dominante Cluster (P1: 74 % Cluster 1, P2: 77 % Cluster 0). |
| artikulieren | 0 | 0,616 | 15 / 41 | Nur eine GermaNet-Bedeutung; im Gold verteilt sich P1 auf 13 Cluster (größter 49 %), P2 liegt zu 94 % in einem Cluster. |
| Armenhaus | 0 | 0,520 | 15 / 35 | Nur eine GermaNet-Bedeutung; im Gold wächst ein zweiter Cluster von 1 % (P1) auf 75 % (P2). |
| Sensation | 0 | 0,406 | 15 / 30 | Nur eine GermaNet-Bedeutung; im Gold verteilt sich P1 auf 63 % / 26 % / Rest, P2 liegt zu 99 % in einem Cluster. |

Alle fünf Fehler entstehen, weil GermaNet die Bedeutung nicht kennt, nicht durch eine falsche
Zuordnung. Das Gleiche gilt für alle Wörter mit einer Bedeutung: Mit der Methode ist für sie
nur JSD = 0 möglich.

**Zusatz: größte Abweichungen innerhalb von Zusatz 2** (Ränge unter 19 Wörtern). Hier geht es
um die Zuordnung selbst:

| Wort | unsere JSD | SemEval | Rang uns / Gold | Was die Daten zeigen |
|---|---|---|---|---|
| Fuß | 0,092 | 0,565 | 1 / 10 | Wir ordnen in beiden Perioden 86–89 % „unteres Ende / Basis / Längenmaß“ zu und nur ≤ 1 % „Körperteil“. Im Gold verschiebt sich die Verteilung von 39/55 % zu 68/2 %. ARI −0,03. |
| vorliegen | 0,302 | 0,190 | 9 / 2 | Wir sehen eine Verschiebung („zur Prüfung verfügbar sein“ 14 → 46 %), im Gold liegen beide Perioden zu 97 % in einem Cluster. ARI −0,01. |
| Abgesang | 0,285 | 0,579 | 8 / 12 | Die Richtung stimmt („Schlussbemerkung“ 47 → 79 %). Im Gold ist P1 aber zu 98 % ein Cluster, wir ordnen dort nur 53 % „Liedstrophenteil“ zu. ARI 0,03. |

## Einordnung und Grenzen

- **Zeitfenster:** DWUG misst 1800–1899 gegen 1946–1990, unsere Haupt-Analyse 1900–1999.
  Die Werte zeigen, ob die Methode Wandel erkennt. Mit den DWDS-JSDs sind sie nicht direkt
  vergleichbar.
- **P1 liegt außerhalb der Validierung.** Unser WSD-Modell ist auf 1901–1999 validiert.
  Dass `context_normalized` kaum etwas ändert, spricht gegen einen großen Orthographie-Effekt.
  Genauigkeit auf Usage-Ebene für das 19. Jh. haben wir aber nicht gemessen.
- **Zwei Gold-Versionen, nicht unabhängig.** SemEval-2020 und DWUG 3.0.0 beruhen auf
  denselben Annotationen in verschiedenen Ständen. Sie sind zwei Fassungen desselben Golds,
  keine zwei unabhängigen Tests.
- **Kleine n.** Die KIs für Zusatz 2 sind breit (z. B. 0,50–0,93).
- **Vergleich mit veröffentlichten Systemen** (XL-LEXEME / XL-DURel) ist der nächste Schritt
  und hier noch nicht enthalten.

## Reproduzieren

```bash
python scripts/dwug/predict_dwug.py      # einige Minuten auf MPS, schreibt data/processed/dwug/predictions/
python scripts/dwug/evaluate_dwug.py     # ~2 min, schreibt analysis/dwug/*.csv + evaluate_log.txt
```
