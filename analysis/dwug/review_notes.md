# DWUG DE – Notizen zur manuellen Cluster-Prüfung

Stand 2026-10-31. Geprüft von Paul in `review_app.py`, extern gegengelesen.
Prüfung der 23 DWUG-Wörter mit ≥ 2 GermaNet-Bedeutungen, die nicht unter den 502 sind.
Ausgangspunkt: `data/processed/dwug/sense_clusters_dwug_0.10.json` (automatisch, τ = 0,10).
Ergebnis: `data/processed/dwug/sense_clusters_dwug_reviewed.json`.

## Überblick

- 23 Wörter, 70 Synsets. Automatisch 55 Cluster, nach der Prüfung **54**.
- **10 Wörter geändert**, 13 unverändert übernommen.
- **5 Wörter haben jetzt nur ein Cluster:** Entscheidung, Frechheit, Seminar, vergönnen,
  vorweisen. Auf Coarse-Ebene ergibt sich für sie JSD = 0, wie bei den Wörtern mit nur einer
  GermaNet-Bedeutung (siehe `INVENTORY.md`, „Entscheidungen“).

## Änderungen je Wort

Cluster als Synset-Mengen; nach der Prüfung mit Label.

| Wort | Cluster vorher → nachher | Art | Automatisch | Nach der Prüfung |
|---|---|---|---|---|
| Behandlung | 2 → 3 | Trennung | {s107037, s19401, s20478}, {s73156} | Medizinische Behandlung {s19401, s20478}; Fallbearbeitung {s107037}; Experimentelle Intervention {s73156} |
| Entscheidung | 2 → 1 | Zusammenlegung | {s18388, s28698}, {s73308} | Entscheidungsfindung {s18388, s28698, s73308} |
| Frechheit | 2 → 1 | Zusammenlegung | {s13627}, {s14083} | Unverschämtheit {s13627, s14083} |
| Seminar | 2 → 1 | Zusammenlegung | {s18118, s75925}, {s109889} | Lehrveranstaltung {s109889, s18118, s75925} |
| Spielball | 4 → 3 | Zusammenlegung | {s33157}, {s38557}, {s7942}, {s11162} | Strategischer Spielzug {s33157}; Manipulierbare Person {s38557}; Spielgerät {s11162, s7942} |
| abbauen | 2 → 4 | Trennung | {s53403, s57372, s58795, s59673}, {s54200} | Demontage {s57372}; Rohstoffgewinnung {s58795}; Reduzierung {s53403, s59673}; Kräfteverlust {s54200} |
| abdecken | 3 → 4 | Trennung | {s56879, s57055}, {s54178}, {s108653} | Abdeckung entfernen {s56879}; Bedecken {s57055}; Bedarf decken {s54178}; Gegenspieler decken {s108653} |
| verbauen | 3 → 4 | Umordnung | {s58902}, {s97885}, {s100861, s59921, s66179} | Flächen bebauen {s58902, s97885}; Zukunftschancen blockieren {s59921}; Baumaterial einbauen {s66179}; Zugang versperren {s100861} |
| vergönnen | 2 → 1 | Zusammenlegung | {s55493}, {s52881} | Jemandem etwas gönnen {s52881, s55493} |
| vorweisen | 2 → 1 | Zusammenlegung | {s52373}, {s58755} | Nachweis vorzeigen {s52373, s58755} |

Unverändert (nur Labels vergeben): Abgesang, Dynamik, Eintagsfliege, Engpass, Manschette,
Ohrwurm, Rezeption, Schmiere, Titel, ausspannen, vorliegen, zersetzen, überspannen.

## Textfehler in den LLM-Texten

Vergleichsbasis ist der GermaNet-Basistext (`baseline` in `senses_dwug.jsonl`).
`core_gloss` und `llm_definition` sind beide LLM-generiert (gpt-4o-mini).
`core_gloss` geht ins Clustering ein, `llm_definition` in die WSD-Repräsentation (`weighted`).

| Wort | Synset | Feld | LLM-Text | GermaNet-Basistext |
|---|---|---|---|---|
| Behandlung | s107037 | `core_gloss` | „Medizinische Versorgung zur Heilung oder Verbesserung von Gesundheitszuständen“ | „Behandlung ist eine Art von Bearbeitung.“ |
| Ohrwurm | s49596 | `core_gloss` | „Melodie, die im Gedächtnis bleibt und wiederholt wird“ | „Ohrwurm ist eine Art von beißendes Fluginsekt.“ |
| ausspannen | s109412 | `core_gloss` | „Anhänger zum Transport von Lasten benutzen“ | „Er spannte den Anhänger aus.“ |
| Seminar | s109889 | `llm_definition` | „… eine Lehrveranstaltung an einer Hochschule …“ | „Seminar ist eine Art von Hochschulinstitut.“ |
| vergönnen | s55493 | `llm_definition` | „… jemandem etwas nicht zu gestatten oder zu verwehren …“ (Bedeutung umgekehrt) | Beispiele: „… ist mir wohl nicht vergönnt.“ / „Man vergönnte ihm nicht zu triumphieren.“ |

Beobachtungen zu den einzelnen Fällen:

- **Behandlung:** Wegen des falschen Glosses hat das automatische Clustering s107037 in das
  medizinische Cluster gelegt. Die Prüfung hat es wieder abgetrennt. Die Definition ist korrekt.
- **Ohrwurm, ausspannen:** Nur der Gloss ist falsch, die Definitionen treffen die Bedeutung.
  Die Cluster waren trotzdem richtig getrennt.
- **Seminar, vergönnen:** Beide Wörter wurden in der Prüfung zu einem Cluster
  zusammengelegt. Die App zeigte dabei die fehlerhafte Definition an.

**Muster.**
1. Wenn GermaNet wenig Text liefert (nur Hyperonym oder ein Beispielsatz), wählt das LLM oft
   die häufigste Bedeutung des Worts statt der gemeinten.
2. Verneinte Beispielsätze können dazu führen, dass das LLM die Bedeutung umkehrt (vergönnen).

**Umgang.** Für die DWUG-Evaluation wird nichts korrigiert. Die Texte bleiben so, wie die
Methode sie erzeugt, genau wie im Paper. Die Fehler sind hier dokumentiert, damit sie bei der
Auswertung und in der Diskussion berücksichtigt werden können.
