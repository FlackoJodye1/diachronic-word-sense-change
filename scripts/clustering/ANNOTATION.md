# Cluster Annotation — Quickstart for Colleagues

Thanks for helping label sense clusters! This page tells you how to install
the tool, run it, and what the task actually is. Time estimate: **~3–5
hours of focused work** depending on your assignment.

---

## 1. Setup (one-time, ~10 min)

You need Python 3.10+ and `git`.

```bash
# 1. Clone the repo
git clone <repo-url> Worde-Sense-Disambiguation
cd Worde-Sense-Disambiguation

# 2. Create a virtual environment + install dependencies
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Pull the latest data (Paul will tell you if needed)
git pull
```

If `pip install` fails on a heavy dependency, you only really need
`flask` and `pandas` for the review app:

```bash
pip install flask pandas
```

---

## 2. Run the dashboard

Replace `<your-name>` with whatever Paul used for your assignment file
(e.g. `anna`, `ben`):

```bash
python scripts/clustering/review_app.py \
    --annotator <your-name> \
    --lemmas-file data/processed/sense_clustering/assignments/<your-name>.txt
```

Then open <http://127.0.0.1:8001> in your browser.

Your edits are saved to
`data/processed/germanet/sense_clusters_reviewed_<your-name>.json`.
Don't worry about that file — Paul will merge everyone's files at the end.

---

## 3. The task

For each lemma on your dashboard:

1. **Read the clusters.** Each cluster groups one or more GermaNet
   *fine* senses that the algorithm thinks belong to the same coarse
   meaning. You see, per sense: synset id, GermaNet class, a short
   gloss, and a fuller definition.
2. **Decide if the grouping is correct.** Ask: *would these senses
   appear under one entry in a learner's dictionary that distinguishes
   3–5 senses per polysemous word?*
   - If yes → leave members as they are.
   - If a sense doesn't belong → use the "Move to" dropdown to put it
     in another cluster, or → `New cluster` to split it off on its own.
3. **Write a short German label** for each cluster (3–7 words).
   Examples for `schlagen`:
   - `körperlich treffen` (physical strike)
   - `Herzschlag` (heart beating)
   - `Bäume fällen` (felling trees)
4. **Click Save.** Move on to the next lemma. (Use the "← prev / next →"
   links in the top nav.)

### Rules of thumb
- Don't aim for perfect — coarse clusters are inherently fuzzy. If two
  senses are close and a dictionary might lump them, lump them.
- Domain-specific extensions (chess, finance, etc.) usually deserve
  their own cluster.
- Metaphorical extensions ("Welle" = wave / "Welle" = trend) can go
  either way; pick what feels right and move on.
- A ⚠ "Constrained" warning means the algorithm collapsed everything to
  one cluster because all senses were too close. Often these *should*
  be split — check carefully.

---

## 4. When you're done

Send Paul:
- `data/processed/germanet/sense_clusters_reviewed_<your-name>.json`

Either commit + push, or zip and email — whichever is easier.

---

## Questions?

Ping Paul on Slack / email.
