#!/usr/bin/env python3
"""
Minimal Flask interface for reviewing and labelling sense clusters.

Loads sense_clusters_0.10.json, lets you:
  - Assign a human-readable label to each cluster
  - Move individual senses to a different cluster or a new cluster

Saves the result to sense_clusters_reviewed.json.

Usage:
    python scripts/clustering/review_app.py
    → open http://127.0.0.1:8001
"""

import json
import sys
from pathlib import Path

from flask import Flask, request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.diachronic.io import load_all_senses

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
SOURCE_FILE = BASE_DIR / "data" / "processed" / "germanet" / "sense_clusters_0.10.json"
REVIEWED_FILE = BASE_DIR / "data" / "processed" / "germanet" / "sense_clusters_reviewed.json"
SUMMARY_FILE = BASE_DIR / "data" / "processed" / "sense_clustering" / "t010" / "cluster_summary.csv"

# ---------------------------------------------------------------------------
# Load data at startup
# ---------------------------------------------------------------------------

with open(SOURCE_FILE, "r", encoding="utf-8") as f:
    source_data = json.load(f)

if not REVIEWED_FILE.exists():
    with open(REVIEWED_FILE, "w", encoding="utf-8") as f:
        json.dump(source_data, f, ensure_ascii=False, indent=2)

with open(REVIEWED_FILE, "r", encoding="utf-8") as f:
    reviewed_data = json.load(f)

senses_df = load_all_senses(BASE_DIR)
senses_lookup = {row["synset_id"]: row.to_dict() for _, row in senses_df.iterrows()}

pilot_lemmas: set[str] = set()
_pilot_path = BASE_DIR / "data" / "processed" / "germanet" / "senses_pilot.jsonl"
if _pilot_path.exists():
    import json as _json
    with open(_pilot_path, encoding="utf-8") as _f:
        for _line in _f:
            pilot_lemmas.add(_json.loads(_line)["lemma"])

constrained_set = set()
if SUMMARY_FILE.exists():
    import pandas as pd
    summary_df = pd.read_csv(SUMMARY_FILE)
    constrained_set = set(summary_df[summary_df["constrained"]]["lemma"].tolist())


def save_reviewed():
    with open(REVIEWED_FILE, "w", encoding="utf-8") as f:
        json.dump(reviewed_data, f, ensure_ascii=False, indent=2)


def count_labeled(lemma):
    clusters = reviewed_data[lemma]["clusters"]
    labeled = sum(1 for c in clusters.values() if c.get("label"))
    return labeled, len(clusters)


def next_cluster_id(lemma):
    """Generate the next available cluster ID for a lemma."""
    existing = reviewed_data[lemma]["clusters"].keys()
    indices = []
    for k in existing:
        try:
            indices.append(int(k.replace(f"{lemma}_c", "")))
        except ValueError:
            pass
    return f"{lemma}_c{max(indices) + 1 if indices else 0}"


def rebuild_mapping(lemma):
    """Rebuild the flat mapping from the clusters structure."""
    clusters = reviewed_data[lemma]["clusters"]
    mapping = {}
    for cluster_id, cluster in clusters.items():
        for sid in cluster["members"]:
            mapping[sid] = cluster_id
    reviewed_data[lemma]["mapping"] = mapping


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
    body { font-family: Arial, sans-serif; margin: 0; background: #f5f5f5; }
    .container { max-width: 960px; margin: 0 auto; padding: 20px; }
    h1 { color: #333; }
    a { color: #007bff; text-decoration: none; }
    a:hover { text-decoration: underline; }

    /* Progress bar */
    .progress-bar-wrap { background: #e9ecef; border-radius: 6px; height: 18px; margin: 10px 0 4px 0; overflow: hidden; }
    .progress-bar-fill { height: 100%; border-radius: 6px; background: #28a745; transition: width 0.3s; }
    .stats-row { display: flex; gap: 24px; margin: 10px 0 18px 0; flex-wrap: wrap; }
    .stat-box { background: white; border: 1px solid #ddd; border-radius: 6px; padding: 10px 18px; min-width: 120px; text-align: center; }
    .stat-box .stat-val { font-size: 22px; font-weight: bold; color: #333; }
    .stat-box .stat-val.green { color: #28a745; }
    .stat-box .stat-val.yellow { color: #e6a800; }
    .stat-box .stat-val.gray { color: #aaa; }
    .stat-box .stat-label { font-size: 11px; color: #888; margin-top: 2px; }
    .section-header { font-size: 13px; font-weight: bold; color: #555; margin: 20px 0 8px 0; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid #ddd; padding-bottom: 4px; }

    /* Dashboard */
    .lemma-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; margin-top: 8px; }
    .lemma-card { background: white; border: 1px solid #ddd; border-radius: 6px; padding: 12px; }
    .lemma-card.done { border-color: #28a745; background: #f0fff4; }
    .lemma-card.partial { border-color: #ffc107; background: #fffdf0; }
    .lemma-card h3 { margin: 0 0 6px 0; font-size: 15px; }
    .lemma-card .progress { font-size: 12px; color: #666; }
    .badge { font-size: 10px; background: #dc3545; color: white; padding: 2px 5px; border-radius: 3px; margin-left: 4px; }

    /* Review page */
    .cluster { background: white; border: 1px solid #ddd; border-radius: 6px; margin-bottom: 20px; padding: 16px; }
    .cluster-header { display: flex; align-items: baseline; gap: 12px; margin-bottom: 12px; }
    .cluster-header h2 { margin: 0; font-size: 16px; color: #333; }
    .cluster-header .count { font-size: 12px; color: #888; }
    .member { border-left: 3px solid #007bff; padding: 8px 12px; margin-bottom: 10px; background: #f8f9fa; border-radius: 0 4px 4px 0; }
    .member-top { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .member .synset { font-weight: bold; font-size: 13px; color: #333; }
    .member .cls { font-size: 11px; color: #999; background: #e9ecef; padding: 1px 5px; border-radius: 3px; }
    .member .gloss { font-size: 13px; color: #555; margin: 4px 0 2px 0; }
    .member .definition { font-size: 12px; color: #777; line-height: 1.4; }
    .move-row { margin-top: 8px; display: flex; align-items: center; gap: 6px; }
    .move-row label { font-size: 11px; color: #999; }
    .move-row select { font-size: 12px; padding: 2px 6px; border: 1px solid #ccc; border-radius: 3px; color: #555; }
    .label-row { display: flex; align-items: center; gap: 10px; margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee; }
    .label-row label { font-size: 13px; font-weight: bold; color: #444; white-space: nowrap; }
    .label-row input { flex: 1; padding: 6px 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; }
    .btn-save { background: #28a745; color: white; border: none; padding: 10px 24px; border-radius: 4px; font-size: 14px; cursor: pointer; margin-top: 10px; }
    .btn-save:hover { background: #218838; }
    .nav { margin-bottom: 20px; display: flex; gap: 16px; align-items: center; }
    .constrained-warning { background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 8px 12px; margin-bottom: 16px; font-size: 13px; }
    .saved-notice { background: #d4edda; border: 1px solid #28a745; border-radius: 4px; padding: 8px 12px; margin-bottom: 16px; font-size: 13px; }
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    lemmas = sorted(reviewed_data.keys())
    total_labeled = 0
    total_clusters = 0
    words_done = 0
    words_partial = 0
    words_empty = 0

    pilot_cards = ""
    expanded_cards = ""

    for lemma in lemmas:
        labeled, n = count_labeled(lemma)
        total_labeled += labeled
        total_clusters += n
        if labeled == n:
            words_done += 1
        elif labeled > 0:
            words_partial += 1
        else:
            words_empty += 1

        cls = "done" if labeled == n else ("partial" if labeled > 0 else "")
        badge = '<span class="badge">C</span>' if lemma in constrained_set else ""
        card = f"""
        <div class="lemma-card {cls}">
            <h3><a href="/review/{lemma}">{lemma}</a>{badge}</h3>
            <div class="progress">{labeled}/{n} clusters labeled</div>
        </div>"""
        if lemma in pilot_lemmas:
            pilot_cards += card
        else:
            expanded_cards += card

    pct = round(100 * total_labeled / total_clusters) if total_clusters else 0
    bar = f'<div class="progress-bar-wrap"><div class="progress-bar-fill" style="width:{pct}%"></div></div>'

    stats = f"""
    <div style="margin-top:6px;font-size:13px;color:#555;">{pct}% of clusters labeled ({total_labeled}/{total_clusters})</div>
    {bar}
    <div class="stats-row">
        <div class="stat-box"><div class="stat-val green">{words_done}</div><div class="stat-label">Words fully done</div></div>
        <div class="stat-box"><div class="stat-val yellow">{words_partial}</div><div class="stat-label">Words in progress</div></div>
        <div class="stat-box"><div class="stat-val gray">{words_empty}</div><div class="stat-label">Words not started</div></div>
        <div class="stat-box"><div class="stat-val">{len(lemmas)}</div><div class="stat-label">Total words</div></div>
    </div>"""

    return f"""<!DOCTYPE html>
<html><head><title>Cluster Review</title><style>{CSS}</style></head>
<body><div class="container">
<h1>Sense Cluster Review</h1>
{stats}
<p style="font-size:12px;color:#666;"><span class="badge">C</span> = constrained (forced split)</p>
<div class="section-header">Pilot words ({len(pilot_lemmas)} lemmas)</div>
<div class="lemma-grid">{pilot_cards}</div>
<div class="section-header">Expanded words ({len(lemmas) - len(pilot_lemmas)} lemmas)</div>
<div class="lemma-grid">{expanded_cards}</div>
</div></body></html>"""


@app.route("/review/<lemma>")
def review(lemma, saved=False):
    if lemma not in reviewed_data:
        return "Lemma not found", 404

    clusters = reviewed_data[lemma]["clusters"]
    cluster_ids = sorted(clusters.keys())
    labeled, n = count_labeled(lemma)

    lemmas_sorted = sorted(reviewed_data.keys())
    idx = lemmas_sorted.index(lemma)
    prev_lemma = lemmas_sorted[idx - 1] if idx > 0 else None
    next_lemma = lemmas_sorted[idx + 1] if idx < len(lemmas_sorted) - 1 else None

    nav = '<a href="/">← Dashboard</a>'
    if prev_lemma:
        nav += f' &nbsp;|&nbsp; <a href="/review/{prev_lemma}">← {prev_lemma}</a>'
    if next_lemma:
        nav += f' &nbsp;|&nbsp; <a href="/review/{next_lemma}">{next_lemma} →</a>'

    warning = ""
    if lemma in constrained_set:
        warning = '<div class="constrained-warning">⚠ <strong>Constrained:</strong> threshold collapsed all senses to 1 cluster — this split was forced. Check if the split is linguistically defensible.</div>'

    notice = '<div class="saved-notice">✓ Saved.</div>' if saved else ""

    clusters_html = ""
    for cluster_id in cluster_ids:
        cluster = clusters[cluster_id]
        members = cluster["members"]
        current_label = cluster.get("label") or ""
        n_members = len(members)

        members_html = ""
        for sid in members:
            sense = senses_lookup.get(sid, {})
            cls = sense.get("class", "—")
            core_gloss = sense.get("core_gloss", "—")
            llm_def = sense.get("llm_definition", "—")

            # Move-to dropdown — other clusters + new cluster option
            options = ""
            for other_id in cluster_ids:
                selected = 'selected' if other_id == cluster_id else ''
                options += f'<option value="{other_id}" {selected}>{other_id}</option>'
            options += f'<option value="__new__">→ New cluster</option>'

            members_html += f"""
            <div class="member">
                <div class="member-top">
                    <span class="synset">{sid}</span>
                    <span class="cls">{cls}</span>
                </div>
                <div class="gloss">{core_gloss}</div>
                <div class="definition">{llm_def}</div>
                <div class="move-row">
                    <label>Move to:</label>
                    <select name="move_{sid}">
                        {options}
                    </select>
                </div>
            </div>"""

        clusters_html += f"""
        <div class="cluster">
            <div class="cluster-header">
                <h2>{cluster_id}</h2>
                <span class="count">{n_members} fine sense{"s" if n_members != 1 else ""}</span>
            </div>
            {members_html}
            <div class="label-row">
                <label>Label:</label>
                <input type="text" name="label_{cluster_id}" value="{current_label}" placeholder="e.g. physical strike">
            </div>
        </div>"""

    return f"""<!DOCTYPE html>
<html><head><title>Review: {lemma}</title><style>{CSS}</style></head>
<body><div class="container">
<div class="nav">{nav}</div>
<h1>{lemma} <span style="font-size:14px;color:#666;font-weight:normal;">({labeled}/{n} labeled)</span></h1>
{warning}{notice}
<form method="post">
{clusters_html}
<button type="submit" class="btn-save">Save</button>
</form>
</div></body></html>"""


@app.route("/review/<lemma>", methods=["POST"])
def save_review(lemma):
    if lemma not in reviewed_data:
        return "Lemma not found", 404

    clusters = reviewed_data[lemma]["clusters"]
    cluster_ids = sorted(clusters.keys())

    # Collect all sense IDs across all clusters
    all_senses = [sid for c in clusters.values() for sid in c["members"]]

    # Process moves — build new cluster assignments
    new_assignments = {}  # sid -> cluster_id
    for sid in all_senses:
        target = request.form.get(f"move_{sid}", "")
        if target == "__new__":
            # Assign to a temporary key; resolved below
            new_assignments[sid] = f"__new__{sid}"
        else:
            new_assignments[sid] = target if target else [c for c in cluster_ids if sid in clusters[c]["members"]][0]

    # Resolve __new__ assignments: group them into a single new cluster per unique key
    # Each sense moved to __new__ gets its own new cluster (split behaviour)
    new_cluster_base = next_cluster_id(lemma)
    new_cluster_counter = int(new_cluster_base.replace(f"{lemma}_c", ""))
    resolved = {}
    for sid, target in new_assignments.items():
        if target.startswith("__new__"):
            resolved[sid] = f"{lemma}_c{new_cluster_counter}"
            new_cluster_counter += 1
        else:
            resolved[sid] = target

    # Rebuild clusters structure
    new_clusters = {}
    # Preserve existing labels
    existing_labels = {cid: c.get("label") for cid, c in clusters.items()}

    for sid, cid in resolved.items():
        if cid not in new_clusters:
            new_clusters[cid] = {"label": existing_labels.get(cid), "members": []}
        new_clusters[cid]["members"].append(sid)

    # Remove any empty clusters (all members moved out)
    new_clusters = {k: v for k, v in new_clusters.items() if v["members"]}

    # Apply labels from form
    for cluster_id in list(new_clusters.keys()):
        label_value = request.form.get(f"label_{cluster_id}", "").strip()
        new_clusters[cluster_id]["label"] = label_value if label_value else None

    reviewed_data[lemma]["clusters"] = new_clusters
    rebuild_mapping(lemma)
    save_reviewed()

    return review(lemma, saved=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8001, debug=True)