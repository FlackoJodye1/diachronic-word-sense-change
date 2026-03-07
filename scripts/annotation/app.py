#!/usr/bin/env python3
"""
Minimal Flask annotation interface.
"""

import pickle
import sys
from datetime import datetime
from pathlib import Path

from flask import Flask, request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.annotation.task import AnnotationResult

app = Flask(__name__)

# Load tasks
tasks_path = Path(__file__).parent.parent.parent / 'data' / 'evaluation' / 'annotation_tasks.pkl'
with open(tasks_path, 'rb') as f:
    tasks = pickle.load(f)

def save_tasks():
    with open(tasks_path, 'wb') as f:
        pickle.dump(tasks, f)

@app.route('/')
def dashboard():
    html = """
    <!DOCTYPE html>
    <html><head><title>WSD Annotation</title>
    <style>
        body { font-family: Arial; margin: 20px; }
        .word { border: 1px solid #ccc; margin: 10px; padding: 10px; }
        .completed { background: #d4edda; }
        .btn { background: #007bff; color: white; padding: 5px 10px; text-decoration: none; }
    </style></head>
    <body>
    <h1>WSD Annotation</h1>
    """
    
    for word, task in tasks.items():
        completed = len([r for r in task.results if r.annotator_id == "user"])
        total = len(task.instances)
        status = "completed" if completed == total else ""
        
        html += f'''
        <div class="word {status}">
            <h3>{word}</h3>
            <p>Progress: {completed}/{total}</p>
            <a href="/annotate/{word}" class="btn">Annotate</a>
        </div>
        '''
    
    html += "</body></html>"
    return html

@app.route('/annotate/<word>')
def annotate(word):
    if word not in tasks:
        return "Word not found", 404
    
    task = tasks[word]
    annotated_ids = {r.instance_id for r in task.results if r.annotator_id == "user"}
    
    # Find next instance
    instance = None
    for inst in task.instances:
        if inst.instance_id not in annotated_ids:
            instance = inst
            break
    
    if not instance:
        return f"<h1>All {word} instances completed!</h1><a href='/'>Back</a>"
    
    html = f"""
    <!DOCTYPE html>
    <html><head><title>Annotate: {word}</title>
    <style>
        body {{ font-family: Arial; margin: 20px; }}
        .context {{ background: #f8f9fa; padding: 15px; margin: 10px 0; }}
        .target {{ background: yellow; font-weight: bold; }}
        .sense {{ border: 1px solid #ccc; margin: 5px; padding: 10px; cursor: pointer; }}
        .sense:hover {{ background: #f0f0f0; }}
        .selected {{ background: #d4edda; border-color: green; }}
        .submit {{ background: #28a745; color: white; padding: 10px 20px; margin: 10px; }}
    </style></head>
    <body>
    <h1>Annotating: {word}</h1>
    <a href="/">← Back to Dashboard</a>
    
    <div class="context">
        <strong>Context:</strong><br>
        {instance.sentence.replace(word, f'<span class="target">{word}</span>')}
    </div>
    
    <h3>Select the correct sense:</h3>
    <form method="post">
    """
    
    for i, sense in enumerate(instance.candidate_senses):
        html += f'''
        <div class="sense" onclick="selectSense(this, '{sense['synset_id']}')">
            {i+1}. {sense['synset_id']} - {sense['text_for_embedding']}
        </div>
        '''
    
    html += f"""
    <input type="hidden" name="instance_id" value="{instance.instance_id}">
    <input type="hidden" name="word" value="{word}">
    <input type="hidden" id="selected_sense" name="selected_sense" value="">
    <br>
    <button type="submit" class="submit" id="submitBtn" disabled>Submit</button>
    </form>
    
    <script>
    function selectSense(element, senseId) {{
        document.querySelectorAll('.sense').forEach(el => el.classList.remove('selected'));
        element.classList.add('selected');
        document.getElementById('selected_sense').value = senseId;
        document.getElementById('submitBtn').disabled = false;
    }}
    </script>
    </body></html>
    """
    
    return html

@app.route('/annotate/<word>', methods=['POST'])
def submit_annotation(word):
    instance_id = request.form['instance_id']
    selected_sense = request.form['selected_sense']
    
    if not selected_sense:
        return "No sense selected", 400
    
    # Create result
    result = AnnotationResult(
        instance_id=instance_id,
        annotator_id="user",
        selected_sense_id=selected_sense,
        confidence=1.0,
        annotation_time_seconds=0,
        notes=None,
        timestamp=datetime.now().isoformat()
    )
    
    # Add result
    tasks[word].add_result(result)
    save_tasks()
    
    # Redirect to next
    return f'<script>window.location.href="/annotate/{word}";</script>'

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)