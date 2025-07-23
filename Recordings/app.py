from flask import Flask, render_template, jsonify
import json
import os

app = Flask(__name__)
LOG_FILE = "record_log.json"

@app.route("/")
def index():
    if not os.path.exists(LOG_FILE):
        return "No log file found."

    with open(LOG_FILE, "r") as f:
        data = json.load(f)

    summary = {}
    for main_class, subclasses in data.items():
        total = sum(sub["count"] for sub in subclasses.values())
        subclass_count = len(subclasses)
        summary[main_class] = {"total": total, "num_subclasses": subclass_count, "subclasses": subclasses}

    return render_template("index.html", summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
