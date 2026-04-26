"""
NPS Classifier Web App
======================
Flask backend — loads the model once, classifies uploaded CSVs on demand.
"""

import ssl
import httpx

# macOS SSL fix (same as nps_classifier.py)
ssl._create_default_https_context = ssl._create_unverified_context
_orig_client = httpx.Client.__init__
def _ssl_client(self, *a, **k):
    k['verify'] = False
    _orig_client(self, *a, **k)
httpx.Client.__init__ = _ssl_client
_orig_async = httpx.AsyncClient.__init__
def _ssl_async(self, *a, **k):
    k['verify'] = False
    _orig_async(self, *a, **k)
httpx.AsyncClient.__init__ = _ssl_async

import os
import uuid
import threading
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import anthropic

from nps_classifier import (
    load_training_data,
    build_embedding_index,
    classify_feedback,
    SentenceTransformer,
    MODEL_NAME,
)

BASE_DIR   = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

NOISE_CATS = {"GIBBERISH", "EMPTY", "NON_ENGLISH"}
META_CATS  = {"POSITIVE_FEEDBACK", "INCONCLUSIVE"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# ── Global model state (loaded once at startup) ───────────────────────────────
_model            = None
_train_df         = None
_train_embeddings = None
_model_ready      = False
_model_error      = None

def _load_model():
    global _model, _train_df, _train_embeddings, _model_ready, _model_error
    try:
        print("[Startup] Loading sentence-transformer …")
        _model = SentenceTransformer(MODEL_NAME)
        print("[Startup] Loading training data …")
        train_raw = load_training_data()
        _train_df, _train_embeddings = build_embedding_index(train_raw, _model)
        _model_ready = True
        print("[Startup] Model ready.")
    except Exception as exc:
        _model_error = str(exc)
        print(f"[Startup] ERROR: {exc}")

# Load in background so Flask can start serving immediately
threading.Thread(target=_load_model, daemon=True).start()

# ── Job registry ──────────────────────────────────────────────────────────────
_jobs: dict = {}   # job_id -> {status, progress, total, error, output_path}
_jobs_lock = threading.Lock()

def _update_job(job_id, **kwargs):
    with _jobs_lock:
        _jobs[job_id].update(kwargs)

# ── Background classification ─────────────────────────────────────────────────
def _run_classification(job_id: str, input_path: str, text_col: str, score_col: str, source_type: str = "nps"):
    try:
        df = _read_csv(input_path, keep_default_na=False)
        texts      = df[text_col].astype(str).tolist()
        raw_scores = df[score_col].tolist() if score_col and score_col in df.columns else [None] * len(df)
        total      = len(df)
        _update_job(job_id, total=total, progress=0)

        def _to_nps(raw):
            """Convert raw score to NPS-scale (0–10) based on source type."""
            if raw is None or raw == "":
                return None
            try:
                val = float(raw)
            except (ValueError, TypeError):
                return None
            if source_type in ("android", "ios"):
                # Map 1–5 stars → 0–10 NPS-like scale
                return (val - 1) * 2.5
            return val

        results = []
        for i, (text, raw_score) in enumerate(zip(texts, raw_scores)):
            nps_score = _to_nps(raw_score) if score_col else None
            res = classify_feedback(
                text, _train_df, _train_embeddings, _model,
                nps_score=nps_score,
            )
            results.append(res)
            if (i + 1) % 200 == 0:
                _update_job(job_id, progress=i + 1)

        _update_job(job_id, progress=total)

        # Build output dataframe
        out = df.copy()
        out["predicted_category"] = [r["predicted_category"] for r in results]
        out["confidence_score"]   = [r["confidence_score"]   for r in results]

        # Clean CSV — keep original cols + two new ones, sort by priority
        def sort_key(cat):
            if cat in NOISE_CATS:  return 3
            if cat == "NULL":      return 4
            if cat in META_CATS:   return 2
            return 1
        out["_sort"] = out["predicted_category"].map(sort_key)
        out_sorted = out.sort_values(
            ["_sort", "predicted_category", "confidence_score"],
            ascending=[True, True, False]
        ).drop(columns=["_sort"])

        output_path = str(OUTPUT_DIR / f"{job_id}_output.csv")
        out_sorted.to_csv(output_path, index=False)

        # Compute summary stats
        cat_counts = out["predicted_category"].value_counts().to_dict()
        actionable = sum(v for k, v in cat_counts.items()
                         if k not in NOISE_CATS | META_CATS | {"NULL"})
        sentiment  = sum(v for k, v in cat_counts.items() if k in META_CATS)
        noise      = sum(v for k, v in cat_counts.items() if k in NOISE_CATS)
        null_count = cat_counts.get("NULL", 0)

        summary = {
            "total":      total,
            "actionable": actionable,
            "sentiment":  sentiment,
            "noise":      noise,
            "null":       null_count,
            "categories": {k: v for k, v in
                           sorted(cat_counts.items(), key=lambda x: -x[1])
                           if k not in {"NULL"}},
        }

        _update_job(job_id,
                    status="done",
                    output_path=output_path,
                    summary=summary)
        os.remove(input_path)

    except Exception as exc:
        _update_job(job_id, status="error", error=str(exc))


# ── AI Insights ───────────────────────────────────────────────────────────────
def _run_ai_insights(job_id: str, summary: dict, spec: str, source_type: str = "nps"):
    """Call Claude to generate tailored insights based on summary + user spec."""
    try:
        _update_job(job_id, ai_status="running")
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            _update_job(job_id, ai_status="error",
                        ai_error="ANTHROPIC_API_KEY not set in environment.")
            return

        client = anthropic.Anthropic(api_key=api_key)

        cats_text = "\n".join(
            f"  - {k}: {v}" for k, v in summary["categories"].items()
        )
        source_label = {"nps": "NPS survey", "android": "Android (Google Play) app review", "ios": "iOS (App Store) app review"}.get(source_type, "NPS survey")
        prompt = f"""You are an expert analyst for a fintech app for teenagers in India.

A batch of {summary['total']} {source_label} responses was classified into these categories:

{cats_text}

Summary:
- Actionable issues: {summary['actionable']}
- Sentiment (positive/inconclusive): {summary['sentiment']}
- Noise (gibberish/empty/non-English): {summary['noise']}
- Unresolved (NULL): {summary['null']}

User's specific focus / instructions:
\"\"\"{spec}\"\"\"

Based on the above data and the user's instructions, provide a concise, insightful analysis (5–8 bullet points). Focus on:
1. The most critical actionable issues given the user's focus area
2. Any patterns or anomalies worth flagging
3. Specific recommendations the product team should prioritize

Be direct and data-driven. Use the actual category names and counts in your response."""

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        ai_text = message.content[0].text
        _update_job(job_id, ai_status="done", ai_insights=ai_text)

    except Exception as exc:
        _update_job(job_id, ai_status="error", ai_error=str(exc))


# ── CSV reading with encoding fallback ───────────────────────────────────────
def _read_csv(source, **kwargs):
    """Try common encodings until one works."""
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=enc, **kwargs)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError("Could not decode file — unsupported encoding.")

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/model-status")
def model_status():
    return jsonify({"ready": _model_ready, "error": _model_error})


@app.route("/columns", methods=["POST"])
def columns():
    """Return column names from uploaded CSV for the user to map."""
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file"}), 400
    try:
        df = _read_csv(f, nrows=5, keep_default_na=False)
        return jsonify({
            "columns": df.columns.tolist(),
            "preview": df.head(3).to_dict("records"),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/classify", methods=["POST"])
def classify_route():
    if not _model_ready:
        return jsonify({"error": "Model not ready yet — please wait"}), 503

    f           = request.files.get("file")
    text_col    = request.form.get("text_col", "")
    score_col   = request.form.get("score_col", "")
    spec        = request.form.get("spec", "").strip()
    source_type = request.form.get("source_type", "nps")

    if not f or not text_col:
        return jsonify({"error": "Missing file or text_col"}), 400

    job_id     = str(uuid.uuid4())
    input_path = str(UPLOAD_DIR / f"{job_id}_input.csv")
    f.save(input_path)

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running", "progress": 0, "total": 0,
            "spec": spec,
            "source_type": source_type,
            "ai_status": "pending" if spec else "none",
        }

    def _classify_then_ai():
        _run_classification(job_id, input_path, text_col, score_col, source_type)
        if spec:
            with _jobs_lock:
                job = _jobs.get(job_id, {})
            if job.get("status") == "done" and job.get("summary"):
                _run_ai_insights(job_id, job["summary"], spec, source_type)

    threading.Thread(target=_classify_then_ai, daemon=True).start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def job_status(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    # Don't send the file path to the client
    safe = {k: v for k, v in job.items() if k != "output_path"}
    return jsonify(safe)


@app.route("/download/<job_id>")
def download(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Not ready"}), 404
    return send_file(
        job["output_path"],
        as_attachment=True,
        download_name="nps_classified_results.csv",
        mimetype="text/csv",
    )


if __name__ == "__main__":
    app.run(debug=False, port=5050, threaded=True)
