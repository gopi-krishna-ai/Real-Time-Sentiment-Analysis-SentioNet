"""
Real-Time Sentiment Analysis (SentioNet prototype)
Flask web application entry point.
"""

import os
import socket

from flask import Flask, jsonify, render_template, request

from utils.sentiment_engine import analyze
from utils.audio_sentiment_engine import analyze_audio

app = Flask(__name__)

MAX_TEXT_LENGTH = 1000
ALLOWED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze_audio", methods=["POST"])
def analyze_audio_route():
    uploaded = request.files.get("audio")
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "Please choose an audio file."}), 400
    ext = os.path.splitext(uploaded.filename)[1].lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        return jsonify({"error": "Unsupported audio format. Use WAV, MP3, M4A, FLAC, OGG, or AAC."}), 400

    import tempfile
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            uploaded.save(tmp.name)
            path = tmp.name
        result = analyze_audio(path)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Audio analysis failed: {e}"}), 500
    finally:
        if path and os.path.exists(path):
            try: os.remove(path)
            except OSError: pass

@app.route("/analyze", methods=["POST"])
def analyze_route():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please enter some text to analyze."}), 400
    if len(text) > MAX_TEXT_LENGTH:
        return jsonify({"error": f"Please keep text under {MAX_TEXT_LENGTH} characters."}), 400

    try:
        result = analyze(text)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(result)


def find_free_port(preferred, attempts=20):
    """Returns the first usable port at or after `preferred`.

    On Windows a killed server can leave a stale LISTENING socket behind, which
    accepts connections and then resets them - the browser reports this as
    "could not reach the server" even though a new server appears to start fine.
    Probing with a real connect() detects those dead sockets, so we skip past
    them instead of silently binding a broken port.
    """
    for port in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.4)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                continue  # something is already answering here
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as binder:
                binder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                binder.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
    raise RuntimeError(f"No free port found in range {preferred}-{preferred + attempts}.")


if __name__ == "__main__":
    chosen_port = find_free_port(int(os.environ.get("PORT", 5150)))
    print("\n" + "=" * 60)
    print(f"  Real-Time Sentiment Analysis is starting")
    print(f"  Open this in your browser:  http://127.0.0.1:{chosen_port}/")
    print("=" * 60 + "\n")
    # use_reloader=False: the reloader re-executes this file, which would run the
    # port probe a second time and can land the child on a different port.
    app.run(debug=True, port=chosen_port, use_reloader=False)
