import os
import json
from datetime import datetime, timezone
from typing import List, Dict

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, firestore

import cohere

# -----------------------------
# Env & Clients Initialization
# -----------------------------
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
COHERE_MODEL = os.getenv("COHERE_MODEL", "command-r-plus")
PORT = int(os.getenv("PORT", "5000"))
HOST = os.getenv("HOST", "127.0.0.1")

# Read Firebase credentials as JSON string from .env
FIREBASE_CREDENTIALS_JSON = os.getenv("FIREBASE_CREDENTIALS", "")

if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY missing in .env")

if not FIREBASE_CREDENTIALS_JSON:
    raise RuntimeError("FIREBASE_CREDENTIALS missing in environment variables")

try:
    cred_dict = json.loads(FIREBASE_CREDENTIALS_JSON)
except json.JSONDecodeError as e:
    raise RuntimeError(f"FIREBASE_CREDENTIALS is not valid JSON: {e}")

# Cohere client
co = cohere.Client(COHERE_API_KEY)

# Firebase Admin / Firestore
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# Helpers
# -----------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def chats_collection(uid: str):
    return db.collection("students").document(uid).collection("chats")

def fetch_chat_history(uid: str, limit: int = 14) -> List[Dict]:
    q = (
        chats_collection(uid)
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(limit)
    )
    docs = list(q.stream())
    return [d.to_dict() for d in reversed(docs)]

def format_history_for_prompt(history: List[Dict]) -> str:
    lines = []
    for item in history:
        role = item.get("role", "user").upper()
        msg = item.get("message", "")
        lines.append(f"{role}: {msg}")
    return "\n".join(lines)

def guardrails_prefix() -> str:
    return (
        "You are a friendly, encouraging AI tutor for students in grades 2-12. "
        "Your job: teach, explain clearly, ask follow-up questions, encourage, and help with school subjects. "
        "Stay strictly within educational content; do not discuss unrelated or unsafe topics. "
        "Use simple steps, examples, and short paragraphs. When helpful, ask the student a question to check understanding.\n"
    )

def build_prompt(history_text: str, latest_user_message: str) -> str:
    return (
        guardrails_prefix()
        + "Here is the recent conversation between YOU (the tutor) and the STUDENT:\n"
        + (history_text if history_text.strip() else "(no previous messages)\n")
        + "\nThe STUDENT just said:\n"
        + f"\"{latest_user_message}\"\n\n"
        "Respond now as the tutor."
    )

def generate_ai_reply(prompt: str) -> str:
    try:
        resp = co.generate(
            model=COHERE_MODEL,
            prompt=prompt,
            max_tokens=220,
            temperature=0.6,
            k=0,
            stop_sequences=[]
        )
        text = (resp.generations[0].text or "").strip()
        return text.replace("\n\n\n", "\n\n").strip()
    except Exception as e:
        return f"Oops—I'm having trouble thinking right now: {e}"

def save_message(uid: str, role: str, message: str) -> None:
    chats_collection(uid).add({
        "role": role,
        "message": message,
        "timestamp": now_iso()
    })

# -----------------------------
# Routes
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    uid = (data.get("uid") or "").strip()
    user_msg = (data.get("message") or "").strip()

    if not uid:
        return jsonify({"error": "uid is required"}), 400
    if not user_msg:
        return jsonify({"error": "message is required"}), 400

    save_message(uid, "user", user_msg)
    history = fetch_chat_history(uid, limit=14)
    history_text = format_history_for_prompt(history)
    prompt = build_prompt(history_text, user_msg)
    ai_reply = generate_ai_reply(prompt)
    save_message(uid, "ai", ai_reply)

    return jsonify({"reply": ai_reply})

@app.route("/generate_test", methods=["POST"])
def generate_test():
    """
    Body:
      { "uid": "<firebase-auth-uid>" }

    Behavior:
      - Fetch user chat history.
      - If enough data, ask Cohere to generate a test.
      - If not enough data (or AI fails), use a default test template.
      - Save test in Firestore under students/{uid}/tests/{testId}.
      - Return test JSON to frontend.
    """
    import json

    data = request.get_json(force=True, silent=True) or {}
    uid = (data.get("uid") or "").strip()

    if not uid:
        return jsonify({"error": "uid is required"}), 400

    # Pull last messages to create context
    history = fetch_chat_history(uid, limit=20)
    history_text = format_history_for_prompt(history)

    # ----------------------------
    # Default fallback test
    # ----------------------------
    default_test = {
        "questions": [
            {"type": "mcq", "subject": "Math", "question": "What is 5 + 3?", "options": ["5","6","7","8"], "answer": "8"},
            {"type": "mcq", "subject": "Math", "question": "Which number is even?", "options": ["3","7","10","9"], "answer": "10"},
            {"type": "mcq", "subject": "Science", "question": "Which planet is known as the Red Planet?", "options": ["Earth","Mars","Venus","Jupiter"], "answer": "Mars"},
            {"type": "mcq", "subject": "English", "question": "Choose the correct plural of 'child'.", "options": ["childs","children","childes","childer"], "answer": "children"},
            {"type": "mcq", "subject": "Math", "question": "What is 12 ÷ 4?", "options": ["2","3","4","6"], "answer": "3"},
            {"type": "mcq", "subject": "Science", "question": "Water boils at ___ °C.", "options": ["50","100","200","0"], "answer": "100"},
            {"type": "mcq", "subject": "General Knowledge", "question": "What is the capital of India?", "options": ["Delhi","Mumbai","Chennai","Kolkata"], "answer": "Delhi"},
            {"type": "mcq", "subject": "Math", "question": "What is the square of 9?", "options": ["18","81","27","72"], "answer": "81"},
            {"type": "mcq", "subject": "English", "question": "Fill in the blank: The sun ___ in the east.", "options": ["rise","rises","rising","rose"], "answer": "rises"},
            {"type": "mcq", "subject": "Science", "question": "Which gas do we breathe in to stay alive?", "options": ["Oxygen","Carbon Dioxide","Nitrogen","Helium"], "answer": "Oxygen"},
            {"type": "short", "subject": "English", "question": "Write a sentence using the word 'school'.", "answer": ""},
            {"type": "short", "subject": "Math", "question": "Explain how you would solve 25 ÷ 5.", "answer": ""},
            {"type": "short", "subject": "Science", "question": "Why is the sun important for life on Earth?", "answer": ""},
            {"type": "short", "subject": "General Knowledge", "question": "Name your favorite subject and explain why.", "answer": ""}
        ]
    }

    # If not enough history, return default
    if len(history) < 5:
        test_data = default_test
    else:
        # Otherwise try Cohere generation
        prompt = f"""
        You are an AI tutor. Based on this student's recent learning:
        {history_text}

        Create a test with:
        - 10 multiple choice questions (4 options each, mark the correct answer)
        - 4 short answer questions (leave 'answer' empty for student to fill).
        - Each question MUST include a "subject" field. Choose from: Math, Science, English, History, General Knowledge.

        Respond ONLY in valid JSON:
        {{
        "questions": [
            {{
            "type": "mcq",
            "subject": "Math",
            "question": "What is 2+2?",
            "options": ["2","3","4","5"],
            "answer": "4"
            }},
            {{
            "type": "short",
            "subject": "Science",
            "question": "Explain the process of photosynthesis.",
            "answer": ""
            }}
        ]
        }}
        """

        ai_reply = generate_ai_reply(prompt)

        try:
            test_data = json.loads(ai_reply)  # parse JSON
        except Exception:
            test_data = default_test  # fallback if Cohere fails

    # Save test to Firestore
    test_ref = db.collection("students").document(uid).collection("tests").document()
    test_ref.set({
        "createdAt": now_iso(),
        "questions": test_data["questions"],
        "completed": False,
        "score": None
    })

    return jsonify({"testId": test_ref.id, "questions": test_data["questions"]})


@app.route("/submit_test", methods=["POST"])
def submit_test():
    data = request.get_json(force=True, silent=True) or {}
    uid = (data.get("uid") or "").strip()
    test_id = (data.get("testId") or "").strip()
    answers = data.get("answers", [])

    if not uid or not test_id:
        return jsonify({"error": "uid and testId are required"}), 400

    # Score calculation
    subject_scores = {}
    correct_count = 0
    total_mcq = 0

    for q in answers:
        subj = q.get("subject", "General")
        if q.get("type") == "mcq":
            total_mcq += 1
            if q.get("studentAnswer") and q.get("studentAnswer") == q.get("answer"):
                correct_count += 1
                subject_scores[subj] = subject_scores.get(subj, {"correct": 0, "total": 0})
                subject_scores[subj]["correct"] += 1
            else:
                subject_scores[subj] = subject_scores.get(subj, {"correct": 0, "total": 0})
            subject_scores[subj]["total"] = subject_scores[subj].get("total", 0) + 1
        else:
            # Short answers not auto-scored, still count towards subject total
            subject_scores[subj] = subject_scores.get(subj, {"correct": 0, "total": 0})
            subject_scores[subj]["total"] += 1

    score = f"{correct_count}/{total_mcq}" if total_mcq > 0 else None

    # Save in Firestore
    test_ref = db.collection("students").document(uid).collection("tests").document(test_id)
    test_ref.update({
        "completed": True,
        "studentAnswers": answers,
        "score": score,
        "subjectScores": subject_scores
    })

    return jsonify({"message": "✅ Test submitted", "score": score, "subjectScores": subject_scores})


@app.route("/get_tests", methods=["POST"])
def get_tests():
    """
    Body: { "uid": "<firebase-auth-uid>" }
    Returns: All tests for this student with scores & dates.
    """
    data = request.get_json(force=True, silent=True) or {}
    uid = (data.get("uid") or "").strip()

    if not uid:
        return jsonify({"error": "uid is required"}), 400

    tests_ref = db.collection("students").document(uid).collection("tests")
    docs = tests_ref.order_by("createdAt").stream()

    results = []
    for d in docs:
        item = d.to_dict()
        item["testId"] = d.id
        results.append(item)

    return jsonify({"tests": results})

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print(f"🚀 AI Tutor backend running on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=True)

