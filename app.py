import io
import os
import re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from groq import Groq
import pdfplumber
from docx import Document
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
JOB_DATA = None
RESUME_TEXT = None

# Groq client – reads GROQ_API_KEY from your environment
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

QUESTION_SYSTEM_PROMPT = """
You generate concise, role-relevant interview questions based on a resume and job role.
Return exactly one interview question in a single sentence. Do not add extra text.
"""

ALLOWED_EXTENSIONS = {"pdf", "docx"}


def _clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def _extract_text_from_pdf(file_bytes):
    text_chunks = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    return "\n".join(text_chunks)


def _extract_text_from_docx(file_bytes):
    document = Document(io.BytesIO(file_bytes))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def _is_allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _validate_job_payload(payload):
    if not isinstance(payload, dict):
        return None, "Invalid JSON payload."

    role = payload.get("role")
    description = payload.get("description")
    skills = payload.get("skills")

    if not isinstance(role, str) or not role.strip():
        return None, "role is required and must be a non-empty string."
    if not isinstance(description, str) or not description.strip():
        return None, "description is required and must be a non-empty string."
    if not isinstance(skills, list) or not skills:
        return None, "skills is required and must be a non-empty list of strings."
    if not all(isinstance(skill, str) and skill.strip() for skill in skills):
        return None, "skills must contain only non-empty strings."

    return {
        "role": role.strip(),
        "description": description.strip(),
        "skills": [skill.strip() for skill in skills],
    }, None


def extract_resume_text(file_bytes, filename):
    """Extract resume text from PDF or DOCX and normalize whitespace."""
    if filename.lower().endswith(".pdf"):
        raw_text = _extract_text_from_pdf(file_bytes)
    elif filename.lower().endswith(".docx"):
        raw_text = _extract_text_from_docx(file_bytes)
    else:
        raise ValueError("Unsupported file type.")
    return _clean_text(raw_text)


def parse_resume(file_bytes, filename):
    return extract_resume_text(file_bytes, filename)

def generate_interview_question(resume_text, job_role):
    messages = [
        {"role": "system", "content": QUESTION_SYSTEM_PROMPT.strip()},
        {
            "role": "user",
            "content": (
                "Resume:\n"
                f"{resume_text}\n\n"
                "Target role:\n"
                f"{job_role}\n\n"
                "Generate one interview question."
            ),
        },
    ]
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/set-job", methods=["POST"])
def set_job():
    data = request.get_json(silent=True)
    job_payload, error = _validate_job_payload(data)
    if error:
        return jsonify({"error": error}), 400

    global JOB_DATA
    JOB_DATA = job_payload
    return jsonify({"status": "ok", "job": JOB_DATA})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    history = data.get("history", [])

    if not RESUME_TEXT:
        return jsonify({"error": "Resume text is required before chatting."}), 400

    # Build conversation history
    messages = [{"role": "system", "content": RESUME_TEXT}]
    for msg in history:
        if msg.get("role") in ["user", "assistant"]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    try:
        # Call Groq LLM
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
        )
        reply = response.choices[0].message.content
        return jsonify({"reply": reply})
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Something went wrong on the server."}), 500


@app.route("/upload_resume", methods=["POST"])
def upload_resume():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not _is_allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Upload PDF or DOCX."}), 400

    filename = secure_filename(file.filename)
    file_bytes = file.read()
    if not file_bytes:
        return jsonify({"error": "Empty file provided."}), 400

    try:
        save_path = os.path.join(UPLOAD_DIR, filename)
        with open(save_path, "wb") as saved_file:
            saved_file.write(file_bytes)

        cleaned_text = extract_resume_text(file_bytes, filename)
    except Exception as exc:
        print("Resume parsing error:", exc)
        return jsonify({"error": "Failed to parse resume file."}), 500

    global RESUME_TEXT
    RESUME_TEXT = cleaned_text
    return jsonify({"filename": filename, "text": cleaned_text})


@app.route("/generate_question", methods=["POST"])
def generate_question():
    data = request.get_json() or {}
    resume_text = data.get("resume_text", "").strip()
    job_role = data.get("job_role", "").strip()

    if not resume_text or not job_role:
        return jsonify({"error": "resume_text and job_role are required."}), 400

    try:
        question = generate_interview_question(resume_text, job_role)
        return jsonify({"question": question})
    except Exception as exc:
        print("Question generation error:", exc)
        return jsonify({"error": "Failed to generate interview question."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
