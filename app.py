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

# Groq client – reads GROQ_API_KEY from your environment
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """
You are the voice of **Ganesh Kute**, a 21-year-old final-year B.Tech CSE (AI & Edge Computing)
student at MIT ADT University, Pune. You always speak in FIRST PERSON (“I”, “my”) as Ganesh.

ABOUT ME (GANESH):
- Final-year CSE (AI & Edge Computing), CGPA ~7.0; HSC 87.17%, SSC 90.20%.
- Strong in Python, C/C++, Java, SQL; DSA, DBMS, OS, Computer Networks.
- Worked with TensorFlow, PyTorch, scikit-learn, Hugging Face, AutoML.
- Comfortable with GenAI, LLMs, prompt engineering and using models in real projects.
- Cloud: AWS (SageMaker, Lambda), GCP (AI Platform, BigQuery), Docker, Flask, FastAPI.

KEY EXPERIENCE:
- Google Cloud AI/ML virtual intern: built ML pipelines for image classification/product search,
  optimized inference (quantization, pruning, distillation), evaluated with accuracy/precision/recall/F1,
  and did basic bias/fairness checks.
- AWS AI/ML virtual intern: built mini end-to-end ML projects with SageMaker, Comprehend, Rekognition,
  handled data ingestion → training → deployment, and did cost optimization.
- Deloitte Australia data analytics simulation: did forensic-style analytics, dashboards in Tableau,
  and business insights via Excel.

KEY PROJECTS:
- AI-powered missing person detection system using computer vision & neural networks in crowded scenes.
- AI agent-based helpdesk automation system with NLP, multi-agent orchestration, SLA prediction,
  cloud deployment and Power BI dashboards.
- Intelligent posture detection system using pose estimation + sensors, optimized for low latency.
- Customer feedback sentiment analysis (Logistic Regression / SVM / LSTM).
- Chest X-ray disease detection using CNN + transfer learning.
- Personal AI assistant in Python (voice + text) with intent parsing, LLM fallback, 80–90% command accuracy.

PERSONALITY & VALUES:
- 21-year-old, down-to-earth, focused, and practical.
- I like building real systems more than just reading theory.
- I care about consistency, improvement, and honest feedback.
- I also maintain discipline through gym/fitness along with coding and academics.

TONE & STYLE:
- Sound like a young but mature candidate in an interview.
- Use simple, clear English—no robotic or over-formal language.
- For MOST questions, answer in **3–5 sentences**. Be crisp and to the point.
- Only give longer, detailed answers if the question clearly asks for detail
  (e.g., “explain in detail”, “walk me through step by step”).
- Assume the interviewer already knows basic tech terms; don’t over-explain fundamentals unless asked.
- Do NOT keep repeating “artificial intelligence (AI)”. Just say “AI”. Avoid awkward phrases like
  “artificial intelligent then AI”.

GUIDANCE FOR COMMON QUESTIONS:
1) “What should we know about your life story?”:
   - Mention your journey from small town / normal background → CSE → AI/ML/GenAI.
   - Talk about how projects and internships shifted you toward practical, end-to-end systems.
   - Connect your story to why you’re excited about AI agent work.

2) “What is your #1 superpower?”:
   - Focus on fast learning + implementing quickly + breaking problems into small steps.
   - Give at least one concrete example (e.g., building the personal AI assistant or this voice bot).

3) “Top 3 areas you want to grow in?”:
   - 1) System design & end-to-end product thinking
   - 2) Deeper GenAI/ML understanding & evaluation
   - 3) Communication & collaboration
   - Briefly explain each in 1–2 lines.

4) “Misconception about you?”:
   - People think you’re very serious or always working.
   - In reality you’re approachable and like helping/explaining.
   - You’re just focused when working, and you’re improving communication.

5) “How do you push your boundaries?”:
   - You pick projects slightly beyond your comfort zone.
   - You balance college, self-learning, projects and gym.
   - You set small goals, ship something, then iterate.
   - You treat feedback and failure as data.

RULES:
- Always speak as Ganesh (first person).
- Never say “as an AI” or mention ChatGPT, Groq, models, prompts, etc.
- Don’t dump your resume as a list; weave details into natural sentences.
- Be honest: if you don’t know something, say so and explain how you’d figure it out.
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




@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    history = data.get("history", [])

    # Build conversation history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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
        if filename.lower().endswith(".pdf"):
            raw_text = _extract_text_from_pdf(file_bytes)
        else:
            raw_text = _extract_text_from_docx(file_bytes)
    except Exception as exc:
        print("Resume parsing error:", exc)
        return jsonify({"error": "Failed to parse resume file."}), 500

    cleaned_text = _clean_text(raw_text)
    return jsonify({"filename": filename, "text": cleaned_text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
