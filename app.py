from flask import Flask, render_template, request, redirect, url_for, session
import os
import pickle
import PyPDF2
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "ultra_resume_ai"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- PDF TEXT EXTRACTION ----------------
def extract_text(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text.lower()


# ---------------- ATS SCORE FUNCTION ----------------
def calculate_ats(text):
    words = len(text.split())
    score = min(int((words / 300) * 100), 100)
    return score


# ---------------- SKILL EXTRACTION ----------------
def extract_skills(text):
    skills_db = [
        "python", "java", "sql", "machine learning", "deep learning",
        "tensorflow", "pandas", "numpy", "html", "css",
        "power bi", "flask", "django"
    ]
    return [skill for skill in skills_db if skill in text]


# ---------------- UPLOAD + PROCESS ----------------
@app.route("/upload", methods=["POST"])
def upload_resume():
    file = request.files["resume"]

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # extract text
    resume_text = extract_text(filepath)

    # ML prediction
    vector = vectorizer.transform([resume_text])
    role = model.predict(vector)[0]

    # ATS score
    ats_score = calculate_ats(resume_text)

    # skills
    skills = extract_skills(resume_text)

    # rank logic
    if ats_score > 80:
        rank = "Top 5%"
    elif ats_score > 60:
        rank = "Top 20%"
    else:
        rank = "Average"

    # recommendation
    if ats_score > 80:
        recommendation = "Excellent resume. Apply for top companies."
    else:
        recommendation = "Improve skills section and add more projects."

    result = {
        "role": role,
        "ats_score": f"{ats_score}%",
        "rank": rank,
        "skills": skills,
        "recommendation": recommendation,
        "text": resume_text[:1200]
    }

    session["result"] = result

    return redirect(url_for("dashboard"))


# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    result = session.get("result")

    if not result:
        return "No data found. Upload resume first."

    return render_template("dashboard.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)