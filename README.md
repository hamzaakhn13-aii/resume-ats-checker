# 📄 AI Resume ATS Checker

A Streamlit app that analyzes a resume with **Google Gemini Flash** and returns:

- an overall **ATS score (0-100)** plus a breakdown (keywords, content impact, formatting, structure, readability)
- strengths and weaknesses
- prioritized, actionable **improvements** with example rewrites
- found vs. missing **keywords** (optionally matched against a pasted job description)

Supports **PDF, DOCX and TXT** resumes.

**Target field selector:** choose *General* or *Cybersecurity* in the sidebar. Cybersecurity mode checks for security keywords (SIEM, incident response, threat hunting...), frameworks (NIST, MITRE ATT&CK, ISO 27001), tools (Splunk, Burp Suite, Nmap...) and certifications (Security+, CISSP, CEH, OSCP...). It only credits certifications actually listed on the resume.

> The score is an AI-based estimate. Real ATS software differs between employers, so treat it as guidance.

## Project structure

```
.
├── app.py             # Streamlit app
├── requirements.txt   # Python dependencies
└── README.md
```

## Run locally

1. Get a free Gemini API key: https://aistudio.google.com/apikey
2. Install and run:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Provide your key (pick one):

```bash
# Option A: environment variable
export GEMINI_API_KEY="your-key"          # Windows PowerShell: $env:GEMINI_API_KEY="your-key"

# Option B: Streamlit secrets file (do NOT commit it)
mkdir -p .streamlit
echo 'GEMINI_API_KEY = "your-key"' > .streamlit/secrets.toml

# Option C: just paste the key in the app's sidebar
```

4. Start the app:

```bash
streamlit run app.py
```

## Configuration

| Setting | How | Default |
|---|---|---|
| `GEMINI_API_KEY` | env var or Streamlit secret | none (sidebar input as fallback) |
| `GEMINI_MODEL` | env var, or edit the "Gemini model" box in the sidebar | `gemini-2.5-flash` |

If Google retires or renames a model, change the model name in the sidebar. The list of current model names is at https://ai.google.dev/gemini-api/docs/models

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (never commit your API key).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose your repo, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Click **Deploy**.

## Notes and limitations

- Scanned (image-only) PDFs contain no text and can't be analyzed. Use a text-based PDF or DOCX.
- Resume text is sent to the Gemini API for analysis. Don't upload documents you aren't comfortable sharing with Google.
- Files are limited to 5 MB and resume text to about 30,000 characters.
