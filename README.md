# 🧠 Customer Service Call Analyzer

This project evaluates customer service call interactions using **speech emotion recognition**, **audio transcription**, and **LLM-based feedback generation**. It includes a **Streamlit-based graphical interface** and an automatic **email alert system** for potentially non-compliant or unsatisfactory interactions.

---

## 🎥 Demo

[▶️ Watch demo video](https://drive.google.com/file/d/1FcXwsf70BkoAHAQVXbsIex8OuSFIz4xE/view?usp=sharing)


---

## 🚀 Features

- 🔊 **Transcribe Customer Service Audio** using OpenAI Whisper
- 🎭 **Detect Emotions Per Speaker** using SpeechBrain
- 🧠 **Analyze Sentiment & Quality** using LLaMA 3 (ChatGroq)
- 💌 **Trigger Email Alerts** for flagged customer experiences
- 💬 **Chat with an LLM Assistant** about call performance
- 🖥️ **Streamlit GUI** for a user-friendly experience

---

## 📂 Project Structure

| File                 | Description                                                |
|----------------------|------------------------------------------------------------|
| `main.py`            | Coordinates transcription, emotion detection, and feedback |
| `transcription.py`   | Runs speech-to-text via Whisper                            |
| `speech_brain_app.py`| Runs emotion diarization using SpeechBrain                 |
| `gpt.py`             | Handles LLM querying and prompt formatting                 |
| `mail.py`            | Sends email alerts using SendGrid                          |
| `download_model.py`  | Handles model downloads from Hugging Face                  |
| `gui_final.py`       | Main Streamlit user interface                              |
| `requirements.txt`   | Python dependencies list                                   |

---

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/Pragateeshwaran/Customer-Service-Intelligence.git
cd Customer-Service-Intelligence

# 2. Install dependencies
pip install -r requirements.txt
````

---

## 🧪 Usage

### ▶️ Run GUI

```bash
streamlit run gui_final.py
```

### 👇 In GUI:

* Upload a `.wav` or `.mp3` file
* The app will:

  * Transcribe the conversation
  * Analyze speaker-wise emotions
  * Score customer satisfaction
  * Trigger email alerts if needed
  * Let you chat with an LLM about the call

---

## 🔐 Environment Variables

Create a `.env` file in the root with:

```env
GROQ_API_KEY=your_groq_api_key
SENDGRID_API_KEY=your_sendgrid_api_key
```

These are required for ChatGroq LLM access and SendGrid email sending.

---

## 🧠 Models Used

* 🎧 Whisper (`openai/whisper-large-v3`)
* 🎭 Emotion Diarization (`speechbrain/emotion-diarization-wavlm-large`)
* 🧠 LLM (`llama3-70b-8192` via ChatGroq API)

---

## 📦 Requirements

Python libraries used:

* `streamlit`
* `langchain`
* `speechbrain`
* `transformers`
* `sendgrid`
* `torch`
* `python-dotenv`

Install them with:

```bash
pip install -r requirements.txt
```

---

## 📧 Email Alerts

* If the analysis detects a negative sentiment or violation,
  an email is sent automatically to the supervisor or relevant authority.
* Email uses the SendGrid API (configured via `.env`)

---

## ✅ Notes

* The project can handle long audio files and multiple speakers.
* GPU acceleration is recommended for Whisper and Emotion Diarization.

---
 

Let me know if you'd like to:
- Add badges (e.g. Python version, License, Built With)
- Include screenshots or a hosted version
- Link a specific paper or repo for the LLM

I'm happy to tweak it to match your style!
```
