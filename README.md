# 🔢 Handwritten Number Sequence Recognizer

A fast, lightweight web application that recognizes handwritten single digits and multi-digit number sequences (e.g. `123`, `42`, `2026`) using a Convolutional Neural Network (CNN) trained on MNIST, OpenCV morphological segmentation, and a modern **FastAPI** backend with an interactive **HTML5/Vanilla CSS/JS** frontend.

Built and optimized specifically for **Render free tier deployment** (ultra-low memory footprint < 200MB RAM, sub-second latency, zero bloat).

---

## Features

- **✍️ Interactive Drawing Pad** — smooth anti-aliased HTML5 canvas with multi-touch support for tablets & phones, stroke thickness adjustments (fine, medium, bold), stroke undo, and clear actions.
- **📤 Drag & Drop Image Upload** — upload photos, scans, or drawings of single- or multi-digit numbers.
- **📖 Natural Reading Order Segmentation** — detects and groups numbers Top-to-Bottom (line-by-line) and Left-to-Right using OpenCV contour analysis and adaptive line clustering.
- **🎯 MNIST Center-of-Mass Preprocessing** — extracts individual digit crops, pads them to square, scales to 20px, and aligns by center-of-mass into the standard 28x28 grayscale format.
- **📊 Interactive Probability Drawer** — click "View Probabilities" on any recognized digit card to explore full softmax probability distributions (0 to 9) with winner highlighting.
- **🚀 Render-Optimized** — powered by FastAPI + Uvicorn + `tensorflow-cpu` + `opencv-python-headless`. No heavy frontend framework memory overhead!

---

## Project Structure

```
digit-recognizer/
├── app.py                  # FastAPI asynchronous backend & inference pipeline
├── train_model.py          # Trains the CNN and saves digit_model.keras
├── utils.py                # Image segmentation, MNIST preprocessing & base64 helpers
├── requirements.txt        # Lightweight dependencies (FastAPI, Uvicorn, TensorFlow CPU)
├── digit_model.keras       # Pre-trained CNN model (~99.3% test accuracy)
├── render.yaml             # Render Blueprint configuration for 1-click deployment
├── templates/
│   └── index.html          # Semantic HTML5 frontend with modern glassmorphism UI
├── static/
│   ├── css/
│   │   └── style.css       # Vanilla CSS design system & micro-animations
│   └── js/
│       └── app.js          # Interactive drawing pad, drag & drop, and async prediction calls
├── .gitignore
└── README.md
```

---

## How Multi-Digit Recognition Works

1. **Polarity Detection & Noise Filtering**: Automatically detects light-on-dark or dark-on-light backgrounds and thresholds background noise using Otsu binarization.
2. **Left-to-Right & Top-to-Bottom Digit Segmentation**: Uses OpenCV morphological operations and contour analysis to isolate individual digit bounding boxes, merging multi-stroke digit parts and grouping into natural reading lines.
3. **MNIST Center-of-Mass Preprocessing**: Each isolated digit crop is padded to square, resized preserving aspect ratio, and aligned by center of mass into a standard 28x28 array.
4. **CNN Sequence Prediction**: The trained TensorFlow CNN predicts each digit sequentially, producing the full sequence (e.g., `"2026"`) and confidence scores for each position.

---

## 1. Run Locally

### Prerequisites
- Python 3.9–3.11 installed
- `pip` available

### Steps

```bash
# 1. Clone your repo
git clone https://github.com/naravulaprudhvi/digit_recognizer.git
cd digit_recognizer

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app (model is already included as digit_model.keras)
uvicorn app:app --reload --port 8000
```

Open your browser at `http://localhost:8000`.

---

## 2. Deploy to Render

### Option A: Using Render Blueprint (Automatic — Recommended)

1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Configure FastAPI app for Render"
   git push -u origin main
   ```
2. Open your [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** > **Blueprint**.
4. Connect your repository: `https://github.com/naravulaprudhvi/digit_recognizer`.
5. Render will automatically read [`render.yaml`](render.yaml) and configure:
   - **Build Command:** `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Python Version:** `3.11.9`
6. Click **Apply**.

---

### Option B: Manual Web Service on Render

1. Go to [Render Dashboard](https://dashboard.render.com/) > **New +** > **Web Service**.
2. Connect your GitHub repository (`naravulaprudhvi/digit_recognizer`).
3. Fill in the following settings:
   - **Name:** `digit-recognizer` (or your choice)
   - **Language:** `Python 3`
   - **Branch:** `main`
   - **Region:** Any (e.g., Oregon, Frankfurt)
   - **Build Command:**
     ```bash
     pip install --upgrade pip && pip install -r requirements.txt
     ```
   - **Start Command:**
     ```bash
     uvicorn app:app --host 0.0.0.0 --port $PORT
     ```
   - **Instance Type:** `Free`
4. In **Advanced** > **Environment Variables**, add:
   - `PYTHON_VERSION` = `3.11.9`
5. Click **Create Web Service**.

Once deployed, Render provides an `https://<your-app-name>.onrender.com` URL to access your live application!

---

## Tech Stack

| Layer | Tool |
|---|---|
| Model | TensorFlow / Keras CNN (~99.3% MNIST test accuracy) |
| Backend | FastAPI + Uvicorn (Asynchronous, lightweight REST API) |
| Frontend | Semantic HTML5 + Vanilla CSS (Glassmorphism design system) |
| Canvas & Logic | HTML5 Canvas API + Vanilla JavaScript (Multi-touch support, Undo/Redo) |
| Vision & Preprocessing | OpenCV (`opencv-python-headless`) + SciPy + NumPy + Pillow |
| Cloud Hosting | Render Web Service / Blueprint (`render.yaml`) |

---

**Author:** Naravula Prudhvi Sri Bhanu Vivek
