# 🔢 Handwritten Number Sequence Recognizer

A web app that recognizes handwritten single digits and multi-digit number sequences (e.g. `123`, `42`, `2026`) using a Convolutional Neural Network (CNN) trained on the MNIST dataset and automated computer vision segmentation.

Choose between two ways to give it numbers:

- **✍️ Write Now** — draw single or multi-digit numbers side-by-side on a wide in-browser pad
- **📤 Upload an Image** — upload a photo/scan of handwritten numbers

The app automatically segments the input image left-to-right into individual digits, preprocesses each digit to match the MNIST standard (28x28 grayscale, centered by center-of-mass, normalized), and predicts the full number sequence with individual digit breakdown cards.

---

## Project structure

```
digit-recognizer/
├── app.py                  # Streamlit web app (UI + sequence prediction)
├── train_model.py          # Trains the CNN and saves digit_model.keras
├── utils.py                # Image segmentation & MNIST preprocessing
├── requirements.txt        # Python dependencies (TensorFlow CPU, OpenCV Headless, Streamlit)
├── digit_model.keras       # Pre-trained CNN model (~99.3% test accuracy)
├── render.yaml             # Render Blueprint configuration for automated deploy
├── Procfile                # Web process start command
├── runtime.txt             # Python runtime version for Render / cloud platforms
├── .python-version         # Python 3.11.9 pin
├── .streamlit/
│   └── config.toml         # Streamlit headless server & theme configuration
├── .gitignore
└── README.md
```

## How Multi-Digit Recognition Works

1. **Polarity Detection & Noise Filtering**: Automatically detects light-on-dark or dark-on-light backgrounds and thresholds background noise.
2. **Left-to-Right & Top-to-Bottom Digit Segmentation**: Uses OpenCV morphological operations and contour analysis to isolate individual digit bounding boxes ordered into natural reading order.
3. **MNIST Center-of-Mass Preprocessing**: Each isolated digit crop is padded to square, resized preserving aspect ratio, and aligned by center of mass into a standard 28x28 array.
4. **CNN Sequence Prediction**: The trained TensorFlow CNN predicts each digit sequentially, producing the full sequence (e.g., `"2026"`) and confidence scores for each position.

---

## 1. Run it locally

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

# 4. Run the app (model is already included as digit_model.keras)
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

---

## 2. Deploy to Render

### Option A: Using Render Blueprint (Automatic — Recommended)

1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Prepare project for Render deployment"
   git push -u origin main
   ```
2. Open your [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** > **Blueprint**.
4. Connect your repository: `https://github.com/naravulaprudhvi/digit_recognizer`.
5. Render will automatically read [`render.yaml`](render.yaml) and configure the build and start commands. Click **Apply**.

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
     streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
     ```
   - **Instance Type:** `Free`
4. In **Advanced** > **Environment Variables**, add:
   - `PYTHON_VERSION` = `3.11.9`
5. Click **Create Web Service**.

Once deployed, Render provides an `https://<your-app-name>.onrender.com` link to access your live app!

---

## Tech stack

| Layer | Tool |
|---|---|
| Model | TensorFlow / Keras CNN (~99.3% MNIST accuracy) |
| Dataset | MNIST (60,000 train / 10,000 test images) |
| Web UI | Streamlit |
| Drawing pad | streamlit-drawable-canvas (560px interactive canvas) |
| Segmentation & Preprocessing | OpenCV (`opencv-python-headless`) + SciPy + NumPy + Pillow |
| Deployment | Render Web Service / Blueprint (`render.yaml`) |

---

**Author:** Naravula Prudhvi Sri Bhanu Vivek

