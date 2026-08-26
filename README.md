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
├── requirements.txt        # Python dependencies (TensorFlow, OpenCV, Streamlit, etc.)
├── digit_model.keras       # Trained model (created after you run train_model.py)
├── .streamlit/config.toml  # App theme
├── .gitignore
└── README.md
```

## How Multi-Digit Recognition Works

1. **Polarity Detection & Noise Filtering**: Automatically detects light-on-dark or dark-on-light backgrounds and thresholds background noise.
2. **Left-to-Right Digit Segmentation**: Uses OpenCV morphological operations and contour analysis to isolate individual digit bounding boxes ordered from left to right.
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
git clone https://github.com/<your-username>/digit-recognizer.git
cd digit-recognizer

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (takes ~5-10 minutes on CPU, downloads MNIST automatically)
python train_model.py

# 5. Run the app
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

---

## Tech stack

| Layer | Tool |
|---|---|
| Model | TensorFlow / Keras CNN (~99.3% MNIST accuracy) |
| Dataset | MNIST (60,000 train / 10,000 test images) |
| Web UI | Streamlit |
| Drawing pad | streamlit-drawable-canvas (560px wide interactive pad) |
| Segmentation & Preprocessing | OpenCV + SciPy + NumPy + Pillow |

---

**Author:** Naravula Prudhvi Sri Bhanu Vivek

