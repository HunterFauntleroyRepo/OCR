# OCR Project (CRNN + CTC)

This is a **Python-based OCR (Optical Character Recognition) project** built with PyTorch.
It recognizes **full ASCII characters** in synthetic single-line text images.

---

## 📁 Project Structure

ocr/
├── data/
│ ├── train/ ← generated training images
│ └── val/ ← generated validation images
# OCR (CRNN + CTC)

Lightweight single-line OCR using a CRNN (CNN → BiLSTM → Linear) with CTC loss,
implemented in Python with PyTorch. The project trains and evaluates on synthetic
single-line images and supports full ASCII recognition.

---

**Quick links**
- Project: OCR (CRNN + CTC)
- Language: Python
- Framework: PyTorch

---

## Project layout

- data/
	- train/     # generated training images
	- val/       # generated validation images
- fonts/       # put .ttf font files here
- src/
	- charset.py
	- dataset.py
	- generate_data.py
	- model.py
	- train.py
	- infer.py
- main.py
- requirements.txt

---

## Setup

1. (Optional) Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Add at least one TTF font to the `fonts/` folder (e.g., `arial.ttf`).

---

## Usage — Quickstart

Run the main menu:

```powershell
python main.py
```

Menu options:
- Generate synthetic dataset — fills `data/train` and `data/val`
- Train OCR model — trains a CRNN and saves the checkpoint (e.g., `models/crnn_ocr.pth`)
- Run inference — predict text from a single image
- Exit

---

## Scripts and responsibilities

- `src/generate_data.py` — create synthetic single-line images (default: 32×128 px,
	random ASCII text length 3–10). Adjust parameters to change image size and text length.
- `src/dataset.py` — dataset loader and preprocessing pipeline.
- `src/model.py` — CRNN model definition (CNN -> BiLSTM -> Linear).
- `src/train.py` — training loop using CTC loss.
- `src/infer.py` — model loading + greedy decoding for prediction.
- `src/charset.py` — character set and encoding helpers.

---

## Tips & notes

- Increase the number of generated samples in `generate_data.py` to improve accuracy.
- To support longer text, increase `IMG_WIDTH` in `generate_data.py` and `dataset.py`.
- Adding more fonts to `fonts/` improves model generalization across typefaces.
- Training uses GPU if available, otherwise CPU.
- This project targets single-line OCR; multi-line OCR requires different data and
	model architecture.

---

## Inference requirements

- Input should be a single-line, grayscale image roughly matching the training size
	(default 32×128 px). Preprocess using the same transforms as the training dataset.

---

## License

MIT License

---

If you'd like, I can also:
- add a short example showing how to run `src/generate_data.py` and `src/train.py`,
- or create a minimal `examples/` folder with a sample font and a tiny dataset.
