# BrowserExtension# ProtectU: Browser Extension with Phishing Detection

ProtectU is a browser extension project designed to detect phishing URLs using machine learning. The project consists of a backend for model training and inference, and a frontend browser extension built with React and Vite.

---

## Project Structure
. ├── .gitignore ├── Dockerfile ├── README.md ├── requirements.txt ├── backend/ │ ├── .env │ ├── alerts.json │ ├── app.py │ ├── logs.log │ ├── datasets/ │ │ ├── combineCSV.py │ │ ├── eda_summary.csv │ │ ├── eda.py │ │ ├── Phishing_URLs.csv │ │ └── URL_dataset.csv │ ├── libs/ │ │ ├── init.py │ │ ├── ExtractFunc.py │ │ ├── FeaturesExtract.py │ │ ├── inference.py │ │ ├── t2.py │ │ ├── test.py │ │ ├── train_model.py │ │ ├── train.py │ │ └── pycache/ │ ├── model/ │ │ └── pca_model.pkl │ └── models/ │ └── phishing_model.pkl └── protectu-extension/ ├── eslint.config.js ├── index.html ├── package.json ├── README.md ├── vite.config.js ├── public/ └── src/



---

## Backend

- **Purpose:** Handles data processing, model training, and inference for phishing URL detection.
- **Key Files:**
  - [`backend/app.py`](backend/app.py): Main API server (likely Flask or FastAPI).
  - [`backend/libs/train.py`](backend/libs/train.py): Model training script.
  - [`backend/libs/inference.py`](backend/libs/inference.py): Model inference logic.
  - [`backend/datasets/`](backend/datasets/): Contains datasets and data processing scripts.
  - [`backend/models/`](backend/models/): Stores trained model files.

- **Setup:**
  1. Install dependencies:
      ```sh
      pip install -r requirements.txt
      ```
  2. Run the backend server:
      ```sh
      cd backend
      python app.py
      ```

- **Model Training:**
  - Use [train.py](http://_vscodecontentref_/4) or [train_model.py](http://_vscodecontentref_/5) to train the phishing detection model.
  - Datasets are located in [datasets](http://_vscodecontentref_/6).

---

## Frontend (Browser Extension)

- **Purpose:** User-facing browser extension built with React and Vite.
- **Key Files:**
  - [src](http://_vscodecontentref_/7): React source code.
  - [index.html](http://_vscodecontentref_/8): Main HTML entry point.
  - [package.json](http://_vscodecontentref_/9): Frontend dependencies and scripts.

- **Setup:**
  1. Install dependencies:
      ```sh
      cd protectu-extension
      npm install
      ```
  2. Start the development server:
      ```sh
      npm run dev
      ```

---

## Docker

- **Purpose:** Provides a reproducible environment for running the backend.
- **Usage:**
  1. Build the Docker image:
      ```sh
      docker build -t protectu-backend .
      ```
  2. Run the container:
      ```sh
      docker run -p 5000:5000 protectu-backend
      ```

---

## Requirements

- Python 3.8+
- Node.js 16+
- Docker (optional, for containerized backend)

---

## Contributing

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request.

---

## License

This project is licensed under the MIT License.

---

## Authors

- [Avram Lavinia]
