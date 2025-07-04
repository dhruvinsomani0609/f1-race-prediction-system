# 🏎️ F1 Race Prediction System

A machine learning-powered system that predicts F1 race outcomes using real-world data from the FastF1 API. This project uses Random Forest models to predict race winners and finishing positions for current F1 drivers.

---

## 💡 Features

* Uses FastF1 to gather real race + qualifying data
* Trains models using Random Forest Classifier & Regressor
* Predicts race winner probabilities and finishing positions
* Filters only current 2025 drivers
* Saves results to CSV

---

## 📂 Project Structure

```
F1 Project/
├── f1_cache/                   # FastF1 HTTP cache
│   ├── 2024/
│   └── 2025/
├── f1_2025_predictions.csv    # Output predictions
├── f1_predictor.py            # Main code file
├── requirements.txt           # Dependencies
└── README.md                  # You're here!
```

---

## ✨ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/dhruvinsomani693/f1-prediction-system.git
cd f1-prediction-system
```

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate       # For Mac/Linux
venv\Scripts\activate          # For Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> Make sure you have Python 3.8 or later installed.

---

## 🚀 How to Run

### Step 1: Start the Prediction Script

```bash
python f1_predictor.py
```

This will:

* Collect race data for 2024 and completed races in 2025
* Train the models
* Predict results only for official 2025 drivers
* Output to `f1_2025_predictions.csv`

---

## 🥇 Example Output (Top 3 Prediction)

```
🏎️ 2025 Race Winner Predictions:
============================================================
| 🥇 Rank | Driver Name             | Team            | 🧠 Win Probability | 🏁 Predicted Finish |
|       1 | **Lando Norris**        | McLaren         |               0.07 |                9.97 |
|       2 | **Oscar Piastri**       | McLaren         |               0.07 |                9.87 |
|       3 | **Charles Leclerc**     | Ferrari         |               0.05 |                9.32 |
|       4 | **Lewis Hamilton**      | Ferrari         |               0.04 |                8.87 |
|       5 | **Max Verstappen**      | Red Bull Racing |               0.03 |                9.30 |
|       6 | **George Russell**      | Mercedes        |               0.02 |                9.23 |
|       7 | **Liam Lawson**         | Racing Bulls    |               0.00 |                8.75 |
|       8 | **Gabriel Bortoleto**   | Stake F1 Team   |               0.00 |                8.96 |
|       9 | **Pierre Gasly**        | Alpine          |               0.00 |                8.48 |
|      10 | **Fernando Alonso**     | Aston Martin    |               0.00 |                8.41 |
|      11 | **Alexander Albon**     | Williams        |               0.00 |                9.47 |
|      12 | **Oliver Bearman**      | Haas            |               0.00 |                8.28 |
|      13 | **Lance Stroll**        | Aston Martin    |               0.00 |                8.84 |
|      14 | **Esteban Ocon**        | Haas            |               0.00 |                8.56 |
|      15 | **Yuki Tsunoda**        | Red Bull Racing |               0.00 |                9.41 |
|      16 | **Carlos Sainz**        | Williams        |               0.00 |                9.58 |
|      17 | **Nico Hulkenberg**     | Stake F1 Team   |               0.00 |                9.11 |
|      18 | **Jack Doohan**         | Alpine          |               0.00 |               17.75 |
|      19 | **Andrea Kimi Antonelli | Mercedes        |               0.00 |                4.00 |

```

---

## 🔧 Tech Stack

* Python 3.8+
* [FastF1](https://theoehrly.github.io/Fast-F1/) for data
* pandas, numpy, matplotlib, seaborn
* scikit-learn for ML models

---

## 🪄 Future Improvements

* Add live race prediction using telemetry
* Use more complex models (e.g., XGBoost, LSTM)
* Improve driver/team encoding
* Build a Streamlit dashboard

---

---

## 🚀 License

This project is licensed under the MIT License. See the LICENSE file for details.
