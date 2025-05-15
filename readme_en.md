# 🧠 Multiclass Time Series Classification with Reservoir Computing

## 🔍 Context

As part of the **AI4Industry** initiative ([ai4industry.fr](https://www.ai4industry.fr)), we had the opportunity to participate in a Use Case proposed by **CATIE**, a technology center specializing in research, development, and innovation in information and electronics.
More info: [catie.fr](https://www.catie.fr)

🎯 **Objective**: Develop a system to **automatically recognize digits drawn using a Z-Motion sensor**.

Sensor used: [Z-Motion - 6Tron](https://6tron.io/use_case/demo-z-motion-ble-imu)

---

## 🧩 Problem

**Multiclass classification of time series data.**

Three groups collected data by drawing digits from 0 to 9 using the Z-Motion sensor. The digits were drawn in 2D (on tables/walls) and 3D (in space). The data was recorded using a Linux console and a Python script provided by CATIE.

We used **configurations 1 and 3** out of the three available:

### 🔧 Configuration 1

* `t`: Timestamp (not always regular)
* `raw acceleration x/y/z`: Linear acceleration on x, y, and z axes
* `magnetic field x/y/z`: Magnetic fields on x, y, and z axes

### 🔧 Configuration 3

* `t`: Timestamp
* `raw acceleration x/y/z`: Linear acceleration
* `quaternion w/x/y/z`: Quaternion components representing 3D orientation

---

## 🧪 Methodology

1. **Data Acquisition & Preprocessing**

   * Reconstruction of digit trajectories from accelerations.
   * Compression of time series and extraction of statistical features for classification.

2. **Classical Modeling Approaches**

   * Models tested: Logistic Regression, Decision Tree, Gradient Boosting
   * Comparison of precision and recall

3. **Reservoir Computing (ESN)**

   * Use of raw time series with an **Echo State Network** (ESN)
   * Time step-by-step prediction ✨

4. **Streamlit App**

   * Interactive visualization of predictions over time ⏳

---

## 📁 Project Structure

```bash
📁 app/
 ┣ 📁 ressources/         # Models, GIF, .sav files
 ┣ 📄 app.py              # Streamlit app
 ┣ 📄 utils.py            # Utility functions
 ┣ 📄 requirements.txt    # Dependencies
 ┣ 📄 debug_app.ipynb     # Debug notebook
📁 data/                  # Data files
📄 feature_engineering.ipynb
📄 feature_importance.ipynb
📄 Visualisation.ipynb
📄 realtime.ipynb
📄 readme.md
📄 readme_en.md
```

---

## 📓 Notebook Descriptions

* **Visualisation.ipynb** 🛰️
  3D display of sensor movement during digit tracing. Handles irregular timestamps using sliding mean.

* **feature\_importance.ipynb** 🧮
  ANOVA analysis + feature importance comparison (Logistic Regression & Random Forest).
  ✅ Precision: **0.94** | Recall: **0.93**

* **feature\_engineering.ipynb** 🛠️
  Feature engineering and model optimization (LogisticRegression, DecisionTree, RandomForest).
  ✅ Score: **0.96** | Time: **0.0044s**

* **reservoir\_computing** 🧠
  Training a simple ESN (Echo State Network), optimizing neuron count, leaking rate, spectral radius.
  ✅ Dataset-wide score: **0.8137**

* **reservoir\_deep** 🧬
  Complex reservoir architectures: Hierarchical ESN, Deep ESN, multi-input setups

* **realtime.ipynb** ⚙️
  Fine-tuning of the Random Forest model for real-time prediction

---

## 🚀 Streamlit App

### Run the app locally:

```bash
# 1. Clone the repository
$ git clone https://github.com/RCaz33/Time-series_Multiclass-classification_Reservoir-computing-INRIA
$ cd Time-series_Multiclass-classification_Reservoir-computing-INRIA

# 2. Create and activate a virtual environment
$ python3 -m venv .venv
$ source .venv/bin/activate

# 3. Install dependencies and launch the app
$ pip install streamlit
$ streamlit run app/app.py
```

### Features 🖥️

* Select a recorded acquisition
* Step-by-step prediction display by **RandomForestClassifier** and **ESN**

![App screenshot](screenshot_app.png)

---

For questions or contributions, feel free to open an issue or a pull request! 🚀
