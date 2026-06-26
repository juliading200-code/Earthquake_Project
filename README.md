# Earthquake FNO Educational App

This project includes a standalone Streamlit application that visualizes earthquake-driven structural response and compares ground-truth simulation outputs against available Fourier Neural Operator (FNO) predictions.

## What the app demonstrates

1. Earthquake selection from three representative records (small, medium, large).
2. Ground-motion time history plotting.
3. Animated six-story building response under selected ground motion.
4. Floor-by-floor comparison of actual response vs FNO prediction.
5. Error metrics (overall and by floor).
6. Automatic discovery of MATLAB variables and their array shapes.

## Expected folder structure

Place the following files in the same folder:

- `streamlit_app.py`
- `requirements.txt`
- `dataset_EQ.mat`
- `train_test_index.mat`
- Optional prediction files:
  - `Earthquake_responses_test.mat`
  - `Earthquake_responses_test_fno.mat`

Current project already contains the required dataset and index files.

## Installation

From the project folder:

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

   pip install -r requirements.txt

3. Run the app:

   streamlit run streamlit_app.py

## Data loading details

The app **does not** parse MAT files as text. It reads data directly using:

- `scipy.io.loadmat`
- `numpy`
- `pandas`

Loaded variables:

- From `dataset_EQ.mat`:
  - `ground_motion`
  - `displacement`
  - `time`
- From `train_test_index.mat`:
  - `test`
  - `train`
- From prediction file (if present):
  - `y_pred`

### Representative earthquake selection

The app uses `train_test_index.mat` test records and computes peak ground acceleration (PGA) for each test earthquake. It then automatically chooses representative records by PGA quantiles:

- Earthquake A: small
- Earthquake B: medium
- Earthquake C: large

### Notes on prediction channels

If the available FNO prediction file has one output channel instead of six, the app clearly reports this and duplicates the single predicted channel for educational side-by-side visualization across six floors.

## Educational explanation in app

The sidebar includes plain-language explanations:

- **Ground Truth**: response from nonlinear structural simulation.
- **FNO Prediction**: response estimated by the neural operator.
- Better overlap between curves indicates better prediction quality.
