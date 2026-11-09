# LC3 Strength Estimator

Web app of the hierarchical Bayesian Bolomey model for the compressive strength of limestone calcined clay cement (LC3)
mixes, fitted to 437 verified mix–age observations from ten published studies (six research groups).

- **Single mix:** strength–age curve with an 80% interval for a new laboratory, and anchored on a Portland cement control
  (28 d or 7 d) from the same laboratory.
- **Batch validation:** upload a CSV of mixes with measured strengths; get predictions, MAE/MAPE/bias/coverage per group,
  and a measured-vs-predicted plot. A template and the training data (as an in-sample example) can be downloaded in the app.
- **Model and reliability:** equations, data domain and the accuracy measured on withheld studies.

Research prototype for screening and validation, not a design tool. Model and validation: M. U. Gazi and X. Shi,
University of Miami; manuscript under review (2026).

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Access code
If `APP_PASSWORD` is set in Streamlit secrets (`.streamlit/secrets.toml` locally, or the app's Secrets on Streamlit
Community Cloud), the app asks for it; without the secret the app is open.

## Files
| File | Content |
|---|---|
| `app.py` | interface |
| `lc3_model.py` | model equations, predictions, domain checks, batch helpers |
| `m2_post_primary.npz` | 4000 posterior draws of the all-data fit (PyMC, NUTS) |
| `example_validation.csv` | the 437 training observations in the batch format, with each study's 28-d control |
