# Animal Detection AI — Streamlit

## Files needed
Keep these four files in the same folder:
- `app.py`
- `requirements.txt`
- `animal_detection_model.keras` (download from Kaggle)
- `animal_classes.json` (download from Kaggle)

## Run locally
1. Install Python 3.10 or 3.11 if possible.
2. Open Command Prompt in this folder.
3. Run:
   `python -m pip install -r requirements.txt`
4. Run:
   `python -m streamlit run app.py`

The app usually opens at http://localhost:8501.

## Publish online with Streamlit Community Cloud
1. Upload all four files to a GitHub repository.
2. Go to https://share.streamlit.io/ and sign in with GitHub.
3. Click "Create app", choose your repository and select `app.py`.
4. Deploy the app.

If GitHub rejects the `.keras` model because it is too large, use Git LFS or store the model in an external model repository and update `app.py` to download it at startup.

## Important model limitation
The current model was trained with only 2 images per class (10 total), so predictions may be unreliable. Collect many more varied images and evaluate on separate validation/test images before relying on it.
