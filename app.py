import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Animal Detection AI", page_icon="🐾", layout="centered")

st.title("🐾 Animal Detection AI")
st.write("Animal ki photo upload karein aur AI se prediction hasil karein.")
st.info("Supported classes: Cat, Lion, Parrot, Sparrow, Tiger")

@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model("animal_detection_model.keras")
    with open("animal_classes.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names

try:
    model, class_names = load_model_and_classes()
except Exception as e:
    st.error("Model ya class-label file load nahi hui. Dono files app.py ke saath honi chahiye.")
    st.code(str(e))
    st.stop()

uploaded_file = st.file_uploader(
    "Animal ki image upload karein",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Detect Animal", type="primary", use_container_width=True):
            resized_image = image.resize((160, 160))
            image_array = np.array(resized_image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            scores = model.predict(image_array, verbose=0)[0]
            best_index = int(np.argmax(scores))
            predicted_animal = class_names[best_index]
            score_percent = float(scores[best_index]) * 100

            st.success(f"Predicted Animal: {predicted_animal.upper()}")
            st.metric("Model Score", f"{score_percent:.2f}%")

            st.subheader("All class scores")
            for i, name in enumerate(class_names):
                score = float(scores[i])
                st.write(f"{name.title()}: {score * 100:.2f}%")
                st.progress(score)

            st.caption(
                "Note: Model score correctness ki guarantee nahi hai. "
                "Is model ko sirf 5 animal classes par train kiya gaya hai, "
                "aur training images kam hain; is liye ghalat prediction mumkin hai."
            )
    except Exception as e:
        st.error(f"Image process nahi ho saki: {e}")
