import streamlit as st
import boto3
import numpy as np
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image
import json

# AWS clients
s3 = boto3.client('s3', region_name='us-east-1')
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
bucket = 'potato-disease-predictor'
model_key = 'model/potato_model.h5'
classes = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___Healthy']

@st.cache_resource
def load_model():
    try:
        obj = s3.get_object(Bucket=bucket, Key=model_key)
        model_bytes = BytesIO(obj['Body'].read())
        model = load_model(model_bytes)
        st.success("✅ DL Model loaded from S3!")
        return model
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        return None

def classify_disease(model, img):
    img_resized = img.resize((224, 224)).convert('RGB')
    img_array = np.array(img_resized) / 255.0
    arr = np.expand_dims(img_array, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    pred_class = classes[np.argmax(preds)]
    confidence = np.max(preds) * 100
    return pred_class, confidence

def generate_treatment(disease):
    disease_name = disease.replace('Potato___', '')
    prompt = f"""You are an expert agronomist. For potato '{disease_name}', generate a concise treatment plan in bullet points: immediate cure, prevention, recovery time. Under 200 words."""
    
    body = json.dumps({
        "prompt": prompt,
        "maxTokens": 300,
        "temperature": 0.5,
        "topP": 0.9
    })
    
    try:
        response = bedrock.invoke_model(
            modelId="amazon.nova-lite-v1:0",
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        response_body = json.loads(response['body'].read())
        treatment = response_body['completions'][0]['data']['text'].strip()
        return treatment
    except Exception as e:
        return f"❌ GenAI failed: {e}\nFallback: Consult expert for {disease_name}."

st.title("🌱 GenAI Potato Disease Advisor")
st.markdown("Upload leaf → Classify → Nova Lite Treatment")

model = load_model()
if model is None:
    st.stop()

uploaded_file = st.file_uploader("📁 Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded", use_column_width=True)
    
    if st.button("🔍 Analyze", type="primary"):
        with st.spinner("Processing..."):
            pred_class, confidence = classify_disease(model, image)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.error(f"**Diagnosis:** {pred_class}")
                st.metric("Confidence", f"{confidence:.1f}%")
            
            with col2:
                st.info("**Treatment Plan:**")
                treatment = generate_treatment(pred_class)
                st.markdown(treatment)

with st.sidebar:
    st.header("🛠️ Pipeline")
    st.write("- DL Classification from S3")
    st.write("- Nova Lite for Treatments")
    st.write("- Deployed on EC2")
