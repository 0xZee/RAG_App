import streamlit as st
from groq import Groq
from PIL import Image
import io
import base64

# Load secrets from st.secrets.toml
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Page configuration
st.set_page_config(
    page_title="Groq OCR",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description in main area
st.subheader("🧾 OCR - TEXT EXTRACTION APP", divider='orange')

    
# Move upload controls to sidebar
with st.sidebar:
    st.caption("🔗  github.com/0xZee - 2025")
    st.divider()
    uploaded_file = st.file_uploader("📂 Reçu, Facture, Copie, image .. 📸", type=['png', 'jpg', 'jpeg'])

    # Add clear button to top right
    if st.sidebar.button("♻️ Effacer / Nouveau", use_container_width=True):
        if 'ocr_result' in st.session_state:
            del st.session_state['ocr_result']
        st.rerun()

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Votre Image", use_container_width=True)

        if st.button("Extraire le Texte 💫", type="primary", use_container_width=True):
            with st.spinner("Analyse et Traitement..."):
                try:
                    # Convert uploaded image to base64
                    image_bytes = uploaded_file.getvalue()
                    base64_image = base64.b64encode(image_bytes).decode('utf-8')

                    client = Groq(api_key=GROQ_API_KEY)

                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": """Analyze the text in the provided image. Extract all readable content
                                        and present it in a structured Markdown format that is clear, concise, emojies-styled
                                        and well-organized. Ensure proper formatting (e.g., headings, lists, or
                                        code blocks) as necessary to represent the content effectively."""},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}",
                                        },
                                    },
                                ],
                            }
                        ],
                        model="meta-llama/llama-4-scout-17b-16e-instruct",
                        temperature=1,
                        max_completion_tokens=1024,
                        top_p=1,
                        stream=False,
                        stop=None,
                    )

                    response_text = chat_completion.choices[0].message.content
                    st.session_state['ocr_result'] = response_text
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

# Main content area for results
if 'ocr_result' in st.session_state:
    result_text = st.session_state['ocr_result']
    st.markdown(result_text)
    # download ocr_result as text button here :
    st.download_button(
        label="⬇️ Télécharger le Résultat (TXT)",
        data=result_text,
        file_name="ocr_result.txt",
        mime="text/plain",
        type="primary",
        use_container_width=True
    )

else:
    st.write("🧾 Extraction de :orange[Texte Structuré] à partir d'images (:orange[Reçu, Facture, Photo-copie]..) assité par l'intelligence artificielle visuelle. ")
    st.warning("◀ Uploader une image and cliquer 'Extracire Texte' pour commencer.")
