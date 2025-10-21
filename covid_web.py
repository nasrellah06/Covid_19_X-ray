## streamlit run "C:\Users\Arrich Nasrellah\Desktop\Première année\PROJET_FA\Covid_Github\covid_web.py" 

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# --- Charger le modèle ---
model = tf.keras.models.load_model("C:\\Users\\Arrich Nasrellah\\Desktop\\Première année\\PROJET_FA\\Covid_Flask\\meilleur_model_covid_RMS.keras")

# Classes
CLASSES = ["Normal", "Covid", "Pneumonia"]

# --- Style CSS ---
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 2rem;
        font-family: 'Inter', sans-serif;
    }
    
    .container {
        max-width: 1200px;
        margin: 0 auto;
        background: white;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1);
        padding: 3rem;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    
    .header {
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .header h1 {
        color: #1f2937;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #2563eb, #1e40af);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metrics {
        display: flex;
        justify-content: center;
        gap: 2rem;
        flex-wrap: wrap;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: #f0f9ff;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border: 1px solid #bae6fd;
        text-align: center;
    }
    
    .metric-card.success {
        background: #f0fdf4;
        border-color: #bbf7d0;
    }
    
    .metric-card.warning {
        background: #fef3c7;
        border-color: #fde68a;
    }
    
    .metric-card strong {
        color: #1e40af;
        display: block;
        margin-bottom: 0.5rem;
    }
    
    .metric-card.success strong {
        color: #166534;
    }
    
    .metric-card.warning strong {
        color: #92400e;
    }
    
    .metric-card span {
        color: #6b7280;
        font-size: 0.9rem;
    }
    
    .section {
        margin: 2rem 0;
    }
    
    .section h2 {
        color: #1f2937;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    
    .info-card {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1px solid #fecaca;
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    
    .info-card.pneumonia {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-color: #fed7aa;
    }
    
    .info-card.normal {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-color: #bbf7d0;
    }
    
    .info-card h3 {
        color: #dc2626;
        margin-bottom: 1.5rem;
        font-size: 1.5rem;
    }
    
    .info-card.pneumonia h3 {
        color: #d97706;
    }
    
    .info-card.normal h3 {
        color: #166534;
    }
    
    .info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
        margin-bottom: 1.5rem;
    }
    
    .info-grid h4 {
        color: #1f2937;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    
    .info-grid p {
        color: #374151;
        line-height: 1.6;
        margin: 0;
    }
    
    .info-grid ul {
        color: #374151;
        margin: 0;
        padding-left: 1.5rem;
    }
    
    .info-grid li {
        margin-bottom: 0.5rem;
    }
    
    .prevention-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
    }
    
    .prevention-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgb(0 0 0 / 0.1);
        border-left: 4px solid #10b981;
    }
    
    .prevention-card.blue {
        border-left-color: #3b82f6;
    }
    
    .prevention-card.orange {
        border-left-color: #f59e0b;
    }
    
    .prevention-card.purple {
        border-left-color: #8b5cf6;
    }
    
    .prevention-card.cyan {
        border-left-color: #06b6d4;
    }
    
    .prevention-card.red {
        border-left-color: #ef4444;
    }
    
    .prevention-card h4 {
        color: #166534;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    
    .prevention-card.blue h4 {
        color: #1e40af;
    }
    
    .prevention-card.orange h4 {
        color: #d97706;
    }
    
    .prevention-card.purple h4 {
        color: #7c3aed;
    }
    
    .prevention-card.cyan h4 {
        color: #0891b2;
    }
    
    .prevention-card.red h4 {
        color: #dc2626;
    }
    
    .prevention-card ul {
        color: #374151;
        margin: 0;
        padding-left: 1.2rem;
        font-size: 0.9rem;
    }
    
    .prevention-card li {
        margin-bottom: 0.3rem;
    }
    
    .message-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 2rem;
        text-align: center;
        border: 1px solid #93c5fd;
    }
    
    .message-box h4 {
        color: #1e40af;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    
    .message-box p {
        color: #374151;
        margin: 0;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .health-message {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        text-align: center;
        border: 1px solid #6ee7b7;
    }
    
    .health-message h4 {
        color: #065f46;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    
    .health-message p {
        color: #374151;
        margin: 0;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    @media (max-width: 768px) {
        .container {
            padding: 1.5rem;
        }
        
        .header h1 {
            font-size: 2rem;
        }
        
        .info-grid {
            grid-template-columns: 1fr;
        }
        
        .prevention-grid {
            grid-template-columns: 1fr;
        }
        
        .metrics {
            flex-direction: column;
            align-items: center;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
    <div class="container">
        <div class="header">
            <h1>🩺 Système de Diagnostic Médical IA</h1>
            <p style="font-size: 1.2rem; color: #6b7280; margin-bottom: 2rem;">
                Analyse automatique de radiographies pulmonaires pour la détection de COVID-19, Pneumonie et cas Normaux
            </p>
            <div class="metrics">
                <div class="metric-card">
                    <strong>🎯 Précision</strong>
                    <span>>95%</span>
                </div>
                <div class="metric-card success">
                    <strong>⚡ Rapidité</strong>
                    <span><2 secondes</span>
                </div>
                <div class="metric-card warning">
                    <strong>🔒 Sécurisé</strong>
                    <span>Données privées</span>
                </div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- Création des onglets ---
tab1, tab2 = st.tabs(["📚 Information Médicale", "🔬 Prédiction IA"])

# --- ONGLET 1: INFORMATION MÉDICALE ---
with tab1:
    st.markdown("""
        <div class="section">
            <h2>🦠 Qu'est-ce que le COVID-19 ?</h2>
            <div class="info-card">
                <h3>🦠 COVID-19 (Coronavirus Disease 2019)</h3>
                <div class="info-grid">
                    <div>
                        <h4>📋 Définition</h4>
                        <p>Le COVID-19 est une maladie infectieuse causée par le coronavirus SARS-CoV-2. 
                        Elle affecte principalement le système respiratoire et peut provoquer des symptômes 
                        allant du simple rhume à des complications respiratoires sévères.</p>
                    </div>
                    <div>
                        <h4>🔍 Symptômes Principaux</h4>
                        <ul>
                            <li>Fièvre et frissons</li>
                            <li>Toux sèche persistante</li>
                            <li>Difficultés respiratoires</li>
                            <li>Perte de goût/odorat</li>
                            <li>Fatigue intense</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="section">
            <h2>🛡️ Comment se Protéger du COVID-19 ?</h2>
            <div class="info-card">
                <h3>🛡️ Mesures de Prévention Essentielles</h3>
                <div class="prevention-grid">
                    <div class="prevention-card">
                        <h4>😷 Port du Masque</h4>
                        <ul>
                            <li>Portez un masque en tissu ou chirurgical</li>
                            <li>Changez-le régulièrement</li>
                            <li>Couvrez le nez et la bouche</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="section">
            <h2>🫁 Qu'est-ce que la Pneumonie ?</h2>
            <div class="info-card pneumonia">
                <h3>🫁 Pneumonie - Infection Pulmonaire</h3>
                <div class="info-grid">
                    <div>
                        <h4>📋 Définition</h4>
                        <p>La pneumonie est une infection qui enflamme les sacs d'air dans un ou les deux poumons. 
                        Les sacs peuvent se remplir de liquide ou de pus, provoquant une toux avec du mucus, 
                        de la fièvre, des frissons et des difficultés respiratoires.</p>
                    </div>
                    <div>
                        <h4>🔍 Symptômes Principaux</h4>
                        <ul>
                            <li>Toux avec mucus</li>
                            <li>Fièvre et frissons</li>
                            <li>Douleur thoracique</li>
                            <li>Essoufflement</li>
                            <li>Fatigue et faiblesse</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="section">
            <h2>✅ Radiographies Pulmonaires Normales</h2>
            <div class="info-card normal">
                <h3>✅ Radiographie Pulmonaire Normale</h3>
                <div class="info-grid">
                    <div>
                        <h4>📋 Caractéristiques</h4>
                        <p>Une radiographie pulmonaire normale montre des poumons clairs sans signes d'infection, 
                        d'inflammation ou d'autres anomalies. Les structures pulmonaires sont bien définies 
                        et il n'y a pas d'opacités anormales.</p>
                    </div>
                    <div>
                        <h4>🔍 Signes de Bonne Santé</h4>
                        <ul>
                            <li>Poumons clairs et bien aérés</li>
                            <li>Pas d'opacités anormales</li>
                            <li>Structures cardiaques normales</li>
                            <li>Pas d'épanchement pleural</li>
                            <li>Symétrie bilatérale</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- ONGLET 2: PRÉDICTION IA ---
with tab2:
    st.markdown("""
        <div style="background: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 16px; padding: 2rem; text-align: center; margin: 2rem 0;">
            <h3 style="color: #1e293b; margin-bottom: 1rem;">📁 Téléchargement d'Image</h3>
            <p style="color: #64748b; margin-bottom: 1.5rem;">
                Veuillez sélectionner une radiographie pulmonaire (format JPG, JPEG ou PNG)
            </p>
            <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; font-size: 0.9rem; color: #64748b;">
                <span>✅ Formats acceptés: JPG, JPEG, PNG</span>
                <span>✅ Taille recommandée: 224x224px</span>
                <span>✅ Qualité: Haute résolution</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📤 Choisissez votre radiographie...", 
        type=["jpg", "jpeg", "png"],
        help="Sélectionnez une image de radiographie pulmonaire pour l'analyse"
    )

    if uploaded_file is not None:
        with st.spinner("Analyse de la radiographie..."):
            img = image.load_img(uploaded_file, target_size=(128, 128))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = model.predict(img_array)
            class_idx = np.argmax(prediction[0])
            confidence = prediction[0][class_idx]

        st.markdown("### 📸 Image Analysée")
        st.image(img, caption="Radiographie pulmonaire analysée", use_container_width=True)

        result_colors = {
            "Normal": {"bg": "#f0fdf4", "border": "#bbf7d0", "text": "#166534", "icon": "✅"},
            "Covid": {"bg": "#fef2f2", "border": "#fecaca", "text": "#dc2626", "icon": "🦠"},
            "Pneumonia": {"bg": "#fffbeb", "border": "#fed7aa", "text": "#d97706", "icon": "🫁"}
        }
        
        result_info = result_colors.get(CLASSES[class_idx], result_colors["Normal"])
        
        st.markdown(f"""
            <div style="
                background: {result_info['bg']}; 
                border: 2px solid {result_info['border']}; 
                border-radius: 16px; 
                padding: 2rem; 
                margin: 2rem 0;
                text-align: center;
                box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
            ">
                <h2 style="color: {result_info['text']}; margin-bottom: 1rem; font-size: 2rem;">
                    {result_info['icon']} Diagnostic: {CLASSES[class_idx]}
                </h2>
                <div style="
                    background: white; 
                    border-radius: 12px; 
                    padding: 1.5rem; 
                    margin: 1rem 0;
                    box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
                ">
                    <h3 style="color: #1f2937; margin-bottom: 0.5rem;">Niveau de Confiance</h3>
                    <div style="
                        font-size: 3rem; 
                        font-weight: 700; 
                        color: {result_info['text']};
                        margin: 1rem 0;
                    ">
                        {confidence*100:.1f}%
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 Analyse Détaillée des Probabilités")
        
        for i, prob in enumerate(prediction[0]):
            class_name = CLASSES[i].lower()
            color_class = class_name if class_name in ['normal', 'covid', 'pneumonia'] else 'normal'
            
            st.markdown(f"""
                <div style="margin: 1rem 0; display: flex; align-items: center; gap: 1rem;">
                    <div style="min-width: 120px; font-weight: 500; color: #1f2937;">
                        <strong>{CLASSES[i]}</strong>
                    </div>
                    <div style="flex: 1; background: #f1f5f9; border-radius: 10px; height: 24px; overflow: hidden; position: relative;">
                        <div style="
                            height: 100%; 
                            border-radius: 10px; 
                            width: {prob*100}%; 
                            background: linear-gradient(135deg, {'#10b981' if class_name == 'normal' else '#ef4444' if class_name == 'covid' else '#f59e0b'}, {'#059669' if class_name == 'normal' else '#dc2626' if class_name == 'covid' else '#d97706'});
                            display: flex; 
                            align-items: center; 
                            justify-content: center; 
                            color: white; 
                            font-weight: 600; 
                            font-size: 0.875rem;
                        ">
                            {prob*100:.1f}%
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        recommendations = {
            "Normal": {
                "message": "Aucune anomalie détectée dans la radiographie.",
                "action": "Continuez à maintenir une bonne santé respiratoire.",
                "color": "#10b981"
            },
            "Covid": {
                "message": "Signes potentiels de COVID-19 détectés.",
                "action": "Consultez immédiatement un médecin et suivez les protocoles de sécurité.",
                "color": "#ef4444"
            },
            "Pneumonia": {
                "message": "Signes de pneumonie détectés.",
                "action": "Consultez un pneumologue pour un diagnostic complet.",
                "color": "#f59e0b"
            }
        }
        
        rec = recommendations.get(CLASSES[class_idx], recommendations["Normal"])
        
        st.markdown("### 🩺 Recommandations")
        st.markdown(f"""
            <div style="
                background: #f8fafc; 
                border-left: 4px solid {rec['color']}; 
                padding: 1.5rem; 
                border-radius: 8px;
                margin: 1rem 0;
            ">
                <h4 style="color: {rec['color']}; margin-bottom: 1rem;">⚠️ Important</h4>
                <p style="color: #1f2937; margin-bottom: 1rem;"><strong>{rec['message']}</strong></p>
                <p style="color: #64748b; margin: 0;">{rec['action']}</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style="
                background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                border: 2px solid #f59e0b; 
                border-radius: 16px; 
                padding: 2rem; 
                margin: 2rem 0;
                box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
            ">
                <h3 style="color: #92400e; margin-bottom: 1.5rem; text-align: center;">
                    ⚠️ Avertissement Médical Important
                </h3>
                <p style="color: #374151; margin: 0; font-size: 0.95rem; text-align: center;">
                    <strong>Ce système est destiné à des fins de recherche et d'éducation uniquement.</strong><br>
                    Ne remplace pas l'avis d'un professionnel de santé qualifié. 
                    En cas d'urgence médicale, appelez immédiatement les services d'urgence.
                </p>
            </div>
        """, unsafe_allow_html=True)
