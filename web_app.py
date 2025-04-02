import streamlit as st
import os
from llm.vhs_chain import interpret_vhs
import torch
from inference.inference import load_model, get_transform, visualize_prediction_measurements
import matplotlib.pyplot as plt

torch.classes.__path__ = []  ##to avoid conflict with streamlit

# Set page title and favicon
st.set_page_config(
    page_title="VHS Analyzer",
    page_icon="❤️",
    layout="wide"
)

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

# Title
st.title("Veterinary Heart Score (VHS) Analyzer")

# Add explanation
with st.expander("About VHS Measurements"):
    st.write("""
    **Vertebral Heart Score (VHS)** is a radiographic measurement that compares heart size to vertebral body length:
    - The **long axis (L)** measurement runs from the carina to the cardiac apex
    - The **short axis (S)** measurement is perpendicular to L at the widest part of the heart
    - **T** is the reference vertebral length measurement
    - VHS = 6 * ((L + S) / T), expressed in vertebral body units
    
    Normal ranges:
    - Dogs: 9.7 ± 0.5 vertebrae (range: 8.7-10.7)
    - Cats: 7.5 ± 0.3 vertebrae (range: 7.0-8.1)
    
    Some breeds have different normal ranges.
    """)

# Navigation functions
def go_to_landing():
    st.session_state.page = 'landing'

def go_to_manual():
    st.session_state.page = 'manual'

def go_to_image():
    st.session_state.page = 'image'

# LANDING PAGE
if st.session_state.page == 'landing':
    st.write("""
    This application helps interpret Vertebral Heart Score (VHS) measurements 
    to assess heart size in dogs and cats. You can either:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Upload a radiograph** for automatic measurement and analysis")
        if st.button("Upload Image", key="btn_upload", use_container_width=True):
            go_to_image()
    
    with col2:
        st.info("**Enter measurements manually** if you already have L and S values")
        if st.button("Manual Entry", key="btn_manual", use_container_width=True):
            go_to_manual()

# IMAGE UPLOAD PAGE
elif st.session_state.page == 'image':
    st.subheader("Radiograph Analysis")
    
    # Back button
    if st.button("← Back", key="back_from_image"):
        go_to_landing()
    
    st.write("Upload a chest radiograph (X-ray) to automatically calculate and interpret the VHS.")
    
    # Form for image upload
    with st.form("image_form"):
        # Basic patient info
        col1, col2 = st.columns(2)
        
        with col1:
            animal_type = st.selectbox("Animal Type", ["Dog", "Cat"])
            breed = st.text_input("Breed (optional)")
        
        with col2:
            age_input = st.number_input("Age in Years (optional)", min_value=0.0, max_value=25.0, value=0.0, step=0.1)
            age = age_input if age_input > 0 else None
            
            weight_input = st.number_input("Weight in kg (optional)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
            weight = weight_input if weight_input > 0 else None
            
            sex = st.selectbox("Sex (optional)", ["Not specified", "Male", "Female", "Male (neutered)", "Female (spayed)"])
            sex = None if sex == "Not specified" else sex
        
        # Image upload
        uploaded_file = st.file_uploader("Upload radiograph image", type=["jpg", "jpeg", "png"])
        
        submit_button = st.form_submit_button(label="Analyze Radiograph")
    
    if submit_button and uploaded_file is not None:
        # Save uploaded file temporarily
        temp_file_path = f"temp_radiograph_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Run analysis
        with st.spinner("Processing radiograph..."):
            try:
                model = load_model('models/best_model_efb7.pt')
                transform = get_transform(resized_image_size=300)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Create figure for visualization
                fig, ax = plt.subplots(figsize=(10, 10))
                
                # Get measurements and visualize prediction
                l_value, s_value, t_value, vhs_score = visualize_prediction_measurements(
                    image_path=temp_file_path,
                    model=model,
                    transform=transform,
                    device=device,
                    ax=ax,
                    resized_image_size=300
                )
                
                # Display the image with predictions
                st.pyplot(fig)
                
                # Display measurements
                st.subheader("Measurements")
                st.write(f"**Long Axis (L):** {l_value:.2f}")
                st.write(f"**Short Axis (S):** {s_value:.2f}")
                st.write(f"**Reference Length (T):** {t_value:.2f}")
                st.write(f"**VHS Score:** {vhs_score:.2f} vertebrae")
                
                # Show reference range based on animal type
                if animal_type == "Dog":
                    st.write("**Reference Range for Dogs:** 8.7-10.7 vertebrae (typically 9.7 ± 0.5)")
                else:
                    st.write("**Reference Range for Cats:** 7.0-8.1 vertebrae (typically 7.5 ± 0.3)")
                
                # Get interpretation from LLM
                with st.spinner("Analyzing VHS score..."):
                    interpretation = interpret_vhs(
                        l_value=float(l_value),
                        s_value=float(s_value),
                        t_value=float(t_value),
                        animal_type=animal_type,
                        breed=breed if breed else None,
                        age=age,
                        weight=weight,
                        sex=sex
                    )
                    
                    # Display interpretation
                    st.subheader("VHS Interpretation")
                    st.write(f"**Normal Range:** {interpretation.normal_range}")
                    st.write(f"**Interpretation:** {interpretation.interpretation}")
                    
                    if "normal" not in interpretation.interpretation.lower():
                        st.write(f"**Severity:** {interpretation.severity}")
                    
                    st.subheader("Possible Conditions")
                    for condition in interpretation.possible_conditions:
                        st.write(f"- {condition}")
                    
                    st.subheader("Recommendations")
                    for rec in interpretation.recommendations:
                        st.write(f"- {rec}")
                    
                    st.subheader("Detailed Clinical Explanation")
                    st.write(interpretation.detailed_explanation)
                
                # Clean up temporary file
                os.remove(temp_file_path)
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.info("Please try again with a different image or use manual entry.")
                
                # Add a debugging expander for developers
                with st.expander("Technical Details (for debugging)", expanded=False):
                    st.code(str(e))
                
                # Clean up temp file if it exists
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

# MANUAL ENTRY PAGE
elif st.session_state.page == 'manual':
    st.subheader("Manual Measurement Entry")
    
    # Back button
    if st.button("← Back", key="back_from_manual"):
        go_to_landing()
    
    st.write("Enter the VHS measurements manually to get an interpretation.")
    
    # Input form (same as your current implementation)
    with st.form("vhs_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            animal_type = st.selectbox("Animal Type", ["Dog", "Cat"])
            l_value = st.number_input("Long Axis (L) Value", min_value=0.0, max_value=15.0, value=5.5, step=0.1)
            s_value = st.number_input("Short Axis (S) Value", min_value=0.0, max_value=10.0, value=4.2, step=0.1)
            t_value = st.number_input("Reference Vertebral Length (T)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            breed = st.text_input("Breed (optional)")
        
        with col2:
            age_input = st.number_input("Age in Years (optional)", min_value=0.0, max_value=25.0, value=0.0, step=0.1)
            age = age_input if age_input > 0 else None
            
            weight_input = st.number_input("Weight in kg (optional)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
            weight = weight_input if weight_input > 0 else None
            
            sex = st.selectbox("Sex (optional)", ["Not specified", "Male", "Female", "Male (neutered)", "Female (spayed)"])
            sex = None if sex == "Not specified" else sex
        
        submit_button = st.form_submit_button(label="Analyze VHS")

    # Calculate VHS and display results when form is submitted
    if submit_button:
        vhs_score = 6 * ((l_value + s_value) / t_value)
        st.write(f"**VHS Score:** {vhs_score:.1f}")
        
        # Show reference range based on animal type
        if animal_type == "Dog":
            st.write("**Reference Range for Dogs:** 8.7-10.7 vertebrae (typically 9.7 ± 0.5)")
        else:
            st.write("**Reference Range for Cats:** 7.0-8.1 vertebrae (typically 7.5 ± 0.3)")
        
        # Add progress indicator
        with st.spinner("Analyzing VHS score..."):
            try:
                interpretation = interpret_vhs(
                    l_value=l_value,
                    s_value=s_value,
                    t_value=t_value,
                    animal_type=animal_type,
                    breed=breed if breed else None,
                    age=age,
                    weight=weight,
                    sex=sex
                )
                
                # Display interpretation
                st.subheader("VHS Interpretation")
                st.write(f"**Normal Range:** {interpretation.normal_range}")
                st.write(f"**Interpretation:** {interpretation.interpretation}")
                
                if "normal" not in interpretation.interpretation.lower():
                    st.write(f"**Severity:** {interpretation.severity}")
                
                st.subheader("Possible Conditions")
                for condition in interpretation.possible_conditions:
                    st.write(f"- {condition}")
                
                st.subheader("Recommendations")
                for rec in interpretation.recommendations:
                    st.write(f"- {rec}")
                
                st.subheader("Detailed Clinical Explanation")
                st.write(interpretation.detailed_explanation)
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.info("Please try again or adjust your input values.")
                
                # Add a debugging expander for developers
                with st.expander("Technical Details (for debugging)", expanded=False):
                    st.code(str(e))