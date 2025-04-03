# VHS Analyzer - Veterinary Heart Score Interpretation Tool

![VHS Analyzer](https://img.shields.io/badge/AI-Veterinary%20Medicine-brightgreen)

A sophisticated application that analyzes and interprets Vertebral Heart Score (VHS) measurements for veterinary cardiology. Combines computer vision with large language models to provide comprehensive clinical assessments of heart size in dogs and cats.

## 📋 Features

- **Dual Input Methods**:
  - Upload radiographs for automatic VHS measurement
  - Manual entry of L, S, and T values
  
- **Automated Analysis**:
  - Computer vision detection of key cardiac measurement points 
  - Accurate calculation of VHS using the formula: 6 * ((L + S) / T)
  
- **Comprehensive Interpretation**:
  - Species and breed-specific normal ranges
  - Clinical significance assessment
  - Severity classification
  - Potential associated conditions
  - Detailed clinical explanations (by an LLM)
  - Veterinary recommendations (by an LLM)

## 🧠 Technology Stack

- **Python 3.11+**
- **Deep Learning**: PyTorch, EfficientNet-B7
- **AI Language Model**: GPT-4o via LangChain
- **Web Framework**: Streamlit
- **Data Visualization**: Matplotlib
- **Image Processing**: PIL, torchvision

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/vhs-analyzer.git
   cd vhs-analyzer

2. **Create a virtual environment**
   ```bash
   python -m venv virtual_env
   source virtual_env/bin/activate  # On Windows: virtual_env\Scripts\activate

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt

4. **Configure OpenAI API key**
   ```bash
    export OPENAI_API_KEY="your_api_key"

5. **Model Download**
   - The model will be automatically download when loading the app.
   - You can also donwload it manually by running:
   ```bash
   python -m utils.download_model

## 🚀 Usage

1. Start the Streamlit application
   ```bash
    streamlit run web_app.py

2. Using the application
  - From the landing page, select either "Upload Image" or "Manual Entry"
  - For image upload: Select animal type, provide optional details, upload a radiograph
  - For manual entry: Enter L, S, and T measurements along with patient details
  - Review the VHS calculation and clinical interpretation

## 🔍 How It Works

### Image Analysis
1. A chest radiograph is processed through EfficientNet-B7
2. The model detects 6 key points (3 pairs) representing:
   - Long axis (L): From carina to cardiac apex
   - Short axis (S): Perpendicular to L at the widest part of the heart
   - Reference length (T): Standard vertebral length

### VHS Calculation
- VHS = 6 * ((L + S) / T)
- Normal ranges:
  - Dogs: 9.7 ± 0.5 vertebrae (range: 8.7-10.7)
  - Cats: 7.5 ± 0.3 vertebrae (range: 7.0-8.1)

### Clinical Interpretation
The GPT-4o model analyzes the measurements considering:
- Animal type
- Breed-specific variations
- Age, weight, and sex
- Deviation from normal ranges

The interpretation includes:
- Normal range confirmation
- Heart size assessment
- Severity classification if enlarged
- Potential conditions
- Recommended follow-up actions
- Detailed clinical explanation

## 📊 Performance

The application has been tested with the following results:
- Model loading time: ~1.42 seconds
- CNN inference time: ~0.44 seconds
- Image processing time: ~0.48 seconds
- LLM interpretation time: ~5.90 seconds
- Total analysis time: <7 seconds per radiograph

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- The EfficientNet implementation is based on VHS reference values based on standard veterinary cardiology literature

---

*This project was developed as part of the AI for Health course at Aivancity.*

