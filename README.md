# PCOS Diagnosis UI 🐠

**Demo:** [Try the CNN model here](https://huggingface.co/spaces/ecats/pcos_diagnosis_ui)

---

## Project Description

### Overview
Polycystic ovary syndrome (PCOS) is a common hormonal disorder affecting women of reproductive age, leading to hormonal imbalances, irregular periods, high androgen levels, and ovarian cysts. It is a significant cause of infertility, with an estimated 8-13% of women affected globally, and up to 70% of cases undiagnosed, according to WHO.

### Proposal
Our project aims to leverage AI and computer vision to aid in diagnosing PCOS based on symptoms and ultrasound images. Additionally, it seeks to assist individuals diagnosed with PCOS in monitoring their condition and its health impacts. This tool is designed to support, not replace, medical professionals.

Early diagnosis and treatment could significantly lower the risks of long-term complications such as type 2 diabetes and heart disease.

### Challenges
- **Data Availability**: Efficient and reliable datasets are essential. Considering ethnic variations in PCOS manifestation and impact is crucial for dataset and model design.
- **Model Evaluation**: Defining the success criteria for the model is a key challenge.

---

## Features
- **AI-Based Diagnosis**: Utilize computer vision to analyze ultrasound images.
- **Monitoring Tool**: Track the progression and impact of PCOS on health.

---

## Note
This repository specifically contains the CNN model for the computer vision part of the project, which analyzes ultrasound images to aid in diagnosing PCOS. The analysis based on symptoms is not included in this repository.

---

## Setup

To run this app locally, clone the repository and install the dependencies:

```bash
git clone https://github.com/OvaTech-AI/pcos_cnn_diagnosis.git
cd pcos_cnn_diagnosis
pip install -r requirements.txt
python app.py
```

---

## License
This project is licensed under the MIT License.

---

Feel free to reach out with any questions or contributions!