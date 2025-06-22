# CAA_Project 2 - Deep Learning Approaches for Pneumonia Detection: Multi-Scale Analysis with Explainable AI
Repository containing the second project developed in the Complementos de Aprendizagem Automática (CAA) course.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Data Download](#data-download)
4. [Usage](#usage)
    1. [Local Execution](#local-execution)
    2. [Google Colab Execution](#google-colab-execution-recommended)
5. [Authors](#authors)
6. [License](#license)

## Introduction
This project extends the comparative study of deep learning approaches for pneumonia detection using chest X-ray images. The study evaluates different CNN architectures (AlexNet, ResNet18, ResNet50, DenseNet169) across multiple image resolutions (28x28, 64x64, 128x128) and incorporates Explainable AI (XAI) techniques including GradCAM, LIME, and SHAP to understand model decision-making processes. The project combines performance analysis with interpretability to provide comprehensive insights into pneumonia detection in medical imaging.

## Installation
To get started with this project, follow the steps below:

### 1. Clone the repository
```bash
git clone https://github.com/mariaabr/CAA_Project2.git
cd CAA_Project2
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Data Download
All images used in this project come from [https://doi.org/10.5281/zenodo.10519652](https://doi.org/10.5281/zenodo.10519652).

- **Image Resolutions:** 28x28, 64x64, 128x128 (no 224x224 used).
- **Folder Structure:** Place the extracted data in `data/pneumoniamnist_<resolution>/` (e.g., `data/pneumoniamnist_28/`).
- **Note:** No images are included in the repository. Download them from the link above and extract to the correct folder.
- **Older models:** For comparison with previous work, you will also require models built in [Project 1](https://github.com/miguel-silva48/CAA_Project) and place them in `output/models/`.

## Usage
To run the project, follow these steps:

### Local Execution
1. Ensure your virtual environment is activated.
2. Download the required data (see above) and extract to the correct folders.
3. Open the desired Jupyter Notebook.
4. Click "Run" to execute the cells in the notebook.

### Google Colab Execution (Recommended)
You can run the notebooks in Google Colab with GPU acceleration. Make sure your Google account is the same as the one used for your Google Drive.

**Steps:**
1. Upload the required data and models to your Google Drive (e.g., `/content/drive/MyDrive/data` and `/content/drive/MyDrive/models`).
2. Use the following commands at the start of your notebook:

```python
# Clone the repository
!git clone https://github.com/mariaabr/CAA_Project2.git

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy the data and models from Drive to the repository
!cp -r /content/drive/MyDrive/data /content/CAA_Project2/
!cp -r /content/drive/MyDrive/models /content/CAA_Project2/output/

# Enter the notebooks folder - session restarting point
%cd CAA_Project2/notebooks
```

- **Note:** You may want to restart the session after copying and before running the notebook to avoid memory issues.
- All code will work as long as the data and models are in the correct folders.
- **Reminder:** Don't forget to download the created models, JSON information files and the executed notebook after running.

## Authors
This project is developed by the following authors:

<table>
  <tr>
     <td align="center">
          <a href="https://github.com/mariaabr">
                <img src="https://avatars0.githubusercontent.com/mariaabr?v=3" width="100px;" alt="Rafaela"/>
                <br />
                <sub>
                     <b>Rafaela Abrunhosa</b>
                     <br>
                     <i>107658</i>
                </sub>
          </a>
     </td>
     <td align="center">
          <a href="https://github.com/miguel-silva48">
                <img src="https://avatars0.githubusercontent.com/miguel-silva48?v=3" width="100px;" alt="Miguel"/>
                <br />
                <sub>
                     <b>Miguel Pinto</b>
                     <br>
                     <i>107449</i>
                </sub>
          </a>
     </td>
  </tr>
</table>

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.