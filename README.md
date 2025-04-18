# CAA_Project - Deep Learning Approaches for Pneumonia Detection: A Comparative Study on Chest X-Ray Images
Repository containing the first project developed in the Complementos de Aprendizagem Automática (CAA) course.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
    1. [Local Execution](#local-execution)
    2. [Google Colab Execution](#google-colab-execution-recommended)
4. [Authors](#authors)
5. [License](#license)

## Introduction
This project aims to compare different deep learning approaches for detecting pneumonia using chest X-ray images. The study evaluates the performance of various models and techniques to identify the most effective method for this task.

## Installation
To get started with this project, follow the steps below:

### 1. Clone the repository
```bash
git clone https://github.com/miguel-silva48/CAA_Project.git
cd CAA_Project
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

## Usage
To run the project, follow these steps:

### Local Execution
1. Ensure your virtual environment is activated.
2. Open the desired Jupyter Notebook.
3. Click "Run" to execute the cells in the notebook.

### Google Colab Execution (Recommended)
You can run the notebooks in Google Colab with GPU acceleration:

1. Access the notebook via GitHub by changing the GitHub URL to a Colab URL:
    - Replace `https://github.com/miguel-silva48/CAA_Project/blob/main/notebooks/<notebook_name>.ipynb` with:
    - `https://githubtocolab.com/miguel-silva48/CAA_Project/blob/main/notebooks/<notebook_name>.ipynb`, this will take you to:
    - `https://colab.research.google.com/github/miguel-silva48/CAA_Project/blob/main/notebooks/<notebook_name>.ipynb`

2. Change runtime to GPU:
    - Click "Runtime" > "Change runtime type" > Select "GPU" (NVIDIA T4 GPU was used during development)

3. Run the following at the beginning of the notebook to clone the repository and set up the environment:
    ```python
    !git clone https://github.com/miguel-silva48/CAA_Project.git
    %cd CAA_Project/notebooks
    ```
    This ensures that all utility files needed for execution are properly found.

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