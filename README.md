# 🌟 BabyJoey 🌟

## A Compact 115 Million Parameter Model - 0.5GB

Welcome to BabyJoey, a streamlined language model designed for efficient performance. This document provides a detailed overview of the project structure to help you get started quickly.

---

### 📂 Project Root File Structure

#### 📝 README.md
Comprehensive documentation for the project, explaining the purpose, setup instructions, and usage of BabyJoey.

---

### 📝 README.md

#### Purpose
BabyJoey is a lightweight language model inspired by GPT-1, featuring 115 million parameters and designed for tasks that require a balance of efficiency and performance. It's perfect for educational purposes, research, and small-scale applications.

#### Setup Instructions
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/babyjoey.git
    cd babyjoey
    ```
2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Configure the Model**:
    Modify `config.py` to set your desired hyperparameters and configurations.
4. **Prepare Data**:
    Ensure your datasets are in place and correctly referenced in `data/dataloader.py`.

#### Usage
- **Training the Model**:
    ```bash
    bash scripts/train.sh
    ```
- **Evaluating the Model**:
    ```bash
    bash scripts/evaluate.sh
    ```

---

### 🚀 main.py
The entry point of BabyJoey. This script parses arguments, sets up configurations, and invokes the training loop.

### ⚙️ config.py
Contains configuration settings and hyperparameters for easy management and modification.

### 📋 requirements.txt
Lists the dependencies required to run BabyJoey, which can be installed using `pip`.

---

### 📁 Directories

#### 🧠 model/
- **model.py**: Defines the BabyJoey model architecture, including the 12 decoder blocks.

#### 📊 data/
- **dataloader.py**: Manages data loading logic, including data preprocessing, batching, and any necessary data augmentation.

#### 🏋️ training/
- **trainer.py**: Contains the training loop, validation logic, and evaluation metrics to train and assess BabyJoey.

#### 🛠️ utils/
- **utils.py**: Provides utility functions used throughout the project, such as logging, saving/loading models, and calculating metrics.

#### 📜 scripts/
- **train.sh**: A shell script to start the training process.
- **evaluate.sh**: A shell script to initiate the evaluation process.

---

### 📂 Additional Directories

#### 📑 logs/
Stores log files generated during training and evaluation for debugging and monitoring purposes.

#### 💾 checkpoints/
Stores model checkpoints saved during training to allow for resuming or fine-tuning.

---

We hope you enjoy working with BabyJoey! If you have any questions, feel free to reach out to our support team or check the detailed documentation in the `README.md`. Happy coding! 🚀
