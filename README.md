# Rich Cousin Poor Sister

# Hindi-Chhattisgarhi Cross-Lingual Transfer Using RL-Guided Distillation

**Authors:**
- Raghav Borikar (M24DS010) - raghavbori@iitbhilai.ac.in
- Vikrant Sahu (P24CS007) - vikrantsahu@iitbhilai.ac.in

**Date:** March 26, 2025 (Last Update)

## 1. Project Overview
This project focuses on developing a specialized framework that efficiently transfers knowledge from Hindi (a high-resource language) to Chhattisgarhi (a low-resource language) using reinforcement learning and knowledge distillation techniques. Our goal is to optimize core transfer learning mechanisms to achieve efficient cross-lingual transfer.

## 2. Key Components
The project utilizes the following key components:

- **NLLB-Enhanced Transfer Learning:** 
  We use the No Language Left Behind (NLLB) dataset as the foundation for our transfer learning model. This dataset provides a diverse set of language pairs to enhance the cross-lingual capabilities of our system.

- **Reinforcement Learning for Selective Knowledge Transfer:** 
  Reinforcement learning (RL) is employed to dynamically determine the optimal parameters for transferring knowledge from Hindi to Chhattisgarhi, ensuring that only the most relevant information is transferred effectively.

- **Bidirectional Lexical Mapping:** 
  The project leverages existing Hindi-Chhattisgarhi parallel data to build a bidirectional lexical mapping, which enhances the quality of the language transfer by providing a richer understanding of both languages.

- **Knowledge Distillation:** 
  We implement confidence-guided distillation techniques to ensure that the distilled knowledge from the high-resource language (Hindi) is effectively transferred to the low-resource language (Chhattisgarhi).

## 3. Setup, Installation & Running the Project

## Step 1: Switch to the `/mt` Branch
Navigate to the `/mt` branch of the GitHub repository. This branch focuses on machine translation. Clone the repository or download it to your local system.

## Step 2: Navigate to the Root Directory on Your Local System
Make sure you're in the root directory of the project on your local machine. Also, ensure that you have sufficient GPU resources available to run the project locally.

## Step 3: Install Dependencies
Install the required dependencies listed in the `requirements.txt` file using `pip` or any other package manager you prefer.

### Step 3a: Resolving Errors Related to WordNet or NLTK
If you encounter any errors related to WordNet or NLTK during installation, run the script `scripts/post_req.py`. This will automatically resolve the issues.

## Step 4 (Optional): Download Dataset
The dataset for the project can be downloaded using the script `scripts/download_data.py`. However, this step is optional as the dataset is small and has already been uploaded to GitHub.

## Step 5: Running the Project
You can run the project in four different modes. Below are the modes and the respective commands to run:

- **Fine-tuning the base model**:
    ```bash
    python -m src.main --mode train_base
    ```

- **Training the model using knowledge distillation**:
    ```bash
    python -m src.main --mode train_distillation
    ```

- **Using reinforcement learning (RL) guided knowledge distillation**:
    ```bash
    python -m src.main --mode train_rl
    ```

- **Evaluation**:
    ```bash
    python -m src.main --mode evaluate
    ```
Note: You may change the config by defining the arguments on command line.

## Step 6: Choose a Mode and Run the Command
Select the mode you wish to run and pass it as an argument when executing the script. For example, to train the base model, use the following command:
```bash
python -m src.main --mode train_base --train_data data/processed/train.json
```
## Step 7: Run Evaluation 
For evaluation: code will ask for a checkpoint path. download the most recent checkpoint from here [https://drive.google.com/file/d/181WNX-74pk_tdb941rz4s5ryDsEmJSDu/view?usp=sharing], create a folder named checkpoints in the root directory & store the downloaded model checkpoint there, set the mode to evaluate and provide the path to the checkpoint. Also provide the relevant model & ensure that the checkpoint belongs to that model itself. else, run the script given below:
  ```bash
  python -m src.main --mode evaluate --model_name facebook/mbart-large-50 --checkpoint_path checkpoints/student_episode_1.pt
  ```

## Step 8: Checkpoint and Log Files
After training or evaluation, the relevant checkpoints, logs, and results will be saved in their respective directories located in the root folder of the project.

## 5. Results
Though Extensive Tests, Ablation Studies & Hyperparameter Tuning (especially RL-Based) is yet to be done, the RL-Guided distilled model has shown a BLEU Score of 31.67 after just 1 episode compared to the score of 25.64 obtained by simple Fine-Tuned Model after 1 epoch of fine tuning.

## 6. Contributing
Feel free to open an issue or submit a pull request if you'd like to contribute to the project.

## 7. Contact
For any inquiries, please contact:
- Raghav Borikar: raghavbori@iitbhilai.ac.in

=======
Project Repo for Deep Learning for Low Resource NLP Course Project.
