# CLIP_RL_FT
CLIP fine tunned with Reinforcement Learning 
CLIP Fine-Tuning and Reinforcement Learning
This project demonstrates the fine-tuning of OpenAI's CLIP model using supervised learning and reinforcement learning (RL). The workflow includes data preparation, model training, validation, RL fine-tuning, and testing, showcasing a comprehensive pipeline for improving a vision-language model.
Features
	•	Fine-tuning the CLIP model on a subset of the MS COCO dataset.
	•	Dataset splitting into training, validation, and test sets (80%/10%/10%).
	•	Supervised learning with cross-entropy loss for text-image matching.
	•	Reinforcement learning fine-tuning with a custom reward function.
	•	Validation and testing for performance evaluation.
Requirements
	•	Python 3.8+
	•	PyTorch
	•	Transformers library by Hugging Face
	•	Datasets library by Hugging Face
	•	TQDM for progress visualization

Usage
1. Dataset Preparation
The script uses the "JotDe/mscoco_100k" dataset from Hugging Face. Ensure you have an internet connection to download the dataset automatically.
2. Training the Model
Run the script to fine-tune the CLIP model using supervised learning for 15 epochs:
python TrainCLIP.ipynb
This step:
	•	Trains the model with cross-entropy loss.
	•	Validates the model after each epoch.
3. Reinforcement Learning Fine-Tuning
The script reduces the dataset size and fine-tunes the model with RL for 3 epochs:
python TrainCLIP_RL.ipynb
This step:
	•	Utilizes a reward function based on prediction correctness and confidence.
	•	Saves the RL fine-tuned model for further testing.

4. Testing
Evaluate the fine-tuned model on the test set:
python Test_CLIP_RL.ipynb
Outputs the test accuracy.
5. Using for captioning
.use with BLIP
.UseME_CLIP_BLIP.ipynb

Results
	•	Supervised Learning: Reports average training and validation losses.
	•	Reinforcement Learning: Outputs average rewards per epoch.
	•	Testing: Final accuracy on the test set.
Model Saving
	•	Fine-tuned Model: Saved under clip_finetuned/
	•	RL Fine-tuned Model: Saved under clip_rl_finetuned/
File Structure
CLIP_RL_FT/
├── Model/                     # Contains the trained models and related artifacts
│   ├── clip_rl_finetuned/     # Fine-tuned CLIP-RL model files
│   	        
├── Scripts/                   # Scripts for training, testing, and usage
│   ├── TrainCLIP.ipynb        # Supervised learning script
│   ├── TrainCLIP_RL.ipynb     # RL fine-tuning script
│   ├── Test_CLIP_RL.ipynb     # Testing script
│   └── UseME_CLIP_BLIP.ipynb  # Script for user input and BLIP usage
│
├── requirements.txt           # Dependencies for the project
├── README                     # Detailed documentation 

Future Improvements
	•	Extend the reward function for more sophisticated RL objectives.
	•	Experiment with larger datasets and batch sizes.
	•	Evaluate the model with additional metrics like precision and recall.

Developed by Shreyas Shashi Kumar Gowda, Rachit Rajesh Parihar.
