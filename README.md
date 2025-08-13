LoRA Fine-Tuned Product Description Generator
This project fine-tunes a GPT model using LoRA (Low-Rank Adaptation) to generate high-quality product descriptions based on a product title and features.

Dataset
Source: Handmade products dataset link: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw_meta_Handmade_Products

After cleaning: 56,000 records

Cleaning script: scripts/cleaning.py removes:

Empty or too-short descriptions (less than 20 characters)

Records without features

HTML tags and URLs

Unwanted metadata such as average rating and rating number

Training Setup
Model: GPT-2 (base) fine-tuned with LoRA

Epochs: 1

Batching: Gradient accumulation for efficiency

Evaluation: Train/validation split to monitor overfitting

Parameters:

Repetition penalty:1.2

No repeat n-gram size: 3

Temperature varied for generation tests
Max length: 512

Hardware: Google Colab (T4 GPU)
Training Progress


Epoch : 1
Training Loss: 1.112 
Validation Loss: 1.047
Example Output

**Prompt:**  
Title: Handwoven Wool Throw Blanket  
Features: 100% natural wool, Traditional loom weaving, Fringed edges, Soft and warm for winter  

**Generated Description:**  
This is a perfect gift to your favorite holiday or special occasion. The blanket features the traditional loom weave pattern and has been handcrafted with high quality materials. The fabric is soft and cozy so you can enjoy warmth on cold nights.





