\# LoRA Fine-Tuned Product Description Generator



This project fine-tunes a GPT model using \*\*LoRA (Low-Rank Adaptation)\*\* to generate high-quality product descriptions based on a product \*\*title\*\* and \*\*features\*\*.



---



\## 📂 Dataset

\- \*\*Source:\*\* Handmade products dataset:https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw\_meta\_Handmade\_Products

\- \*\*After cleaning:\*\* 56,000 records

\- \*\*Cleaning Script:\*\* \[`scripts/cleaning.py`](scripts/cleaning.py) removes:

&nbsp; - Empty or too-short descriptions (<20 characters)

&nbsp; - Records without features

&nbsp; - HTML tags and URLs

&nbsp; - Unwanted metadata (`average\_rating`, `rating\_number`, etc.)



---



\## ⚙️ Training Setup

\- \*\*Model:\*\* GPT-2 (base) fine-tuned with LoRA

\- \*\*Epochs:\*\* 1

\- \*\*Batching:\*\* Gradient accumulation for efficiency  

\- \*\*Evaluation:\*\* Train/Validation split to monitor overfitting  

\- \*\*Parameters:\*\*  

&nbsp; - `repetition\_penalty = 1.2`

&nbsp; - `no\_repeat\_ngram\_size = 3

&nbsp; - `temperature` varied for generation tests

&nbsp; - `max\_length = 512

\- \*\*Hardware:\*\* Google Colab (T4 GPU)



---



\## 📈 Training Progress

| Epoch | Training Loss | Validation Loss |

|-------|--------------|-----------------|

| 1     | 1.112        | 1.047        |





---



\## 📝 Example Output



\*\*Prompt:\*\*  

Title: Handwoven Wool Throw Blanket  

Features: 100% natural wool, Traditional loom weaving, Fringed edges, Soft and warm for winter  



\*\*Generated Description:\*\*  

This is a perfect gift to your favorite holiday or special occasion. The blanket features the traditional loom weave pattern and has been handcrafted with high quality materials. The fabric is soft and cozy so you can enjoy warmth on cold nights.



---



\## 🚀 How to Run



```bash

\# Clean dataset

python scripts/cleaning.py



\# Train model with LoRA

python scripts/train\_lora.py



\# Generate sample outputs

python scripts/generate.py



