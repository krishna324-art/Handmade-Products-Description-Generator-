import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your fine-tuned model
model_path = "./lora_finetuned_model"  # <-- change this to your LoRA model folder
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()

# Unseen prompts in dataset format
test_prompts = [
  
    {
        "title": "Handmade Leather Dog Collar with Brass Buckle",
        "features": [
            "Full-grain leather",
            "Solid brass hardware",
            "Adjustable fit",
            "Suitable for medium and large dogs"
        ]
    }
    
]

# Try different temperatures
temperatures = [0.3, 0.7, 1.0]

for prompt_data in test_prompts:
    prompt = f"Title: {prompt_data['title']}\nFeatures: {', '.join(prompt_data['features'])}\nDescription:"
    print(f"\n--- Prompt ---\n{prompt}\n")
    
    for temp in temperatures:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=temp,
            top_p=0.9
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Print only the generated description part
        description = generated_text.split("Description:")[-1].strip()
        print(f"[Temperature {temp}] {description}")

including 
top_p,repetition_penalty and no_repeat_ngram_size  

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load fine-tuned model
model_path = "./lora_finetuned_model"  # <-- change to your LoRA model folder
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()


test_prompts = [
    
   
    {
        "title": "Handwoven Wool Throw Blanket",
        "features": ["100% natural wool", "Traditional loom weaving", "Fringed edges", "Soft and warm for winter"]
    },
    {
        "title": "Rustic Metal Wall Clock",
        "features": ["Aged bronze finish", "Silent sweep movement", "Large Roman numerals", "Easy wall mounting"]
    }
]


# Temperatures for generation
temperatures = [0.3, 0.7, 1.0]

for prompt_data in test_prompts:
    # Convert to one-line prompt for generation
    features_text = ", ".join(prompt_data["features"])
    prompt = f'Title: {prompt_data["title"]} Features: {features_text} Description:'
    
    print(f"\n--- Prompt ---\n{prompt}\n")
    
    for temp in temperatures:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=temp,
            top_p=0.9,
            repetition_penalty=1.2,   # discourage repeating tokens
            no_repeat_ngram_size=3 
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract generated description part
        description = generated_text.split("Description:")[-1].strip()
        print(f"[Temperature {temp}] {description}")
