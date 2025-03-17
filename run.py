import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Function to load the dataset
def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    return dataset

# Prepare data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Load training and validation datasets
train_dataset = load_dataset('path/to/train.txt', tokenizer)
val_dataset = load_dataset('path/to/val.txt', tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./fine-tuned-model')
tokenizer.save_pretrained('./fine-tuned-model')

# Load the fine-tuned model for inference
fine_tuned_model = GPT2LMHeadModel.from_pretrained('./fine-tuned-model')
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained('./fine-tuned-model')

# Function to generate responses
def generate_response(prompt):
    inputs = fine_tuned_tokenizer(prompt, return_tensors='pt')
    outputs = fine_tuned_model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1)
    response = fine_tuned_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage
prompt = "What is the policy for data privacy?"
response = generate_response(prompt)
print(response)
