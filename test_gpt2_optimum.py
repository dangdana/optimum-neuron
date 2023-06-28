import os

import torch
from transformers import AutoTokenizer
from optimum.neuron import NeuronModelForCausalLM


os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer-inference"

# Compilation does not work with batch_size = 1
batch_size = 2
seq_length = 128

# Load and convert the Hub model to Neuron format
model_neuron = NeuronModelForCausalLM.from_pretrained(
    "gpt2", batch_size=batch_size, sequence_length=seq_length, export=True, tp_degree=2, amp="f32"
)

print("HF model converted to Neuron")

# Get a tokenizer and example input
tokenizer = AutoTokenizer.from_pretrained("gpt2")
prompt_text = "Hello, I'm a language model,"
# We need to replicate the text because batch_size is not 1
prompts = [prompt_text for _ in range(batch_size)]

# Encode text and generate using AWS sampling loop
encoded_text = tokenizer(prompts, return_tensors='pt')
with torch.inference_mode():
    generated_sequence = model_neuron.model.sample(encoded_text.input_ids, sequence_length=seq_length)
    print([tokenizer.decode(tok) for tok in generated_sequence])

print("Outputs generated using AWS sampling loop")

# Specifiy padding options
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

# Encode tokens and generate using temperature
tokens = tokenizer(prompts, padding=True, return_tensors='pt')
model_neuron.reset_generation() # Need to check if this can be automated
sample_output = model_neuron.generate(
    **tokens,
    do_sample=True,
    max_length=seq_length,
    temperature=0.7,
)
print([tokenizer.decode(tok) for tok in sample_output])

print("Outputs generated using HF generate")