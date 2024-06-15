from transformers import GPT2Tokenizer, pipeline, GPT2LMHeadModel
import torch

train_txt = "dante.txt"
output_txt = "./results/exercise2/output.txt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(train_txt, 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
txt_lenght = len(text)
tokenized_text = tokenizer.encode(text, return_tensors="pt")
print("# of characters in Dante's Inferno:", len(text))
print("Tokenized length of Dante's Inferno:", len(tokenized_text[0]))
print("Ratio:", len(tokenized_text[0]) / len(text))

# generator = pipeline('text-generation', model='gpt2')
# input_prompt = "Bill Evans is the greatest jazz pianist"
# print("Prompt:", input_prompt)
# # print(generator(input_prompt, max_length=50, num_return_sequences=1, temperature=0.9))
# print("Generator:", generator(input_prompt, max_length=50, num_return_sequences=1))

model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "Alice and Bob"

model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
model.to(device)

generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True, temperature=1)
# generated_ids = model.generate(**model_inputs, max_new_tokens=200, do_sample=True, temperature=0.5)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("generator: ", generated_text)

#Save output
# with open(output_txt, 'w') as f:
#     # write to the file
#     f.write(generated_text)
#     # close the file
#     f.close()