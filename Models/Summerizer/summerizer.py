from transformers import T5Tokenizer, TFT5ForConditionalGeneration 

model_path = "./t5"
model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name,legacy=False)
model = TFT5ForConditionalGeneration.from_pretrained(model_name)
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print(f"Model saved to {model_path}")


# text = input("Enter Text to summerize : ")
# mlength = int(len(text) * 0.40)
# inputs = tokenizer.encode("summarize: " + text, return_tensors="pt")
# summary_ids = model.generate(inputs, max_length=mlength, min_length=100, length_penalty=1.0, num_beams=10, early_stopping=True)
# Summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
# print("Summerizing Text\n\n")
# print(Summary)