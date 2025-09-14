if __name__ == '__main__':
    from transformers import MarianMTModel, MarianTokenizer

    model_path = "D:/MiniProject/LanguageTranslator/jobfiles/translator"
    translation_model_name = "Helsinki-NLP/opus-mt-en-hi"

    translator_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
    translator_model = MarianMTModel.from_pretrained(translation_model_name)

    translator_tokenizer.save_pretrained(model_path)
    translator_model.save_pretrained(model_path)
    print(f"Saved Model and Tokenizer to path: {model_path}")