# functions/translations.py
from deep_translator import GoogleTranslator

def split_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def translate_text(data):
    try:
        text_to_translate = data.get('text', '')
        target_language = data.get('target_language', 'en')

        # Split text into chunks
        text_chunks = split_text(text_to_translate)

        # Translate each chunk and concatenate results
        translated_text = ''
        for chunk in text_chunks:
            translation = GoogleTranslator(source='auto', target=target_language).translate(chunk)
            translated_text += translation + ' '

        return {'translation': translated_text.strip()}

    except Exception as e:
        return {'error': str(e)}
