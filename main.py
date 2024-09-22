import os
import argparse
import whisper
from transformers import pipeline

def read_file(path) -> str:
    with open(path, 'r', encoding="utf-8") as f:
        data = f.read()
        f.close()
    return data

def write_result(path, file_name, data) -> None:
    print('Writing results...')
    with open(f'{path}/{file_name}.txt', 'w', encoding="utf-8") as f:
        f.write(data)
        f.close()

def text2summary(text) -> str:
    model_name= 'LukasStankevicius/t5-base-lithuanian-news-summaries-175'
    print('Loading "t5-base-lithuanian-news-summaries-175" model...')
    my_pipeline = pipeline(task="text2text-generation", model=model_name, framework="pt")
    print('Summarizing...')
    summary = my_pipeline(text)[0]['generated_text']
    return summary
    
def audio2text(audio_path) -> str:
    print('Loading "Whisper" speech recognition model...')
    model = whisper.load_model('large')
    print('Transcribing...')
    result = model.transcribe(audio_path)
    return result['text']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-inputFile")
    parser.add_argument("-outputPath")
    args = parser.parse_args()
    
    if args.inputFile == None:
        print('Error: No file was provided. Please use -inputFile flag')
        return
    
    audio_formats = ('mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm')
    text_formats = ('txt')
    input_file_path = args.inputFile
    input_file_name = os.path.basename(input_file_path).rsplit('.')[0]
    output_path = args.outputPath
    
    if output_path == None:
        output_path = os.getcwd()
    if not os.path.exists(output_path):
        print('Error: Dir does not exist')
        return
    if not os.path.isfile(input_file_path):
        print('Error: Provided file does not exist')
        return
    if not input_file_path.endswith(audio_formats) and not input_file_path.endswith(text_formats):
        print(f'Error: Wrong file format. Please provide one of these: {audio_formats} or {text_formats}')
        return
    
    if input_file_path.endswith(audio_formats):
        text = audio2text(input_file_path)
        write_result(output_path, f'{input_file_name}_text', text)
        summary = text2summary(text)
        write_result(output_path, f'{input_file_name}_summary', summary)
        print('Done!')
    elif input_file_path.endswith(text_formats):
        text = read_file(input_file_path)
        summary = text2summary(text)
        write_result(output_path, f'{input_file_name}_summary', summary)
        print('Done!')
        
    
if __name__ == '__main__':
    main()