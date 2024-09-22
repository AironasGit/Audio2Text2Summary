# Audio to text to summary for Lithuanian language
 Script that uses [Whisper](https://github.com/openai/whisper) model to turn audio into text and [LukasStankevicius/t5-base-lithuanian-news-summaries-175](https://huggingface.co/LukasStankevicius/t5-base-lithuanian-news-summaries-175) model to create a summary of the generated text

## Setup
```
git clone https://github.com/AironasGit/Audio2Text2Summary.git
```

## Usage for Windows
Open CMD and activate virtual enviroment
```
.\Audio2Text2Summary\.venv\Scripts\activate
```
Run main.py with -inputFile and -outputPath flags
```
python main.py -inputFile [file path] -outputPath [dir path]
```
## Flags
```
-inputFile
```
* Accepted audio formats: .mp3, .mp4, .mpeg, .mpga, .m4a, .wav, .webm
* Accepts .txt file and creates a summary for it

```
-outputPath
```
* Accepts a path to a directory where the generated text and summary will be placed
* If outputPath is not provided the current working directory will be used