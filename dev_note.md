# Comparing Self-supervised models for speech emotion recognition task

## Corpus Rule
- Corpus needs to be formatted as follows:
    - Must have audio directory that contains 16kHz wav files for all speech
    - Audio files must follow the following naming convention:
        - <Audio-directory>/<utterance-id>.wav
    - Must have label file that contains csv files for all speech
    - Labels must be in the following format:
        - <utterance-id> <categorical-emotion-1> <categorical-emotion-2> ... <EmoAct> <EmoVal> <EmoDom> <Split_Set>
- If corpus does not follow the above format, additional script is needed to match this format
- Such scripts need to be placed in the "preprocess/corpus" directory
- If categorical emotion label is not provided, write 0 for all the categorical emotions
- If dimensional emotion label is not provided, write 0 for all the dimensional emotions

## Preprocessing
- All data must be processed as "Utterance" class objects
- The list of all data must be processed as "UtteranceList" class objects
