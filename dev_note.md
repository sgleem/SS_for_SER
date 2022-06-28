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

## 06.28 - SG
- Integrate categorical & dimensional label systems
    - Each server has its own configuration folder in "config"
    - Each Json file manages its file path and labeling system for each corpus
        - audio_path: path of the directory that contains all audios
        - label_path: csv file path that has the preprocessed labels (MSP-Podcast style)
        - data_split_type: nomenclature of each data split in label file
            - train: name of the training set
            - dev: name of the validation set
            - test: name of the test set
        - categorical: additional information for categorical emotion system
            - emo_type: array that contains all categorical emotion classes in label files
                - **Make sure to match the order of emotions**: different class order generates the different label vector even when you read the same label file
        - dimensional: additional information for dimensional emotion system
            - max_score: maximum score of emotional attribute
            - min_score: minimum score of emotional attribute
            - flip_aro: "True" if you need to flip the arousal label (only for MSP-IMPROV corpus)
                - You can omit this attribute if you don't need it
        - description: comment for each config file. It is highly recommended to write it for checking whether you put the correct configuration file or the wrong one
    - config/conf.json: Maps the argument to each configuration files
        - config_root: root directory of configuration files for your server
        - **corpus name**: file path of corpus name after the config_root
        - After you change the config/conf.json, you can specify the corpus by putting --corpus_type=="corpus name"" for train.py and test.py