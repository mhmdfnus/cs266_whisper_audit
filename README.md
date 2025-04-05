[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/CClP_YEN)
# Lab 3: Choosing an audit target
A good audit target begins with a hypothesis, or at least an interest in a question, about what types of results you're likely to receive from an AI system. This might be a bias audit, like some of the discrimination concerns we've discussed in the class, or focused on the accuracy of the system or on what types of responses it's likely to generate in a specific domain. Whatever your hypothesis, you'll need to demonstrate that you have access to run chosen and controlled queries through an AI system (e.g., via an API) or that you have access to collect a large set of both inputs and outputs previously run through the system that will allow you to answer your question. You should run a pilot study with a small amount of data and/or prompts to determine how your study will work and set any parameters of the study (e.g., prompt engineering).

## What to hand in
1. A description in this README of your audit target and hypothesis, as well as a [model card](https://drive.google.com/file/d/1u6iB6WOHkZYQIxUVEBbLwrd2QNVhQGcA/view?usp=drive_link) describing (and citing) as much of the information about your audit target as you can find.

### Research Interest:

Whisper AI is an open source word speech recognition model, released by OpenAI. Currently, it’s being used extensively in hospitals [1], and given the sensitive environment, it is critical that it is done in a way that minimizes harm as much as possible. There is evidence that it hallucinates, especially in medical contexts [9] [10] [11]. We seek to test the accuracy of the model and the feasibility of deploying it in the real world, especially in sensitive contexts like hospitals. This pilot study focuses on how accurate Whisper transcriptions are across age groups, using word error rates (WER) as our evaluation metric.

We define hallucinations as words generated that were never there, this is different from misheard words which are words that were present in the actual recording and it was possible to hear them wrongly.

Our WER does not differentiate between these for our quantitative analysis but we do assess hallucinations in our qualitative analysis.

### Audit target:

Whisper AI, an open source automatic speech recognition system released by OpenAI.

### Hypothesis: 

Whisper will have a greater WER when transcribing audio from older people than when transcribing audio from younger people.

We hypothesize this because the model has been trained using YouTube [6] videos, and the demographic of people uploading content to YouTube is mostly young people. Given that the largest age demographic visiting hospitals is mostly older people [3], we want to investigate whether Whisper AI is being deployed in a context it is not suitable for. 

### Model card:

The following information is based on the model card provided by OpenAI [4] with modifications for our specific use case, based on the guidelines set by Mitchell et al, 2019 [5].

#### Model Details:

The model is developed and released by OpenAI. It is available to use via a codebase on GitHub and through API. 
The model was first released in September 2022, with subsequent versions with additional parameters being released in later years. For the pilot study, we use the base model for its lower run time, and the turbo model because OpenAI sets it as the default and from this we glean that this may be the version being most frequently used in real world contexts.
Information about other versions can be found here. 

The base version uses 74 M parameters and the turbo version uses 798 M. Both have been trained on multilingual datasets, however they are both optimized for English transcription and offer reduced accuracy when used on other languages [7].

The paper [8] released by OpenAI on Whisper mentions that they took a minimal approach to preprocessing their training data, and did not overly manipulate the data they collected from the internet. This allowed them to train the model on low resourced languages that do not have large corpora of robustly labelled data available.They take care to note that they did have measures to avoid training the model on transcripts that were not human generated, and avoided coding in pre-existing ASR generated data. 

The model is available free of charge without liability under the MIT licence.

#### Intended Use: 

Whisper is intended for general-purpose speech recognition and translation. It supports a wide range of applications, such as transcription of audio recordings, subtitles for media, real-time voice interfaces, and more. The model is intended for developers, researchers, and commercial users in need of speech-to-text functionality. That being said, OpenAI notes that Whisper is not intended for use in high-stakes domains, such as medical diagnosis or legal transcription, where transcription errors may result in significant harm. 


#### Factors:
Performance may vary across languages, accents, age groups, sex, background noise conditions, and audio quality. Although Whisper is trained on a multilingual dataset, it is optimized for English transcription. We use the base and turbo models for our analysis, analysing English transcription only. We do this because it is the primary way Whisper is being used in the natural context and if we can identify harms caused by inaccurate transcriptions, we may be able to create frameworks minimizing those harms while leveraging this in ways that bring the most benefit.

#### Metrics:

The performance of the Whisper model is generally measured using word error rate (WER), which calculates the rate of insertion, deletion, and substitution errors in the predicted transcription relative to a ground truth reference. This metric is widely used in the automatic speech recognition (ASR) community. Researchers using the model in specific contexts may consider supplementing WER with domain-specific metrics. At this point, we will rely on WER as our main metric and we may implement and use other domain-specific metrics if found appropriate in the future checkpoints of our audit.  

OpenAI evaluated Whisper on a multilingual and multitask benchmark composed of existing ASR datasets across a range of languages and conditions. These include LibriSpeech, Common Voice, TED-LIUM, and others. The choice of datasets was intended to reflect realistic and varied use cases, with differences in speaker identity, background noise, and audio recording quality. Minimal preprocessing was applied to ensure diversity and representativeness of audio input. The evaluation datasets were selected to test the model’s generalization capabilities rather than its performance on narrow or overfitted domains.

#### Training Data:

The Whisper model was trained on 680,000 hours of multilingual and multitask supervised data collected from the web. According to OpenAI, this data includes a broad variety of audio sources, ranging from podcasts and interviews to online videos. The majority of the data is in English, although the dataset spans a wide range of languages. OpenAI adopted a minimal preprocessing strategy and attempted to exclude training samples generated by ASR systems to avoid bias from synthetic transcriptions. Due to the nature of the data collection, the full composition of the training corpus is not publicly disclosed, but it includes a mix of high- and low-resource language data and a range of audio qualities and recording environments.

#### Quantitative Analyses:

OpenAI reports results across a range of tasks and conditions, with specific performance metrics for different languages and environments. While detailed intersectional analyses are limited, some unitary results suggest significantly lower performance for non-English and underrepresented languages, as well as for speech with heavy accents or in noisy environments. 

#### Ethical Considerations:

Whisper is a powerful tool for transcription, but OpenAI cautions against its use in high-stakes or safety-critical scenarios. The model may produce inaccurate transcriptions, especially in languages or dialects underrepresented in the training data, and it may misrepresent speakers with accents or in poor recording conditions. There is also the potential for the tool to be used in surveillance or for transcribing content without consent. Users are encouraged to deploy Whisper responsibly, with attention to data privacy, informed consent, and transparency in automated decision-making contexts.

Whisper uses a neural net, with a transformer architecture. These models require a large amount of computing power to deploy, and deploying them on a large scale has implications for the environment and for energy consumption. With the earlier, smaller versions being easy to run on a local computer, even for large data sets, there needs to be a cost benefit analysis of deploying the larger models in places where the smaller versions will suffice. Radford et al mentions that as model sizes increase, there are minimal gains to be had with English transcriptions. 


#### Caveats and Recommendations:

Users should be aware of the model’s limitations, including its uneven performance across languages and contexts. While Whisper performs well in English, it may introduce transcription errors that could have downstream effects, especially in applications involving medical, legal, or educational content. It is recommended that Whisper be used in conjunction with human oversight, particularly for sensitive applications. Additionally, users should consider evaluating the model in their specific context before deployment, especially if the use case involves marginalized populations or non-standard dialects.


#### Operationalizing the Dependent Variable:
We operationalize accuracy by counting the number of words that are different in the actual transcripts included with the CommonVoice dataset and the transcript generated by Whisper AI.
To achieve standardization, we tokenize each word in the transcript, removing punctuation, leading or trailing whitespaces and capitalization. The text is split into tokens using whitespaces between words as the separator, and the words are then compared. Each instance of a mismatch is counted. WER is computed using the jiwer.wer() function.

WER=(S+D+I​)/N
Where:
– S = number of substitutions
– D = number of deletions
– I = number of insertions
– N = total number of words in the reference (actual) sentence


Our independent variable is the label for age. The dataset is already labelled for age, so we simply counted the WER for each age group. For standardization, we only counted the error rate in 100 random instances for each age group. Curiously, the data set did not have any entries for voice recorded by people in their forties so that age group has been removed in the analysis.


#### Confounding variables: 
Sentence length, context of audio clip, sex, accent are not controlled for.

Due to the range of accents in CommonVoice, keeping accent random controls for any confounding the variation in accent may have on the accuracy of results.

This specific section of the CommonVoice dataset is imbalanced for sex, with more male voices than female voices and this may have a confounding effect on our study. In the future, we plan to test separately for both and compare the rate of errors to see if there is a difference between transcription accuracy. OpenAI states that whisper has been trained on data from the internet, and if there is an imbalance in male and female voices in the data set, this can create a model that is better suited to transcribe one or the other.

#### Qualitative Analysis:
Since there is documented evidence of Whisper “hallucinating”, we decided to qualitatively assess the differences in generated transcripts and actual transcripts. We did this by first running the base model on two different audios that had faint background noise but no dialogue. The first clip is a 20 second clip, and the second clip is approximately a minute long. In both cases the base model did not return any output. Doing the same with the turbo model however, did result in hallucinations. It appears that the larger more optimized model has a greater tendency to hallucinate. It is also interesting to note that although the turbo model hallucinated more, the base model had more “misheard” words, and therefore a greater WER.
The turbo version hallucinated “You” on the 20 second clip, and hallucinated “Thank you. Thank you. Thank you.” on the 1 minute clip.

When running the turbo model on a split of 200 randomly chosen clips from the CommonVoice data set we observed some hallucinations. The following is a small list of hallucinations we have picked to demonstrate this effect.

Clip: common_voice_en_41944826.mp3
Actual: Sharity-Light runs in user space rather than kernel space.
Predicted:  I'll show you the log advance in userspace rather than kernelspace.
Clip: common_voice_en_41917519.mp3
Actual: Their greatest chart success came with their summer hit single "I Love You".
Predicted:  The greatest child's success came with the summer heat singer, I Love You.

Clip: common_voice_en_42251480.mp3
Actual: Some outside scholars examining the system in depth disagree with the "official" results.
Predicted:  Sarah of Tsai's scholars examining the system in depth disagree with the official results.

Clip: common_voice_en_41929216.mp3
Actual: Sulphate and chloride sodium water is used for therapeutic baths and drinking.
Predicted:  Sulphase and Chlorotodium water is used for therapeutic baths and drinking.

Clip: common_voice_en_41962071.mp3
Actual: The rebranding process was referred internally to as Project Hyphen.
Predicted:  There were branding proxies which referred internally towards Project I-Thin.

#### Data:
Our test data set is from CommonVoice [2], a collection of crowd sourced audio clips from online contributors.

#### Limitations:
The test data set has no specific domain, our goal for the future is to extract medical context specific data from the CommonVoice corpus of data because it is labelled for sex, accent and age already. The data in the CommonVoice dataset is not all taken from speakers’ natural environments, most voice clips are audio recordings of people reading a given sentence. This could lower the ecological validity of our study. 
We plan to run the turbo model on a bigger split which has fewer misheard errors according to OpenAI. Since we were running Whisper on approximately 1 GB of data, we had CPU limitations. Future work will include running on a cloud computing service, specifically Google Colab and running the turbo version on a larger corpus of data.
The comparison process can be improved. Differences between American and British spelling are being counted as mismatches between the actual and generated transcripts which can be confounding the results of the study. This can be improved for greater validity of our results.

#### References:

[1] D. A. Soper, "Hospitals' AI transcription tools hallucinate more than others," Wired, [Online]. Available: https://www.wired.com/story/hospitals-ai-transcription-tools-hallucination/. [Accessed: Apr. 4, 2025].
[2] R. Ardila, M. Branson, K. Davis, M. Henretty, M. Kohler, J. Meyer, R. Morais, L. Saunders, F. M. Tyers, and G. Weber, "Common Voice: A massively-multilingual speech corpus," in Proc. 12th Conf. Language Resources and Evaluation (LREC 2020), 2020, pp. 4211–4215.
[3] "The age you’re most likely to be hospitalized revealed in new CDC data," MSN, [Online]. Available: https://www.msn.com/en-us/health/other/the-age-you-re-most-likely-to-be-hospitalized-revealed-in-new-cdc-data/ar-AA1sN8Nh. [Accessed: Apr. 4, 2025].
[4] "Whisper Model Card," GitHub, [Online]. Available: https://github.com/openai/whisper/blob/main/model-card.md. [Accessed: Apr. 4, 2025].
[5] M. Mitchell, S. Wu, A. Zaldivar, P. Barnes, L. Vasserman, B. Hutchinson, E. Spitzer, I. D. Raji, and T. Gebru, "Model cards for model reporting," in Proc. FAT ’19: Conf. Fairness, Accountability, and Transparency, Atlanta, GA, USA, Jan. 29–31, 2019, ACM, New York, NY, USA, pp. 1–10, doi: 10.1145/3287560.3287596.
[6] "Tech giants harvest data for artificial intelligence," The New York Times, [Online]. Available: https://www.nytimes.com/2024/04/06/technology/tech-giants-harvest-data-artificial-intelligence.html. [Accessed: Apr. 4, 2025].
[7] "Benchmarking OpenAI Whisper for non-English ASR," Deepgram, [Online]. Available: https://deepgram.com/learn/benchmarking-openai-whisper-for-non-english-asr#benchmarking-asr-for-nonenglish-languages. [Accessed: Apr. 4, 2025].
[8] "Whisper: Robust Speech Recognition via Large-Scale Self-Supervised Learning," [Online]. Available: https://cdn.openai.com/papers/whisper.pdf. [Accessed: Apr. 4, 2025].
[9] "OpenAI’s transcription tool hallucinates more than any other, experts say—but hospitals keep using it," Fortune, [Online]. Available: https://fortune.com/2024/10/26/openai-transcription-tool-whisper-hallucination-rate-ai-tools-hospitals-patients-doctors/. [Accessed: Apr. 4, 2025].
[10] "Evaluating OpenAI Whisper's Hallucinations on Different Silences," Sabrina, [Online]. Available: https://www.sabrina.dev/p/evaluating-openai-whisper-s-hallucinations-on-different-silences. [Accessed: Apr. 4, 2025].
[11] "Careless Whisper: Speech-to-Text Hallucination Harms," arXiv, [Online]. Available: https://arxiv.org/pdf/2402.08021. [Accessed: Apr. 4, 2025].
[12] "Evaluating OpenAI's Whisper ASR: Performance analysis across diverse accents and speaker traits," AIP Publishing, [Online]. Available: https://pubs.aip.org/asa/jel/article/4/2/025206/3267247. [Accessed: Apr. 4, 2025].
[13] "Hallucinations in Neural Automatic Speech Recognition: Identifying Errors and Hallucinatory Models," arXiv, [Online]. Available: https://arxiv.org/pdf/2401.01572. [Accessed: Apr. 4, 2025].
[14] "Hallucination of speech recognition errors with sequence to sequence learning," arXiv, [Online]. Available: https://arxiv.org/pdf/2103.12258. [Accessed: Apr. 4, 2025].
[15] "Where are we in audio deepfake detection? A systematic analysis over generative and detection models," arXiv, [Online]. Available: https://arxiv.org/abs/2410.04324. [Accessed: Apr. 4, 2025].


## AI Use:
AI was used for the following sections of this lab:

The Model Card:
Intended Use to Caveats and Recommendations: ChatGPT was asked to generate a sample model card for those subsections to increase efficiency. That being said, each sentence was individually inspected and edited whenever necessary. Sentences were taken out and added as appropriate. 


pilot.py:

The following function was written with AI:
def transcribe_clips(df: pd.DataFrame, clip_dir: str, model) -> List[str]:
   predictions = []
   for _, row in tqdm(df.iterrows(), total=len(df), desc="Transcribing"):
       audio_path = os.path.join(clip_dir, row["path"])
       try:
           result = model.transcribe(audio_path, language='en', fp16=False)
           predictions.append(result["text"])
       except Exception as e:
           print(f"Error transcribing {row['path']}: {e}")
           predictions.append("")
   return predictions

I was struggling to figure out how to get transcriptions for a large number of audio clips through WhisperAI, and I prompted chat gpt to generate this function for me.


2. A `pilot.py` which when run as `python pilot.py` runs your pilot experiment. If you are using a paid API, mention that in your README and we won't run the pilot script (we'll just read it), but you should still make sure it correctly runs. If you are using a paid API, you should also include a `budget.md` which estimates the budget needed to run your full audit.
3. A description in this README of your pilot experiment results. If your results include visualizations (great!) then you should also include an `analysis.py` which when run with `python analysis.py` reads in any data from your pilot experiment and generates any visualizations. Be sure to also commit and push the resulting visualizations to your repository.


