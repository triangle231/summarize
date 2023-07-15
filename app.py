from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
import re
from langdetect import detect
import requests
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from collections import defaultdict
from heapq import nlargest
from flask import Flask, request, jsonify
app = Flask(__name__)

class SimpleSummarizer:
    def __init__(self):
        nltk.download('stopwords')
        self.stopwords = set(nltk.corpus.stopwords.words('english'))

    def _compute_frequencies(self, word_sent):
        freq = defaultdict(int)
        for sentence in word_sent:
            for word in sentence:
                if word not in self.stopwords:
                    freq[word] += 1
        return freq

    def summarize(self, text, n_sentences):
        sentences = nltk.sent_tokenize(text)
        word_sent = [nltk.word_tokenize(s.lower()) for s in sentences]
        self._freq = self._compute_frequencies(word_sent)
        ranking = defaultdict(int)
        for i, sentence in enumerate(word_sent):
            for word in sentence:
                if word in self._freq:
                    ranking[i] += self._freq[word]

        sents_idx = nlargest(n_sentences, ranking, key=ranking.get)
        return [sentences[j] for j in sorted(sents_idx)]
        
def summarize_text1(text, percentage):
    ss = SimpleSummarizer()
    
    # Count the number of periods in the text
    num_periods = text.count('.')
    
    # Calculate the number of sentences to include in the summary
    n_sentences = max(1, round(num_periods * (percentage / 100)))
    
    summaries = ss.summarize(text, n_sentences)
    return ' '.join(summaries)

api = ["85359c5de9mshe69974eec3d153bp1707d9jsn4b1edae7b040", "cc2001c498mshe5cc91d83fb74dfp1745fejsn881eb803db07", "97d49eacbbmsh293de6502a85130p10751ejsn23c8886f6105", "bd0abfd574msh7a1c3764c533e44p178679jsn257199a6b3e2", "e3d8d9977dmsh63df612b01e9da3p1af660jsn78c09299e6e4"]

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def extract_video_id(url: str) -> str:
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid YouTube URL")

def summarize_text2(text):
    stop_words = set(stopwords.words("english"))
    
    words = word_tokenize(text)
    words_filtered = [word for word in words if word.lower() not in stop_words]

    freq_table = {}
    for word in words_filtered:
        if word in freq_table:
            freq_table[word] += 1
        else:
            freq_table[word] = 1

    sentences = sent_tokenize(text)
    sentence_score = []

    for sentence in sentences:
        sentence_words = word_tokenize(sentence.lower())
        score = 0

        for word in sentence_words:
            if word in freq_table:
                score += freq_table[word]

        sentence_score.append((sentence, score))

    sentence_score.sort(key=lambda x: x[1], reverse=True)

    summarized_sentences = []
    for i in range(min(5, len(sentence_score))):
        summarized_sentences.append(sentence_score[i][0])

    return ' '.join(summarized_sentences)

def format_time(time):
    hours, remainder = divmod(time, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{float(seconds):06.2f}"

def get_transcript(video_id):
    transcript = ""
    try:
        top_languages = [
            'ko',  # Korean
            'en',  # English
            'zh-Hans',  # Chinese Simplified
            'zh-Hant',  # Chinese Traditional
            'es',  # Spanish
            'hi',  # Hindi
            'ar',  # Arabic
            'pt',  # Portuguese
            'ru',  # Russian
            'fr',  # French
            'de'   # German
        ]
        results = YouTubeTranscriptApi.get_transcript(video_id, languages=top_languages)
        for r in results:
            start_time = format_time(r['start'])
            end_time = format_time(r['start'] + r['duration'])
            transcript += f"{r['text']}"#{start_time} - {end_time} \n
    except Exception as e:
        transcript = "자동 생성된 자막을 찾을 수 없습니다."
    return transcript

def download_video(video_id, download_path="."):
    ydl_opts_video = {
        "format": "bestvideo",
        "outtmpl": f"{download_path}/%(title)s_video.%(ext)s",
    }

    ydl_opts_audio = {
        "format": "bestaudio",
        "outtmpl": f"{download_path}/%(title)s_audio.%(ext)s",
    }

    with yt_dlp.YoutubeDL(ydl_opts_video) as ydl_video:
        ydl_video.download([f"http://www.youtube.com/watch?v={video_id}"])

    with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl_audio:
        ydl_audio.download([f"http://www.youtube.com/watch?v={video_id}"])

def translate_text(text: str, from_lang: str, to_lang: str, max_length: int = 1000) -> str:
    text_sentences = sent_tokenize(text)

    final_translated_text = ""

    for sentence in text_sentences:
        if len(sentence) > max_length:
            # 방법 1: 단어 단위로 나누어 길이가 max_length를 넘지 않게 분할
            words = word_tokenize(sentence)
            sent_parts = []
            part = ""
            for word in words:
                if len(part) + len(word) <= max_length:
                    part += word + " "
                else:
                    sent_parts.append(part)
                    part = word + " "
            sent_parts.append(part)

            # 방법 2: 길이를 초과한 부분은 건너뛰기
            sent_parts = [sentence[:max_length]]

            for sent_part in sent_parts:
                url = "https://playentry.org/api/expansionBlock/papago/translate/n2mt"
                params = {
                    "text": sent_part,
                    "target": to_lang,
                    "source": from_lang,
                }

                response = requests.get(url, params=params)

                if response.status_code == 200:
                    json_data = response.json()
                    translated_text = json_data["translatedText"]
                    final_translated_text += translated_text + " "
                else:
                    return f"Error: Status code {response.status_code}"
        else:
            url = "https://playentry.org/api/expansionBlock/papago/translate/n2mt"
            params = {
                "text": sentence,
                "target": to_lang,
                "source": from_lang,
            }

            response = requests.get(url, params=params)

            if response.status_code == 200:
                json_data = response.json()
                translated_text = json_data["translatedText"]
                final_translated_text += translated_text + " "
            else:
                return f"Error: Status code {response.status_code}"

    return final_translated_text.strip()

def makeSummarize(url, lang, Percentage):
    ss = SimpleSummarizer()

    video_id = extract_video_id(url)

    transcript = get_transcript(video_id)
    #download_video(video_id)

    input_text = transcript.replace("[음악]", "")

    if lang != 'auto':
        _detected_language = lang
    else:
        _detected_language = detect_language(input_text)

    english_text = translate_text(input_text, from_lang=_detected_language, to_lang="en")

    summarizeText = summarize_text2("(The lines that people say)" + summarize_text1(english_text, Percentage))#summarize_long_string("pszemraj/led-base-book-summary", "The lines that people say:" + english_text)

    translated_back_text = translate_text(summarizeText, from_lang="en", to_lang=_detected_language)

    return translated_back_text

@app.route("/summarize", methods=["GET"])
def summarize():
    input_url = request.args.get('url', None)
    input_lang = request.args.get('lang', None)
    input_percentage = request.args.get('percentage', None)

    if input_url is None or input_lang is None or input_percentage is None:
        return jsonify({"error": "url과 lang과 percentage 매개변수가 필요합니다."}), 400

    summary = makeSummarize(input_url, input_lang, input_percentage)
    return jsonify({"summary": summary})
