import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import numpy as np
from flask import Flask, request, jsonify
app = Flask(__name__)
from textsum.summarize import Summarizer
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
import re
from langdetect import detect
import requests
from typing import List
from nltk.tokenize import sent_tokenize, word_tokenize
import random

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

def summarize_long_string(model_name, long_string):
    summarizer = Summarizer(
        model_name_or_path=model_name,
        #token_batch_length=4096,
    )

    out_str = summarizer.summarize_string(long_string)
    return out_str

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

def translate_text(text: str, from_lang: str, to_lang: str, max_length: int = 200) -> str:
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
                url = "https://microsoft-translator-text.p.rapidapi.com/translate"
                querystring = {
                    "to[0]": to_lang,
                    "api-version": "3.0",
                    "from": from_lang,
                    "profanityAction": "NoAction",
                    "textType": "plain"
                }
                payload = [{"Text": sent_part}]
                headers = {
                    "content-type": "application/json",
                    "X-RapidAPI-Key": "bd0abfd574msh7a1c3764c533e44p178679jsn257199a6b3e2",
                    "X-RapidAPI-Host": "microsoft-translator-text.p.rapidapi.com",
                }
                response = requests.post(url, json=payload, headers=headers, params=querystring, stream=True)

                if response.status_code == 200:
                    json_data = response.json()
                    translated_text = json_data[0]['translations'][0]['text']
                    final_translated_text += translated_text + " "
                else:
                    return f"Error: Status code {response.status_code}"
        else:
            url = "https://microsoft-translator-text.p.rapidapi.com/translate"
            querystring = {
                "to[0]": to_lang,
                "api-version": "3.0",
                "from": from_lang,
                "profanityAction": "NoAction",
                "textType": "plain"
            }
            payload = [{"Text": sentence}]
            headers = {
                "content-type": "application/json",
                "X-RapidAPI-Key": random.choice(api),
                "X-RapidAPI-Host": "microsoft-translator-text.p.rapidapi.com",
            }
            response = requests.post(url, json=payload, headers=headers, params=querystring, stream=True)

            if response.status_code == 200:
                json_data = response.json()
                translated_text = json_data[0]['translations'][0]['text']
                final_translated_text += translated_text + " "
            else:
                return f"Error: Status code {response.status_code}"

    return final_translated_text.strip()

def makeSummarize(url, lang):
    video_id = extract_video_id(url)

    transcript = get_transcript(video_id)
    #download_video(video_id)

    input_text = transcript

    if lang != 'auto':
        _detected_language = lang
    else:
        _detected_language = detect_language(input_text)

    english_text = translate_text(input_text, from_lang=_detected_language, to_lang="en").replace("[ Music ]", "")

    summarizeText = summarize_long_string("pszemraj/led-base-book-summary", "The lines that people say: " + english_text)

    translated_back_text = translate_text(summarizeText, from_lang="en", to_lang=_detected_language)

    return translated_back_text

@app.route("/summarize", methods=["GET"])
def summarize():
    # URL 쿼리 스트링에서 'url'과 'lang' 파라미터를 가져옵니다.
    input_url = request.args.get('url', None)
    input_lang = request.args.get('lang', None)

    # 'url'과 'lang' 파라미터가 제공되지 않으면 오류 메시지를 반환합니다.
    if input_url is None or input_lang is None:
        return jsonify({"error": "url과 lang 매개변수가 필요합니다."}), 400

    # makeSummarize 함수를 호출하여 결과를 반환합니다.
    summary = makeSummarize(input_url, input_lang)
    return jsonify({"summary": summary})
