import requests
import json

def emotion_detector(text_to_analyze):
    """
    Analyzes the emotion of the provided text and returns the scores in a dictionary format.

    Args:
        text_to_analyze (str): The text to analyze for emotions.

    Returns:
        dict: A dictionary containing the emotion scores and the dominant emotion.
              Returns None if the API call fails or the response format is unexpected.
    """
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    header = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    input_json = {"raw_document": {"text": text_to_analyze}}
    response = requests.post(url, json=input_json, headers=header)

    if response.status_code == 200:
        try:
            emotion_data = response.json()
            emotions = emotion_data['emotionPredictions'][0]['emotion']

            # Extract emotion scores
            anger_score = emotions['anger']
            disgust_score = emotions['disgust']
            fear_score = emotions['fear']
            joy_score = emotions['joy']
            sadness_score = emotions['sadness']

            # Determine the dominant emotion
            dominant_emotion = max(emotions, key=emotions.get)

            # Return the formatted output
            return {
                'anger': anger_score,
                'disgust': disgust_score,
                'fear': fear_score,
                'joy': joy_score,
                'sadness': sadness_score,
                'dominant_emotion': dominant_emotion
            }
        except (KeyError, IndexError, ValueError):
            print("Error: Unexpected response format from the API.")
            return None
    elif response.status_code == 400:
        # Return dictionary with None values for 400 status code
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }
    else:
        print(f"Error: API call failed with status code {response.status_code}")
        return None
        