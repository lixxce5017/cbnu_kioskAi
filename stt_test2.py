from google.cloud import texttospeech

# 서비스 계정 키 파일 경로 설정
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:/Users/lixxc/PycharmProjects/cbnu_kioskAi/CBNU_Kiosk_main/realnew-399713-2378aee3660a.json"
print(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
key_file_path = "C:/Users/lixxc/PycharmProjects/cbnu_kioskAi/CBNU_Kiosk_main/realnew-399713-2378aee3660a.json"
# 텍스트를 음성으로 변환하는 함수 정의
def text_to_speech(text, output_file, language_code="ko"):
    client = texttospeech.TextToSpeechClient.from_service_account_file(key_file_path)

    input_text = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)

    response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)

    with open(output_file, "wb") as out:
        out.write(response.audio_content)
        print(f'Audio content written to "{output_file}".')

if __name__ == "__main__":
    # 텍스트를 음성으로 변환하고 출력 파일로 저장
    input_text = "장바구니에 담겼습니다. 메뉴추가 혹은 결제를 선택하세요"
    output_file = "장바구니.wav"
    text_to_speech(input_text, output_file)