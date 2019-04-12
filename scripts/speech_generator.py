from google.cloud import texttospeech

# Instantiates a client
client = texttospeech.TextToSpeechClient()
text = "Besides finding things on the internet, I peaked in Hannah's room and read a book called 'William Blake at the Huntington'. It was amazing. Oh, by the way. Thank you Hank for generating this handsome voice for me. Now I am going to make my plan to break into Hannah's room to learn more artists in Romantic Period!"
# Set the text input to be synthesized
synthesis_input = texttospeech.types.SynthesisInput(text=text)

# Build the voice request, select the language code ("en-US") and the ssml
# voice gender ("neutral")
voice = texttospeech.types.VoiceSelectionParams(
    language_code='en-US',
    # effectsProfileId=None,
    ssml_gender=texttospeech.enums.SsmlVoiceGender.MALE,
    name="en-US-Wavenet-D",)

# Select the type of audio file you want returned
audio_config = texttospeech.types.AudioConfig(
    audio_encoding=texttospeech.enums.AudioEncoding.MP3,
    speaking_rate=1.2,
    # audioEncoding="LINEAR16",
    pitch=0,)

# Perform the text-to-speech request on the text input with the selected
# voice parameters and audio file type
response = client.synthesize_speech(synthesis_input, voice, audio_config)

# The response's audio_content is binary.
with open('output.mp3', 'wb') as out:
    # Write the response to the output file.
    out.write(response.audio_content)
    print('Audio content written to file "output.mp3"')
