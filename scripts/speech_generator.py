from google.cloud import texttospeech

# Instantiates a client
client = texttospeech.TextToSpeechClient()
text = "Our brains are not that reliable. Sometimes we can easily mistake a 2D images as a 3D object, even under the circumstances that we know such 3D objects cannot exist in the real world. Sometimes we mistakenly feel motion out of a stationary image. So what can do with those phenomenons? Op Art, or optical art, is the art that makes illusions based on the functions of our brain. Modern magician Jerry Andrus made a series of art using optical illusions in the fifties, challenging the audience to question what they see. In order to create the artists' desired effects, most of Op Art artists use shadings and perspective shift to add depth to their paintings so that the image looks real 3D object from one angle but not the other. This works because we tend to think shades as the property of 3D object rather than a different color. Besides creating 3D effects, advanced techniques such as neon color spreading effect are also used to create an optical illusion. The illusion happens because our brains try hard to create lines out of fragmentations of colors. There are theories about why it happens, but there isn't a well-recognized theory that can explain this phenomenon. Other artists use peripheral drift illusion, 'an illusion based on temporal differences in luminance processing producing a signal that tricks the motion system', to make the audience feel motion out of a stationary picture. Also, contrasts are used not only to create virtual impacts but also to create illusions. Check this out: is the black spiraling into the center?"
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
