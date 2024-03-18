"""
This is a simple example of a "bot" or agent with the following capabilities:
  - "listen" to audio input and transcribe it via whisper
  - "think" about what it has heard via gpt4
  - "remember" what it has heard and seen in "memory"
  - "propose" comments or questions based on its "thoughts" and "memories"
  - "speak" via text-to-speech through

Prepared for Bonny Doon 2024.
 - Michael Bommarito (@mjbommar)
 - John Scrudato (@JSv4)

"""

# imports
import argparse
import io
import logging
import os

# packages
import numpy
import numpy.typing
import openai
import pyaudio
import pydub
import pydub.playback
import tqdm
import whisper

# whisper sampling parameters
SAMPLE_RATE = 16000
NUM_CHANNELS = 1
FRAME_SIZE = 16384
STREAM_BUFFER = 1024 * 16

# thresholds for snr/quality
MIN_SPEECH_PROB = 0.75

# default video device for cv2 device selection
VIDEO_DEVICE = 0

# set openai model
OPENAI_MODEL = "gpt-4-0125-preview"

# context from simple text file for people following along at home
with open("context.txt", "rt", encoding="utf-8") as context_file:
    CONTEXT = context_file.read()

# initialize basic file logger so people can follow along
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()],
)


def propose_utterance(
    transcript: list[str],
    openai_client: openai.AsyncOpenAI,
    temperature: float = 0.5,
) -> str:
    """
    Ask the model to propose a question or comment based on the transcript and scene.

    Args:
        transcript (str): The transcript to propose an utterance for.
        scene (str): The scene to propose an utterance for.
        openai_client (openai.AsyncOpenAI): The OpenAI client to use for proposing the utterance.
        temperature (float): The temperature to use for the model.

    Returns:
        str: The proposed utterance.
    """
    # format the transcript from as a bulleted list of text fragments
    transcript_formatted = "\n".join([f"- {line}" for line in transcript])

    # set up the message chain to propose an utterance
    message_prompts = [
        {
            "role": "system",
            "content": CONTEXT,
        },
        {
            "role": "user",
            "content": f"""## Transcript\n{transcript_formatted}""",
        },
    ]

    # send to client
    utterance_response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=message_prompts,
        max_tokens=1024,
        temperature=temperature,
    )

    # return the response
    return utterance_response.choices[0].message.content


def listen(
    # parameters
    sample_rate: int = SAMPLE_RATE,
    frame_size: int = FRAME_SIZE,
    stream_buffer: int = STREAM_BUFFER,
) -> numpy.typing.NDArray:
    """
    Listen to the microphone

    Args:
        sample_rate (int): The sample rate to use for the audio stream.
        frame_size (int): The frame size to use for the audio stream.
        stream_buffer (int): The stream buffer size to use for the audio stream.

    Returns:
        numpy.typing.ArrayLike: The audio frame as a numpy array.
    """
    # create the audio stream
    audio = pyaudio.PyAudio()

    # open the microphone stream
    stream = audio.open(
        format=pyaudio.paFloat32,  # compatible with whisper
        channels=NUM_CHANNELS,  # mono
        rate=sample_rate,  # whisper sample rate is 16kHz
        input=True,  # input stream
        frames_per_buffer=frame_size,  # buffer size
    )

    stream_data = b""

    while True:
        # read the audio data frame from the stream on thread
        # frame = stream.read(STREAM_BUFFER)
        try:
            frame = stream.read(stream_buffer)

            # append to the stream data
            stream_data += frame
        except KeyboardInterrupt as e:
            # catch ctrl-c interrupt and break
            break

    # get a pydub AudioSegment from it
    frame_segment = pydub.AudioSegment(
        data=stream_data,
        frame_rate=sample_rate,
        sample_width=4,  # 32-bit float
        channels=NUM_CHANNELS,
    )

    # close
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # yield as array frame
    return numpy.frombuffer(frame_segment.raw_data, dtype=numpy.float32)


def main(
    whisper_model: str = "small.en",
    sample_rate: int = SAMPLE_RATE,
    frame_size: int = FRAME_SIZE,
    stream_buffer: int = STREAM_BUFFER,
    min_speech_prob: float = MIN_SPEECH_PROB,
    openai_api_key: str = os.environ.get("OPENAI_API_KEY"),
    temperature: float = 0.5,
):
    """
    The main async method for the bot.

    Args:
        whisper_model (str): The whisper model to use for transcription.
        sample_rate (int): The sample rate to use for the audio stream.
        frame_size (int): The frame size to use for the audio stream.
        stream_buffer (int): The stream buffer size to use for the audio stream.
        min_speech_prob (float): The minimum speech probability to use for the audio stream.
        video_device (int): The video device to use for the camera.
        openai_model (str): The OpenAI model to use for the bot.
        openai_api_key (str): The OpenAI API key to use for the bot.
        temperature (float): The temperature to use for the model's utterance generation.

    Returns:
        None
    """
    # load the whisper tiny model
    model = whisper.load_model(whisper_model)

    # initialize the OpenAI async client
    openai_client = openai.OpenAI(api_key=openai_api_key)

    # setup loop
    prog_bar = tqdm.tqdm()
    step_count = 0
    transcript = []
    while True:
        # update the progress bar count
        prog_bar.update(1)

        # set status to listening
        prog_bar.set_description("Listening...")

        # listen to the microphone
        try:
            audio_buffer = listen(
                sample_rate=sample_rate,
                frame_size=frame_size,
                stream_buffer=stream_buffer,
            )

            # transcribe
            buffer_transcription = model.transcribe(audio_buffer)
        except KeyboardInterrupt as e:
            # catch ctrl-c interrupt and break
            pass

        # skip if no segments
        if len(buffer_transcription.get("segments", [])) == 0:
            LOGGER.info("Audio segment: no audio segments, skipping...")
        else:
            # check if no_speech_prob < (1 - MIN_SPEECH_PROB)
            found_sound = True
            for segment in buffer_transcription["segments"]:
                if segment["no_speech_prob"] > (1 - min_speech_prob):
                    LOGGER.info(
                        f"Audio segment: no_speech_prob > {1 - min_speech_prob}, skipping..."
                    )
                    found_sound = False

            if found_sound:
                # get the text
                if buffer_transcription.get("text", None):
                    transcript.append(buffer_transcription.get("text"))

        # log it
        if len(transcript) > 0:
            LOGGER.info(f"Audio segment: {transcript[-1]}")

        # now get the utterance
        prog_bar.set_description("Thinking of a comment...")
        utterance = None
        if transcript is not None:
            utterance = propose_utterance(
                transcript, openai_client, temperature=temperature
            )

            # add to the transcript
            transcript.append("Response: " + utterance)

        # log it
        LOGGER.info(f"Proposed utterance: {utterance}")

        # now use the TTS API to generate the audio
        prog_bar.set_description("Warming up the pipes...")
        tts_response = openai_client.audio.speech.create(
            model="tts-1", voice="alloy", input=utterance
        )

        # get wav buffer
        tts_buffer = io.BytesIO(tts_response.content)

        # play the TTS response via pydub/ffmpeg
        pydub.playback.play(pydub.AudioSegment.from_file(tts_buffer))

        # increment the step count
        step_count += 1


# main method
if __name__ == "__main__":
    # setup argparse for main
    parser = argparse.ArgumentParser(description="Run the bot.")
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="small.en",
        help="The whisper model to use for transcription.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=SAMPLE_RATE,
        help="The sample rate to use for the audio stream.",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=FRAME_SIZE,
        help="The frame size to use for the audio stream.",
    )
    parser.add_argument(
        "--stream-buffer",
        type=int,
        default=STREAM_BUFFER,
        help="The stream buffer size to use for the audio stream.",
    )
    parser.add_argument(
        "--min-speech-prob",
        type=float,
        default=MIN_SPEECH_PROB,
        help="The minimum speech probability to use for the audio stream.",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY"),
        help="The OpenAI API key to use for the bot.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="The temperature to use for the model's utterance generation.",
    )

    # parse
    args = parser.parse_args()

    # run the main method
    main(**vars(args))
