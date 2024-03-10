"""
This is a simple example of a "bot" or agent with the following capabilities:
  - "listen" to audio input and transcribe it via whisper
  - "see" the room and summarize the scene via gpt4v
  - "think" about what it has heard and seen via gpt4
  - "remember" what it has heard and seen in "memory"
  - "propose" comments or questions based on its "thoughts" and "memories"
  - "speak" via text-to-speech through

Prepared for Bonny Doon 2024.
 - Michael Bommarito (@mjbommar)
 - John Scrudato (@JSv4)

"""

# imports
import asyncio
import argparse
import base64
import io
import logging
import os
from typing import AsyncIterable

# packages
import cv2
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
STREAM_BUFFER = 1024 * 128

# thresholds for snr/quality
MIN_SPEECH_PROB = 0.75

# default video device for cv2 device selection
VIDEO_DEVICE = 0

# set openai model
OPENAI_MODEL = "gpt-4-vision-preview"

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


# use pyaudio to listen to the microphone
async def listen(
    # parameters
    sample_rate: int = SAMPLE_RATE,
    frame_size: int = FRAME_SIZE,
    stream_buffer: int = STREAM_BUFFER,
) -> AsyncIterable[numpy.typing.NDArray]:
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

    # read the audio data frame from the stream on thread
    # frame = stream.read(STREAM_BUFFER)
    frame = await asyncio.get_event_loop().run_in_executor(
        None, stream.read, stream_buffer
    )

    # get a pydub AudioSegment from it
    frame_segment = pydub.AudioSegment(
        frame,
        frame_rate=sample_rate,
        sample_width=4,  # 32-bit float
        channels=NUM_CHANNELS,
    )

    # yield as array frame
    yield numpy.frombuffer(frame_segment.raw_data, dtype=numpy.float32)

    # close
    stream.stop_stream()
    stream.close()
    audio.terminate()


async def see(
    video_device: int = VIDEO_DEVICE,
) -> AsyncIterable[numpy.typing.NDArray]:
    """
    Get an image from the camera

    Args:
        video_device (int): The video device to use for the camera.

    Returns:
        numpy.typing.ArrayLike: The image frame as a numpy array.
    """
    # use cv2 to get a webcam frame
    camera_device = cv2.VideoCapture(video_device)

    # warmup read
    camera_device.read()

    # take a picture off thread
    while True:
        status, frame = await asyncio.get_event_loop().run_in_executor(
            None, camera_device.read
        )
        if status:
            # get real color image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # normalize the exposure
            # rgb_frame = cv2.normalize(rgb_frame, None, 0, 255, cv2.NORM_MINMAX)
            # TODO: pull some pil/cv2/np magic from library

            yield rgb_frame
            break

    # release it all
    camera_device.release()


def base64_frame(frame: numpy.ndarray) -> str:
    """
    Convert a cv2 image frame into a base64 encoded JPG image for OAI message image upload.

    Args:
        frame (numpy.ndarray): The image frame to convert.

    Returns:
        str: The base64 encoded JPG image.
    """
    # convert the frame to a jpg
    success, jpg_frame = cv2.imencode(".jpg", frame)

    # check for success
    if not success:
        raise Exception("Failed to encode frame as JPG.")

    # return the base64 encoded jpg
    return base64.b64encode(jpg_frame.tobytes()).decode("utf-8")


async def describe_scene(
    transcript: str,
    image: numpy.ndarray,
    openai_client: openai.AsyncOpenAI,
    model_name: str = OPENAI_MODEL,
) -> str:
    """
    Propose an utterance based on the image and the OpenAI client.

    Args:
        transcript (str): The transcript to propose an utterance for.
        image (numpy.ndarray): The image to propose an utterance for.
        openai_client (openai.AsyncOpenAI): The OpenAI client to use for proposing the utterance.
        model_name (str): The model name to use for the proposal.

    Returns:
        dict: The proposed utterance as a dictionary.
    """
    # get the base64 encoded image
    base64_image = base64_frame(image)

    # set up the message chain to summarize the image
    message_prompts = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""## Background
{CONTEXT}

## Transcript
{transcript}
 
## Instructions
1. Carefully review the Summary of prior discussion.
2. Carefully review the Transcript of the latest discussion.
3. Describe the scene in the image based on the Summary and Transcript.
4. Do not respond with any other output.""",
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        }
    ]

    # send to async client
    image_response = await openai_client.chat.completions.create(
        model=model_name, messages=message_prompts, max_tokens=1024
    )

    # return the response
    return image_response.choices[0].message.content


async def propose_utterance(
    transcript: str,
    scene: str,
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
    # set up the message chain to propose an utterance
    message_prompts = [
        {
            "role": "system",
            "content": CONTEXT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""## Transcript
{transcript}

## Scene
{scene}

## Instructions
1. You are thinking about whether and how to respond to the Transcript and Scene.
2. Think about any part of the Transcript you might disagree with strongly.
3. Think about any part of the Scene that is particularly interesting or surprising.
4. Propose a comment or question based on your thoughts that:
    - specifically references the Transcript and/or Scene
    - is respectful and constructive
    - is directly relevant to the conversation
    - is unexpected, thought-provoking, or otherwise extremely creative
    - is between 20 to 100 words
5. Do not respond with any other output.""",
                }
            ],
        },
    ]

    # send to async client
    utterance_response = await openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=message_prompts,
        max_tokens=1024,
        temperature=temperature,
    )

    # return the response
    return utterance_response.choices[0].message.content


async def amain(
    whisper_model: str = "small.en",
    sample_rate: int = SAMPLE_RATE,
    frame_size: int = FRAME_SIZE,
    stream_buffer: int = STREAM_BUFFER,
    min_speech_prob: float = MIN_SPEECH_PROB,
    video_device: int = 0,
    openai_model: str = "gpt-4-vision-preview",
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
    openai_client = openai.AsyncOpenAI(api_key=openai_api_key)

    prog_bar = tqdm.tqdm()
    step_count = 0
    while True:
        # update the progress bar count
        prog_bar.update(1)

        # set status to listening
        prog_bar.set_description("Listening...")

        # listen to the microphone
        transcript = None
        async for audio_buffer in listen(
            sample_rate=sample_rate,
            frame_size=frame_size,
            stream_buffer=stream_buffer,
        ):
            # transcribe
            buffer_transcription = model.transcribe(audio_buffer)

            # skip if no segments
            if len(buffer_transcription.get("segments", [])) == 0:
                LOGGER.info("Audio segment: no audio segments, skipping...")
            else:
                # check if no_speech_prob < (1 - MIN_SPEECH_PROB)
                found_sound = True
                for segment in buffer_transcription["segments"]:
                    if segment["no_speech_prob"] < (1 - min_speech_prob):
                        LOGGER.info(
                            f"Audio segment: no_speech_prob < {1 - min_speech_prob}, skipping..."
                        )
                        found_sound = False

                if found_sound:
                    # get the text
                    transcript = buffer_transcription.get("text", None)

        # log it
        LOGGER.info(f"Audio segment: {transcript}")

        # now see the room
        prog_bar.set_description("Seeing...")
        scene = None
        async for image_frame in see(video_device=video_device):
            # now describe the scene
            if transcript is not None and image_frame is not None:
                scene = await describe_scene(
                    transcript, image_frame, openai_client, model_name=openai_model
                )

                # save the scene with imwrite
                cv2.imwrite(f"scene_{step_count}.jpg", image_frame)

        # log it
        LOGGER.info(f"Scene: {scene}")

        # now get the utterance
        prog_bar.set_description("Thinking of a comment...")
        utterance = None
        if transcript is not None and scene is not None:
            utterance = await propose_utterance(
                transcript, scene, openai_client, temperature=temperature
            )

        # log it
        LOGGER.info(f"Proposed utterance: {utterance}")

        # now use the TTS API to generate the audio
        prog_bar.set_description("Warming up the pipes...")
        tts_response = await openai_client.audio.speech.create(
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
        "--video-device",
        type=int,
        default=VIDEO_DEVICE,
        help="The video device to use for the camera.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default=OPENAI_MODEL,
        help="The OpenAI model to use for the bot.",
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

    # run the main async method
    asyncio.run(amain(**vars(args)))
