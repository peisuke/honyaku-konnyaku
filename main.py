from __future__ import division

import time
import argparse
import re
import sys
from threading import Thread
from queue import Queue

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from google.cloud import translate_v2 as translate

import pyaudio
from six.moves import queue

# Audio recording parameters
RATE = 16000
CHUNK = 1024

class pycolor:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RETURN = '\033[07m'
    ACCENT = '\033[01m'
    FLASH = '\033[05m'
    RED_FLASH = '\033[05;41m'
    END = '\033[0m'

class MicrophoneStream(object):
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1, 
            rate=self._rate,
            input=True, 
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)

class Translator:
    def __init__(self, source='ja', target='en'):
        self.client = translate.Client()
        self.message_queue = Queue()
        self.index = 0
        self.translated = {}
        self.source = source
        self.target = target
        self.is_finish = False

        for _ in range(4):
            thread = Thread(target=self.translate_worker)
            thread.setDaemon(True)
            thread.start()
        
        thread = Thread(target=self.print_worker)
        thread.start()

    def join(self):
        self.is_finish = True
        time.sleep(1)
        self.message_queue.join()

    def put(self, message):
        self.message_queue.put({'index': self.index, 'message': message})
        self.index += 1

    def print_worker(self):
        current = 0
        while not self.is_finish:
            if current in self.translated:
                print(self.translated[current]['translatedText'])
                current += 1

    def translate_worker(self):
        while not self.is_finish:
            result = self.message_queue.get()
            index = result['index']
            transcript = result['message']
            translation = self.client.translate(transcript, 
                                                source_language=self.source, 
                                                target_language=self.target, 
                                                model='nmt')
            self.translated[index] = translation
            self.message_queue.task_done()

def listen_print_loop(responses, translator):
    """Iterates through server responses and prints them.
    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    num_chars_printed = 0
    
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + '\r')
            sys.stdout.flush()
            num_chars_printed = len(transcript)
        else:
            #print(pycolor.YELLOW+transcript + overwrite_chars+pycolor.END)
            print(transcript + overwrite_chars)
            translator.put(transcript)

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if transcript == 'quit':
                print('Exiting..')
                return True

            num_chars_printed = 0
            return False
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', '-l', choices=['en', 'ja'], required=True)
    args = parser.parse_args()

    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    if args.language == 'en':
        language_code = 'en-US'  # a BCP-47 language tag
        source = 'en'
        target = 'ja'
    else:
        language_code = 'ja-JP'  # a BCP-47 language tag
        source = 'ja'
        target = 'en'

    client = speech.SpeechClient()
    translator = Translator(source=source, target=target)

    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code)
    streaming_config = types.StreamingRecognitionConfig(
        config=config,
        single_utterance=True,
        interim_results=True)

    print('Running...')
    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (types.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)

        while True:
            try:
                responses = client.streaming_recognize(streaming_config, requests, timeout=120)
                ret = listen_print_loop(responses, translator)
                if ret == True:
                    break
            except Exception as e:
                pass

    translator.join()

if __name__ == '__main__':
    main()
