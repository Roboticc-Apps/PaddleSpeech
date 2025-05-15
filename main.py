from paddlespeech.cli.tts.infer import TTSExecutor

tts = TTSExecutor()
tts(am="fastspeech2_ljspeech",voc="hifigan_ljspeech",lang="en", text="Life was like a box of chocolates, you never know what you're gonna get.", output="output.wav", use_onnx=True)