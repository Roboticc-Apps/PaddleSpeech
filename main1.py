from paddlespeech.cli.tts.infer import TTSExecutor
from paddlespeech.server.bin.paddlespeech_client import TTSOnlineClientExecutor

tts = TTSExecutor()
tts(
    am="fastspeech2_ljspeech",
    voc="hifigan_ljspeech",
    lang="en",
    text="Life was like a box of chocolates, you never know what you're gonna get.",
    output="output.wav",
    use_onnx=True)

executor = TTSOnlineClientExecutor()
executor(
    input="Life was like a box of chocolates, you never know what you're gonna get.",
    server_ip="127.0.0.1",
    port=8092,
    protocol="http",
    spk_id=0,
    output="./output1.wav",
    play=False)
