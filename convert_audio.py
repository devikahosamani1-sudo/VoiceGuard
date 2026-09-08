import subprocess
from imageio_ffmpeg import get_ffmpeg_exe

input_file = "myvoice.mp4"

ffmpeg = get_ffmpeg_exe()

subprocess.run([
    ffmpeg,
    "-y",
    "-i", input_file,
    "-vn",
    "-ac", "1",
    "-ar", "16000",
    "test.wav"
], check=True)

print("Conversion completed!")
print("Created: test.wav")