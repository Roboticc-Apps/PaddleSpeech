import requests
import base64
import pyaudio
import logging
import argparse
import json # Untuk mengirim payload
import time # Untuk logging sederhana

# Konfigurasi logging dasar
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konfigurasi API Server default, bisa di-override oleh argumen command line
DEFAULT_SERVER_IP = "0.0.0.0"
DEFAULT_SERVER_PORT = 8092

class RealtimeTTSPlayer:
    def __init__(self, server_ip=DEFAULT_SERVER_IP, port=DEFAULT_SERVER_PORT):
        self.server_ip = server_ip
        self.port = port
        # PADDLESPEECH_API_BASE_URL tidak digunakan di sini, URL dibangun secara langsung
        self.tts_stream_url = f"http://{self.server_ip}:{self.port}/paddlespeech/tts/streaming"
        self.samplerate_url = f"http://{self.server_ip}:{self.port}/paddlespeech/tts/streaming/samplerate"
        
        self.sample_rate = self._get_server_sample_rate()
        if self.sample_rate == 0:
            # Jika tidak bisa mendapatkan SR server, kita tidak bisa melanjutkan dengan PyAudio
            raise ConnectionError(f"Could not get a valid sample rate from server at {self.samplerate_url}. Aborting.")

        self.pyaudio_instance = pyaudio.PyAudio()
        self.audio_stream = None

    def _get_server_sample_rate(self):
        response_for_sr = None # Definisikan di luar try agar bisa diakses di except JSONDecodeError
        try:
            response_for_sr = requests.get(self.samplerate_url, timeout=5) # Tambahkan timeout
            response_for_sr.raise_for_status() # Akan raise HTTPError untuk status 4xx/5xx
            data = response_for_sr.json()
            sr = int(data.get("sample_rate", 0))
            if sr > 0:
                logger.info(f"Server native streaming sample rate: {sr} Hz.")
                return sr
            else:
                logger.error(f"Server at {self.samplerate_url} did not return a valid sample_rate in JSON response: {data}")
                return 0
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed when trying to get sample rate from {self.samplerate_url}: {e}")
            return 0
        except json.JSONDecodeError as e:
            # Akses response_for_sr di sini karena sudah didefinisikan
            response_text = response_for_sr.text if response_for_sr else "No response object"
            logger.error(f"Failed to decode JSON response for sample rate from {self.samplerate_url}: {e}. Response text: {response_text[:200]}")
            return 0
        except Exception as e: # Menangkap error lain yang mungkin terjadi
            logger.error(f"An unexpected error occurred while getting sample rate from {self.samplerate_url}: {e}")
            return 0

    def _initialize_audio_stream(self):
        if self.audio_stream is None and self.sample_rate > 0:
            try:
                self.audio_stream = self.pyaudio_instance.open(
                    format=pyaudio.paInt16, # Sesuai dengan TTSHttpHandler (width 2 bytes)
                    channels=1,             # Asumsi mono untuk TTS
                    rate=self.sample_rate,
                    output=True
                )
                logger.info(f"PyAudio stream opened with SR={self.sample_rate} Hz, Format=paInt16, Channels=1.")
            except Exception as e:
                logger.error(f"Failed to open PyAudio stream: {e}", exc_info=True)
                self.audio_stream = None # Pastikan stream adalah None jika gagal dibuka

    def play_tts_stream(self, text: str, spk_id: int = 0, speed: float = 1.0, volume: float = 1.0, requested_sr: int = 0):
        if self.sample_rate == 0: # Seharusnya sudah ditangani di __init__
            logger.error("Cannot play TTS stream: server sample rate is unknown or invalid.")
            return

        self._initialize_audio_stream()
        if not self.audio_stream:
            logger.error("PyAudio stream could not be initialized. Cannot play audio.")
            return

        payload = {
            "text": text,
            "spk_id": spk_id,
            "speed": speed,
            "volume": volume,
            "sample_rate": requested_sr, # Server mungkin menggunakan SR native-nya atau yang ini
        }

        logger.info(f"Requesting TTS stream from {self.tts_stream_url} for: \"{text}\" with payload: {payload}")
        response = None
        total_bytes_played = 0 # Inisialisasi di sini agar bisa diakses di finally
        try:
            response = requests.post(self.tts_stream_url, json=payload, stream=True, timeout=10) # Tambahkan timeout
            response.raise_for_status()
            
            logger.info("Receiving audio stream...")
            start_time = time.time()
            first_chunk_received_time = None
            # total_bytes_played = 0 # Sudah diinisialisasi di atas

            for chunk_b64 in response.iter_content(chunk_size=None): # Biarkan server menentukan ukuran chunk
                if chunk_b64: 
                    if first_chunk_received_time is None:
                        first_chunk_received_time = time.time()
                        time_to_first_chunk = first_chunk_received_time - start_time
                        logger.info(f"Time to first audio chunk: {time_to_first_chunk:.3f} s. Chunk size: {len(chunk_b64)} bytes (base64).")
                    
                    try:
                        raw_audio_chunk = base64.b64decode(chunk_b64)
                        if self.audio_stream and self.audio_stream.is_active(): # Hanya tulis jika stream aktif
                           self.audio_stream.write(raw_audio_chunk)
                           total_bytes_played += len(raw_audio_chunk)
                        elif not self.audio_stream: # Jika stream belum diinisialisasi
                            logger.error("PyAudio stream is None, cannot write audio chunk.")
                            break
                        else: # Jika stream tidak aktif
                           logger.warning("PyAudio stream is not active, cannot write audio chunk.")
                           break 
                    except base64.binascii.Error as b64_err:
                        logger.error(f"Error decoding base64 chunk: {b64_err}. Chunk (first 50 bytes): {chunk_b64[:50]}")
                        break 
                    except Exception as e_write:
                        logger.error(f"Error writing to PyAudio stream: {e_write}", exc_info=True)
                        break
            
            if first_chunk_received_time is not None:
                total_stream_time = time.time() - start_time
                estimated_duration_played = total_bytes_played / (2 * self.sample_rate) if self.sample_rate > 0 else 0
                logger.info(f"Finished receiving stream. Total stream processing time: {total_stream_time:.3f} s. Bytes played: {total_bytes_played}. Estimated audio duration: {estimated_duration_played:.3f} s")
            else:
                logger.warning("No audio chunks received from server.")

        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error during streaming: {http_err}")
            if response is not None and hasattr(response, 'text') and response.text:
                logger.error(f"Response content (up to 500 chars): {response.text[:500]}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request error during streaming: {req_err}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during TTS streaming: {e}", exc_info=True)
        finally:
            if response:
                response.close()
            if self.audio_stream:
                if self.audio_stream.is_active():
                    logger.info("Waiting for PyAudio stream to finish writing buffered data...")
                    # Waktu tunggu dinamis kecil, tergantung pada jumlah data yang mungkin masih ada di buffer PyAudio
                    # Ini adalah heuristik, mungkin perlu penyesuaian.
                    # Tujuan utamanya adalah memberi waktu pada PyAudio untuk mengosongkan buffer internalnya.
                    buffer_clear_time = (total_bytes_played / (self.sample_rate * 2 * 10)) if total_bytes_played > 0 and self.sample_rate > 0 else 0.1
                    time.sleep(min(buffer_clear_time, 0.5) + 0.1) # Batasi waktu tunggu maksimal + sedikit overhead
                    self.audio_stream.stop_stream()
                self.audio_stream.close()
                logger.info("PyAudio stream closed.")
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
                logger.info("PyAudio instance terminated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realtime TTS Streaming Client for PaddleSpeech")
    parser.add_argument("text", type=str, help="Text to synthesize and stream.")
    parser.add_argument("--server_ip", type=str, default=DEFAULT_SERVER_IP, help=f"PaddleSpeech API server IP (default: {DEFAULT_SERVER_IP}).")
    parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT, help=f"PaddleSpeech API server port (default: {DEFAULT_SERVER_PORT}).")
    parser.add_argument("--spk_id", type=int, default=0, help="Speaker ID (default: 0).")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (default: 1.0).")
    parser.add_argument("--volume", type=float, default=1.0, help="Speech volume (default: 1.0).")
    parser.add_argument("--sample_rate", type=int, default=0, 
                        help="Requested sample rate for synthesis (0 for server default). Playback will use server's native streaming rate.")
    
    args = parser.parse_args()

    player = None
    try:
        player = RealtimeTTSPlayer(server_ip=args.server_ip, port=args.port)
        player.play_tts_stream(
            text=args.text,
            spk_id=args.spk_id,
            speed=args.speed,
            volume=args.volume,
            requested_sr=args.sample_rate
        )
    except ConnectionError as ce: # Ditangkap jika _get_server_sample_rate gagal
        logger.error(f"Initialization failed: {ce}")
    except Exception as e: # Menangkap error lain yang mungkin terjadi saat inisialisasi atau pemanggilan
        logger.error(f"An unhandled error occurred in main execution: {e}", exc_info=True)
