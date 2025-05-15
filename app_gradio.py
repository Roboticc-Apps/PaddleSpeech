import gradio as gr
import requests
import base64
import soundfile as sf
import numpy as np
import io
import logging # Ditambahkan untuk logging
import os # Dipertahankan jika diperlukan di tempat lain
from urllib.parse import urljoin # Dipertahankan jika diperlukan di tempat lain

# Konfigurasi logging dasar
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API Configuration ---
# Replace with your PaddleSpeech API base URL if different or publicly deployed
PADDLESPEECH_API_BASE_URL = "http://0.0.0.0:8092"

# --- API Call Helper Function ---

def call_api(method, endpoint, json_payload=None, params=None, expect_json=True):
    """
    Generic function to call the API.
    Returns:
        - (parsed_json_data, None, headers) if successful, JSON is valid, and expect_json is True.
        - (response_bytes, None, headers) if successful and expect_json is False.
        - (error_dict, "APIError", headers) if HTTPError or RequestException. error_dict contains error details.
        - (error_dict, "InvalidJSONError", headers) if response is not valid JSON and expect_json is True.
        - (error_dict, "ClientError", None) for other client-side issues like unsupported method.
    """
    url = f"{PADDLESPEECH_API_BASE_URL}{endpoint}"
    response_obj = None 
    try:
        if method.upper() == "GET":
            response_obj = requests.get(url, params=params)
        elif method.upper() == "POST":
            response_obj = requests.post(url, json=json_payload) # Payload ke server tetap JSON
        else:
            return {"error": f"Unsupported HTTP method: {method}"}, "ClientError", None

        response_obj.raise_for_status() 

        if expect_json:
            try:
                data = response_obj.json()
                return data, None, response_obj.headers
            except requests.exceptions.JSONDecodeError:
                return {"error": "Response from API was not valid JSON.", "raw_response_text": response_obj.text}, "InvalidJSONError", response_obj.headers
        else:
            # Jika tidak mengharapkan JSON, kembalikan konten mentah dan header
            return response_obj.content, None, response_obj.headers # Mengembalikan bytes
    
    except requests.exceptions.HTTPError as http_err:
        error_text = response_obj.text if response_obj else "No response object available."
        headers = response_obj.headers if response_obj else None
        return {"error": f"HTTP error: {http_err}", "response_text": error_text}, "APIError", headers
    except requests.exceptions.RequestException as req_err:
        # RequestException bisa terjadi sebelum response_obj ada (mis. DNS failure)
        return {"error": f"Request error: {req_err}"}, "APIError", None
    except Exception as e: 
        return {"error": f"An unexpected client-side error occurred: {e}"}, "ClientError", None

# Helper to format data for gr.JSON output component
def format_for_json_output(api_data, error_type_from_call_api, error_payload_from_call_api):
    """
    Ensures the data returned to a gr.JSON component is a dictionary or list.
    Wraps primitives in a dictionary: {"value": primitive}.
    Passes through dictionaries, lists, and error dictionaries.
    """
    if error_type_from_call_api:
        # error_payload_from_call_api is already a dict (the error_dict from call_api)
        return error_payload_from_call_api
    
    # If no error from call_api, api_data is the successfully parsed JSON data
    if isinstance(api_data, (dict, list)):
        return api_data # Already a dict/list, suitable for gr.JSON
    else:
        # Wrap primitives (str, int, float, bool, None) in a dict
        return {"value": api_data}

# --- Functions for each API Endpoint ---

def get_tts_help_func():
    api_result, error_type = call_api("GET", "/paddlespeech/tts/help")
    # If call_api had an error, api_result is the error dictionary.
    # Otherwise, api_result is the parsed data from response.json().
    return format_for_json_output(api_result, error_type, api_result if error_type else None)

def generate_tts_func(text, spk_id, speed, volume, sample_rate_input, save_path_server):
    if not text:
        # Ensure all return paths match the number of outputs for this function
        return "Input text cannot be empty.", None, {"error": "Input text cannot be empty."}

    payload = {
        "text": text,
        "spk_id": int(spk_id),
        "speed": float(speed),
        "volume": float(volume),
        "sample_rate": int(sample_rate_input), # This is the sample rate requested to the server
        "save_path": save_path_server if save_path_server else "output_from_gradio.wav"
    }
    
    api_result, error_type = call_api("POST", "/paddlespeech/tts", json_payload=payload)

    # For the JSON output component, format the result/error
    # If call_api had an error, api_result is the error dictionary.
    # Otherwise, api_result is the parsed data (expected to be a dict for this endpoint).
    json_to_display = format_for_json_output(api_result, error_type, api_result if error_type else None)

    if error_type:
        # api_result is the error dictionary
        return f"Error: {api_result.get('error', 'Unknown API error')}", None, json_to_display

    # No error from call_api, api_result is the parsed JSON (expected to be a dict for this endpoint)
    if api_result and api_result.get("success"):
        result_data = api_result.get("result", {})
        audio_base64 = result_data.get("audio")
        
        if audio_base64:
            try:
                audio_bytes = base64.b64decode(audio_base64)
                audio_data_np, actual_sr_from_wav = sf.read(io.BytesIO(audio_bytes), dtype='float32')
                
                status_message = f"Success! Audio generated. Actual sample rate: {actual_sr_from_wav} Hz."
                if result_data.get("save_path"):
                    status_message += f" Saved on server as: {result_data['save_path']}"
                
                return status_message, (actual_sr_from_wav, audio_data_np), json_to_display
            except Exception as e:
                return f"Error processing audio: {e}", None, \
                       format_for_json_output({"error": "Audio processing failed", "details": str(e)}, "ClientError", None)
        else:
            return "Success, but no audio data in response.", None, json_to_display
    else:
        # Handle cases where API call was successful HTTP-wise, but "success" flag in JSON is false or missing
        err_msg = api_result.get("message", {}).get("description", "API indicated failure or unexpected response structure.")
        return f"Failed to generate audio: {err_msg}", None, json_to_display


def stream_tts_func(text, spk_id, speed, volume, sample_rate_input, save_path_server):
    if not text:
        return "Input text for streaming cannot be empty.", None, \
               format_for_json_output({"error": "Input text for streaming cannot be empty."}, "ClientError", None)

    # --- Get server's native streaming sample rate ---
    # Ini adalah sample rate yang DIHARAPKAN server untuk menghasilkan audio streaming.
    server_native_sr = 0
    try:
        # Gunakan call_api yang sudah ada, karena endpoint samplerate mengembalikan JSON
        sr_data, sr_error_type, _ = call_api("GET", "/paddlespeech/tts/streaming/samplerate", expect_json=True)
        if not sr_error_type and sr_data and "sample_rate" in sr_data:
            server_native_sr = int(sr_data["sample_rate"])
            logger.info(f"Successfully fetched server's native streaming sample rate: {server_native_sr} Hz.")
        else:
            err_msg = sr_data.get("error", "No sample_rate key in response") if sr_error_type or not sr_data else "No sample_rate key"
            logger.warning(f"Could not fetch server's native streaming sample rate. Error: {err_msg}")
    except Exception as e_sr:
        logger.warning(f"Exception while fetching server's native streaming sample rate: {e_sr}")

    # Jika gagal mendapatkan SR server, gunakan input pengguna jika valid, atau fallback
    if server_native_sr == 0:
        if int(sample_rate_input) != 0:
            server_native_sr = int(sample_rate_input)
            logger.warning(f"Using sample rate from user input as server_native_sr: {server_native_sr} Hz.")
        else:
            server_native_sr = 24000 # Default fallback yang masuk akal (sesuaikan jika model Anda berbeda)
            logger.warning(f"Using default fallback for server_native_sr: {server_native_sr} Hz.")
    
    payload = {
        "text": text,
        "spk_id": int(spk_id),
        "speed": float(speed),
        "volume": float(volume),
        # Kirim sample_rate_input ke server; server mungkin menggunakannya atau mengabaikannya.
        # TTSHttpHandler tidak mengirim SR dalam payload POST-nya, tapi API server mungkin menerimanya.
        "sample_rate": int(sample_rate_input), 
        "save_path": save_path_server if save_path_server else "stream_output_from_gradio.wav"
    }
    
    # Panggil API, harapkan respons biner (audio mentah), bukan JSON
    api_result_data, error_type, response_headers = call_api(
        "POST", "/paddlespeech/tts/streaming", json_payload=payload, expect_json=False
    )

    json_to_display = {} # Untuk menampilkan status atau error di komponen JSON

    if error_type:
        json_to_display = format_for_json_output(api_result_data, error_type, api_result_data if error_type else None)
        return f"Error: {api_result_data.get('error', 'Unknown API error')}", None, json_to_display

    if api_result_data and isinstance(api_result_data, bytes):
        try:
            # Logging tambahan
            if response_headers:
                logger.info(f"Streaming API Response Headers: {response_headers}")
                content_type = response_headers.get('Content-Type')
                logger.info(f"Streaming API Content-Type: {content_type}")

            # Berdasarkan perilaku TTSHttpHandler yang men-decode base64 per chunk,
            # ada kemungkinan bahwa ketika kita TIDAK men-stream (requests menggabungkan semua chunk),
            # server mungkin mengirim:
            # 1. Seluruh audio yang di-encode base64 SEKALI (ideal).
            # 2. Byte audio mentah langsung (tanpa base64 sama sekali).
            # 3. Gabungan dari string base64 per chunk (ini akan jadi base64 yang tidak valid).

            decoded_audio_bytes = None
            # is_likely_base64_encoded = False # Tidak digunakan secara eksplisit saat ini

            # Coba deteksi apakah ini mungkin base64. String base64 biasanya lebih panjang dari data aslinya.
            # Dan hanya berisi karakter tertentu. Ini bukan deteksi sempurna.
            # Karakter valid: A-Z, a-z, 0-9, +, /, =
            # Kita bisa mencoba decode, dan jika gagal, anggap itu bukan base64.
            try:
                # Coba decode. Jika ini bukan string base64 yang valid, akan error.
                # Jika ini adalah gabungan dari beberapa string base64, ini juga akan error atau salah.
                temp_decoded = base64.b64decode(api_result_data, validate=True)
                # Jika berhasil tanpa error, kemungkinan ini adalah satu blok base64.
                # Periksa apakah ukurannya masuk akal (decode base64 mengurangi ukuran sekitar 25%)
                # Heuristik: ukuran decode harus antara ~50% dan ~80% dari ukuran encode.
                # Ukuran persisnya adalah ceil(n/4)*3 - (jumlah '=' di akhir).
                # Untuk data audio yang besar, rasio mendekati 0.75.
                if len(temp_decoded) > 0 and (len(api_result_data) * 0.5 < len(temp_decoded) < len(api_result_data) * 0.85):
                    decoded_audio_bytes = temp_decoded
                    # is_likely_base64_encoded = True
                    logger.info(f"Successfully base64 decoded audio data. Original size: {len(api_result_data)}, Decoded size: {len(decoded_audio_bytes)}")
                else:
                    logger.warning(f"Base64 decode menghasilkan ukuran yang tidak terduga (original: {len(api_result_data)}, decoded: {len(temp_decoded)}) atau decoded size 0. Mengasumsikan bukan base64 yang valid atau data kosong.")
                    decoded_audio_bytes = api_result_data # Fallback ke data asli
            except base64.binascii.Error as b64_error:
                logger.warning(f"Failed to base64 decode the response as a single block: {b64_error}. Assuming raw audio bytes.")
                decoded_audio_bytes = api_result_data # Gunakan data asli jika decode gagal

            # Simpan byte yang akan diproses untuk inspeksi jika diperlukan
            # file_to_debug = "debug_stream_output_processed.raw"
            # with open(file_to_debug, "wb") as f_raw:
            #     f_raw.write(decoded_audio_bytes)
            # logger.info(f"Bytes to be processed by soundfile saved to {file_to_debug}")

            sr_to_use_for_playback = server_native_sr # Mulai dengan SR server yang diketahui/diasumsikan
            audio_data_np = None
            
            try:
                # Coba baca sebagai format standar (WAV, FLAC, dll.) terlebih dahulu
                # Ini akan berhasil jika decoded_audio_bytes adalah file WAV yang valid.
                audio_data_np, sr_from_file_std = sf.read(io.BytesIO(decoded_audio_bytes), dtype='float32')
                logger.info(f"Berhasil membaca audio (setelah decode/asumsi) sebagai format standar. SR dari file: {sr_from_file_std} Hz.")
                # Jika berhasil dibaca sebagai format standar, SR dari file lebih diutamakan
                if sr_from_file_std > 0: # Pastikan SR yang terdeteksi valid
                    sr_to_use_for_playback = sr_from_file_std
            except sf.LibsndfileError as e_std:
                logger.warning(f"Gagal membaca audio (setelah decode/asumsi) sebagai format standar (mis. WAV): {e_std}. Mencoba sebagai RAW PCM.")
                # Jika gagal, coba baca sebagai RAW PCM 16-bit (sesuai PyAudio width 2 di TTSHttpHandler)
                # Kita HARUS menggunakan sr_to_use_for_playback (dari /samplerate atau fallback) di sini.
                if sr_to_use_for_playback > 0:
                    try:
                        logger.info(f"Mencoba membaca sebagai RAW PCM, SR={sr_to_use_for_playback}, Channels=1, Subtype=PCM_16")
                        audio_data_np, _ = sf.read(io.BytesIO(decoded_audio_bytes),
                                                              samplerate=sr_to_use_for_playback,
                                                              channels=1, # Asumsi mono untuk TTS
                                                              format='RAW',
                                                              subtype='PCM_16', # Sesuai PyAudio width 2
                                                              dtype='float32') # Minta float32 untuk Gradio
                        logger.info("Berhasil membaca sebagai RAW PCM_16.")
                    except sf.LibsndfileError as e_raw:
                        logger.error(f"Gagal membaca sebagai RAW PCM_16: {e_raw}")
                        raise e_raw # Lemparkan error jika RAW PCM juga gagal
                else: # sr_to_use_for_playback masih 0 (seharusnya tidak terjadi jika fallback SR server bekerja)
                    logger.error("Sample rate server tidak dapat ditentukan, tidak dapat mencoba RAW PCM.")
                    raise e_std # Lemparkan error standar asli

            if audio_data_np is None: # Seharusnya tidak terjadi jika salah satu try di atas berhasil
                raise ValueError("Gagal memproses data audio menjadi NumPy array.")

            status_message = f"Success! Audio received. Sample rate for playback: {sr_to_use_for_playback} Hz."
            if save_path_server:
                 status_message += f" Server mungkin telah menyimpan sebagai '{save_path_server}'."

            json_to_display = {"status": "success", "message": status_message, "sample_rate": sr_to_use_for_playback}
            return status_message, (sr_to_use_for_playback, audio_data_np), json_to_display
        except Exception as e: # Menangkap semua error pemrosesan audio di sini
            status_message = f"Error processing received audio: {e}"
            logger.error(f"Error processing audio: {e}", exc_info=True)
            # Tampilkan preview dari data yang mungkin bukan base64 atau audio
            raw_preview_text = ""
            if isinstance(api_result_data, bytes): # api_result_data adalah data asli sebelum decode
                try:
                    raw_preview_text = api_result_data[:200].decode('utf-8', errors='replace')
                except: # Fallback jika decode ke utf-8 gagal
                    raw_preview_text = str(api_result_data[:200]) # Representasi string dari byte
            else: # Jika api_result_data bukan bytes (seharusnya tidak terjadi di sini)
                raw_preview_text = str(api_result_data)[:200]
            json_to_display = {"status": "error", "message": status_message, "details": str(e), "raw_preview": raw_preview_text}
            return status_message, None, json_to_display
    else:
        status_message = "API call successful but no audio data (bytes) received."
        json_to_display = {"status": "warning", "message": status_message}
        return status_message, None, json_to_display


def get_streaming_samplerate_func():
    api_result, error_type = call_api("GET", "/paddlespeech/tts/streaming/samplerate")
    return format_for_json_output(api_result, error_type, api_result if error_type else None)

# --- Create Gradio Interface ---
# The Gradio UI layout (gr.Blocks, gr.TabItem, etc.) remains the same as your existing code.
# Ensure all .click() calls correctly map to these updated functions and their outputs.

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Gradio Interface for PaddleSpeech TTS API")
    gr.Markdown(f"Using API at: `{PADDLESPEECH_API_BASE_URL}`")
    # Menghapus catatan spesifik tentang /static/ karena streaming sekarang menangani audio langsung
    # gr.Markdown(
    #     "**Penting**: Agar audio streaming dapat diputar, server API PaddleSpeech Anda "
    #     "harus dikonfigurasi untuk menyajikan file audio yang disimpan melalui URL yang dapat diakses publik "
    #     f"(misalnya, di bawah `{PADDLESPEECH_API_BASE_URL}/static/...`). "
    #     "Fungsi ini mengasumsikan konfigurasi server seperti itu."
    # )

    with gr.Tabs():
        with gr.TabItem("Standard TTS"):
            gr.Markdown("## Generate Audio (Text-to-Speech)")
            with gr.Row():
                with gr.Column(scale=2):
                    tts_text_input = gr.Textbox(label="Text to speak", lines=3, placeholder="Enter text here...")
                    tts_spk_id_input = gr.Number(label="Speaker ID (spk_id)", value=0, precision=0)
                    tts_speed_input = gr.Slider(label="Speed", minimum=0.1, maximum=3.0, value=1.0, step=0.1)
                    tts_volume_input = gr.Slider(label="Volume", minimum=0.1, maximum=2.0, value=1.0, step=0.1)
                    tts_samplerate_input = gr.Dropdown(label="Input Sample Rate (0 for server default)", choices=[0, 8000, 16000, 22050, 24000, 44100, 48000], value=0, type="value", allow_custom_value=True)
                    tts_savepath_input = gr.Textbox(label="Save Path on Server (optional)", placeholder="example_output.wav")
                    tts_submit_button = gr.Button("Generate Audio", variant="primary")
                with gr.Column(scale=3):
                    tts_status_output = gr.Textbox(label="Status", interactive=False)
                    tts_audio_output = gr.Audio(label="Generated Audio", type="numpy") 
                    tts_json_output = gr.JSON(label="API Response (JSON)")
            
            tts_submit_button.click(
                fn=generate_tts_func,
                inputs=[tts_text_input, tts_spk_id_input, tts_speed_input, tts_volume_input, tts_samplerate_input, tts_savepath_input],
                outputs=[tts_status_output, tts_audio_output, tts_json_output]
            )

        with gr.TabItem("Streaming TTS"): # Nama tab diubah sedikit
            gr.Markdown("## Streaming TTS (Request and Play)") # Judul diubah
            gr.Markdown(
                "**Note:** This interface will send a TTS request to the streaming endpoint. "
                "If successful, the received audio will be playable below."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    stream_text_input = gr.Textbox(label="Text to stream", lines=3, placeholder="Enter text here...")
                    stream_spk_id_input = gr.Number(label="Speaker ID (spk_id)", value=0, precision=0)
                    stream_speed_input = gr.Slider(label="Speed", minimum=0.1, maximum=3.0, value=1.0, step=0.1)
                    stream_volume_input = gr.Slider(label="Volume", minimum=0.1, maximum=2.0, value=1.0, step=0.1)
                    stream_samplerate_input = gr.Dropdown(label="Input Sample Rate (0 for server default/detect)", choices=[0, 8000, 16000, 22050, 24000, 44100, 48000], value=0, type="value", allow_custom_value=True)
                    stream_savepath_input = gr.Textbox(label="Save Filename on Server (optional, server-dependent)", placeholder="stream_output.wav") # Label diubah
                    stream_submit_button = gr.Button("Start Streaming TTS", variant="primary")
                with gr.Column(scale=3):
                    stream_status_output = gr.Textbox(label="Request Status", interactive=False)
                    stream_audio_output = gr.Audio(label="Streamed Audio", type="numpy") 
                    stream_json_output = gr.JSON(label="Streaming API Info (JSON)")
            
            stream_submit_button.click(
                fn=stream_tts_func,
                inputs=[stream_text_input, stream_spk_id_input, stream_speed_input, stream_volume_input, stream_samplerate_input, stream_savepath_input],
                outputs=[stream_status_output, stream_audio_output, stream_json_output]
            )

        with gr.TabItem("Help & Other Info"):
            gr.Markdown("## Additional API Information")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### TTS Help (`/paddlespeech/tts/help`)")
                    help_button = gr.Button("Get TTS Help")
                    help_output = gr.JSON(label="Help Response")
                    help_button.click(fn=get_tts_help_func, inputs=None, outputs=help_output)
                with gr.Column():
                    gr.Markdown("### Streaming Sample Rate (`/paddlespeech/tts/streaming/samplerate`)")
                    stream_sr_button = gr.Button("Get Streaming Sample Rate")
                    stream_sr_output = gr.JSON(label="Streaming Sample Rate Response")
                    stream_sr_button.click(fn=get_streaming_samplerate_func, inputs=None, outputs=stream_sr_output)

# Launch the Gradio app
# share=True will create a temporary public URL (requires internet connection)
# If your PaddleSpeech API is running locally and not accessible from the internet,
# the Gradio Relay (used by share=True) might not be able to reach your API
# unless your API is also tunneled or publicly deployed.
# If both API and Gradio run on the same machine, and you only access from local network,
# share=False or not setting it is also sufficient.
if __name__ == "__main__":
    demo.launch(share=True)
