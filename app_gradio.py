import gradio as gr
import requests
import base64
import soundfile as sf
import numpy as np
import io

# --- API Configuration ---
# Replace with your PaddleSpeech API base URL if different or publicly deployed
PADDLESPEECH_API_BASE_URL = "http://0.0.0.0:8092"

# --- API Call Helper Function ---

def call_api(method, endpoint, json_payload=None, params=None):
    """Generic function to call the API and handle basic responses."""
    url = f"{PADDLESPEECH_API_BASE_URL}{endpoint}"
    try:
        if method.upper() == "GET":
            response = requests.get(url, params=params)
        elif method.upper() == "POST":
            response = requests.post(url, json=json_payload)
        else:
            return {"error": f"Unsupported HTTP method: {method}"}, None

        response.raise_for_status() # Raise HTTPError for 4xx/5xx status codes

        # Try to parse JSON, if it fails, return raw text
        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError:
            data = response.text # If response is not JSON (e.g., a simple string)
        return data, None # data, error_message
    
    except requests.exceptions.HTTPError as http_err:
        error_message = f"HTTP error: {http_err} - Response: {response.text}"
        return None, error_message
    except requests.exceptions.RequestException as req_err:
        error_message = f"Request error: {req_err}"
        return None, error_message
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        return None, error_message

# --- Functions for each API Endpoint ---

def get_tts_help_func():
    response_data, error = call_api("GET", "/paddlespeech/tts/help")
    if error:
        return {"error": error}
    return response_data

def generate_tts_func(text, spk_id, speed, volume, sample_rate_input, save_path_server):
    if not text:
        return "Input text cannot be empty.", None, None

    payload = {
        "text": text,
        "spk_id": int(spk_id),
        "speed": float(speed),
        "volume": float(volume),
        "sample_rate": int(sample_rate_input), # This is the sample rate requested to the server
        "save_path": save_path_server if save_path_server else "output_from_gradio.wav"
    }
    
    response_data, error = call_api("POST", "/paddlespeech/tts", json_payload=payload)

    if error:
        return f"Error: {error}", None, response_data # Status, Audio, JSON Response

    if response_data and response_data.get("success"):
        result = response_data.get("result", {})
        audio_base64 = result.get("audio")
        
        if audio_base64:
            try:
                audio_bytes = base64.b64decode(audio_base64)
                # Use soundfile to read the actual sample rate and data from WAV audio bytes
                # This is safer as it gets the actual sample rate from the WAV header
                audio_data_np, actual_sr_from_wav = sf.read(io.BytesIO(audio_bytes), dtype='float32')
                
                # Gradio Audio component needs (sample_rate, numpy_array)
                # or a file path. We use the former.
                status_message = f"Success! Audio generated. Actual sample rate: {actual_sr_from_wav} Hz."
                if result.get("save_path"):
                    status_message += f" Saved on server as: {result['save_path']}"
                
                return status_message, (actual_sr_from_wav, audio_data_np), response_data
            except Exception as e:
                return f"Error processing audio: {e}", None, response_data
        else:
            return "Success, but no audio data in response.", None, response_data
    else:
        err_msg = response_data.get("message", {}).get("description", "Unknown error from API.") if response_data else "No response from API."
        return f"Failed to generate audio: {err_msg}", None, response_data


def stream_tts_func(text, spk_id, speed, volume, sample_rate_input, save_path_server):
    if not text:
        return "Input text for streaming cannot be empty.", None

    payload = {
        "text": text,
        "spk_id": int(spk_id),
        "speed": float(speed),
        "volume": float(volume),
        "sample_rate": int(sample_rate_input),
        "save_path": save_path_server if save_path_server else "stream_output_from_gradio.wav"
    }
    
    # The streaming API might have different response behavior.
    # The OpenAPI spec says a successful response is a "string" within application/json.
    response_data, error = call_api("POST", "/paddlespeech/tts/streaming", json_payload=payload)

    if error:
        return f"Error: {error}", response_data # Status, JSON Response
    
    # Since Gradio doesn't natively handle continuous streaming audio easily,
    # we'll just display the initial server response.
    status_message = f"Streaming request sent. Server response:"
    return status_message, response_data


def get_streaming_samplerate_func():
    response_data, error = call_api("GET", "/paddlespeech/tts/streaming/samplerate")
    if error:
        return {"error": error}
    return response_data

# --- Create Gradio Interface ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Gradio Interface for PaddleSpeech TTS API")
    gr.Markdown(f"Using API at: `{PADDLESPEECH_API_BASE_URL}`")

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
                    tts_audio_output = gr.Audio(label="Generated Audio", type="numpy") # type="numpy" as we send (sr, data)
                    tts_json_output = gr.JSON(label="API Response (JSON)")
            
            tts_submit_button.click(
                fn=generate_tts_func,
                inputs=[tts_text_input, tts_spk_id_input, tts_speed_input, tts_volume_input, tts_samplerate_input, tts_savepath_input],
                outputs=[tts_status_output, tts_audio_output, tts_json_output]
            )

        with gr.TabItem("Streaming TTS (Info)"):
            gr.Markdown("## Streaming TTS (Request Info)")
            gr.Markdown(
                "**Note:** This interface will send a streaming TTS request and display the initial server response. "
                "Gradio does not directly play chunked streaming audio without advanced customization."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    stream_text_input = gr.Textbox(label="Text to stream", lines=3, placeholder="Enter text here...")
                    stream_spk_id_input = gr.Number(label="Speaker ID (spk_id)", value=0, precision=0)
                    stream_speed_input = gr.Slider(label="Speed", minimum=0.1, maximum=3.0, value=1.0, step=0.1)
                    stream_volume_input = gr.Slider(label="Volume", minimum=0.1, maximum=2.0, value=1.0, step=0.1)
                    stream_samplerate_input = gr.Dropdown(label="Input Sample Rate (0 for server default)", choices=[0, 8000, 16000, 22050, 24000, 44100, 48000], value=0, type="value", allow_custom_value=True)
                    stream_savepath_input = gr.Textbox(label="Save Path on Server (optional)", placeholder="example_stream_output.wav")
                    stream_submit_button = gr.Button("Start Streaming TTS", variant="primary")
                with gr.Column(scale=3):
                    stream_status_output = gr.Textbox(label="Request Status", interactive=False)
                    stream_json_output = gr.JSON(label="Streaming API Response (JSON)")
            
            stream_submit_button.click(
                fn=stream_tts_func,
                inputs=[stream_text_input, stream_spk_id_input, stream_speed_input, stream_volume_input, stream_samplerate_input, stream_savepath_input],
                outputs=[stream_status_output, stream_json_output]
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
    demo.launch(share=True) # Set share=True to get a public link
    # For local development without a public link: demo.launch()
