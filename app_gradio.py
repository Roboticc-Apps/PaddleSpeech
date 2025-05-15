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
    """
    Generic function to call the API.
    Returns:
        - (parsed_json_data, None) if successful and JSON is valid. 
          parsed_json_data can be any Python type corresponding to JSON types (dict, list, str, int, etc.).
        - (error_dict, "APIError") if HTTPError or RequestException. error_dict contains error details.
        - (error_dict, "InvalidJSONError") if response is not valid JSON. error_dict contains raw text.
        - (error_dict, "ClientError") for other client-side issues like unsupported method.
    """
    url = f"{PADDLESPEECH_API_BASE_URL}{endpoint}"
    response_obj = None # To store response for error reporting if available
    try:
        if method.upper() == "GET":
            response_obj = requests.get(url, params=params)
        elif method.upper() == "POST":
            response_obj = requests.post(url, json=json_payload)
        else:
            return {"error": f"Unsupported HTTP method: {method}"}, "ClientError"

        response_obj.raise_for_status() 

        try:
            data = response_obj.json()
            return data, None 
        except requests.exceptions.JSONDecodeError:
            return {"error": "Response from API was not valid JSON.", "raw_response_text": response_obj.text}, "InvalidJSONError"
    
    except requests.exceptions.HTTPError as http_err:
        error_text = response_obj.text if response_obj else "No response object available."
        return {"error": f"HTTP error: {http_err}", "response_text": error_text}, "APIError"
    except requests.exceptions.RequestException as req_err:
        return {"error": f"Request error: {req_err}"}, "APIError"
    except Exception as e: 
        return {"error": f"An unexpected client-side error occurred: {e}"}, "ClientError"

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
        return "Input text for streaming cannot be empty.", \
               format_for_json_output({"error": "Input text for streaming cannot be empty."}, "ClientError", None)

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
    api_result, error_type = call_api("POST", "/paddlespeech/tts/streaming", json_payload=payload)
    json_to_display = format_for_json_output(api_result, error_type, api_result if error_type else None)

    if error_type:
        return f"Error: {api_result.get('error', 'Unknown API error')}", json_to_display
    
    status_message = "Streaming request sent. Server response:"
    return status_message, json_to_display


def get_streaming_samplerate_func():
    api_result, error_type = call_api("GET", "/paddlespeech/tts/streaming/samplerate")
    return format_for_json_output(api_result, error_type, api_result if error_type else None)

# --- Create Gradio Interface ---
# The Gradio UI layout (gr.Blocks, gr.TabItem, etc.) remains the same as your existing code.
# Ensure all .click() calls correctly map to these updated functions and their outputs.

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
                    tts_audio_output = gr.Audio(label="Generated Audio", type="numpy") 
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
    demo.launch(share=True)
