document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('text-input');
    const spkIdInput = document.getElementById('spk-id-input');
    const speedInput = document.getElementById('speed-input');
    const volumeInput = document.getElementById('volume-input');
    const sampleRateInput = document.getElementById('sample-rate-input');
    const playButton = document.getElementById('play-button');
    const stopButton = document.getElementById('stop-button');
    const statusDiv = document.getElementById('status');

    // Elemen untuk metrik performa
    const performanceMetricsDiv = document.getElementById('performance-metrics');
    const metricTtfb = document.getElementById('metric-ttfb');
    const metricChunks = document.getElementById('metric-chunks');
    const metricBytes = document.getElementById('metric-bytes');
    const metricAudioDuration = document.getElementById('metric-audio-duration');
    const metricTotalTime = document.getElementById('metric-total-time');

    let audioContext;
    let sampleRateFromServer = 0;
    let nextPlayTime = 0;
    let isPlaying = false;
    let abortController = null; // Untuk membatalkan fetch
    let audioSourceNodes = []; // Menyimpan semua source node yang dijadwalkan

    const PADDLESPEECH_API_BASE_URL = "http://0.0.0.0:8092"; // Sesuaikan jika perlu

    async function getStreamingSampleRate() {
        statusDiv.textContent = "Fetching server sample rate...";
        try {
            const response = await fetch(`${PADDLESPEECH_API_BASE_URL}/paddlespeech/tts/streaming/samplerate`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            if (data && data.sample_rate) {
                console.log(`Server sample rate: ${data.sample_rate}`);
                statusDiv.textContent = `Server sample rate: ${data.sample_rate} Hz. Ready.`;
                return parseInt(data.sample_rate, 10);
            } else {
                console.error("Could not get sample rate from server response:", data);
                statusDiv.textContent = "Error: Could not get valid sample rate from server.";
                return 24000; // Fallback default
            }
        } catch (error) {
            console.error("Error fetching sample rate:", error);
            statusDiv.textContent = `Error fetching sample rate: ${error.message}`;
            return 24000; // Fallback default
        }
    }
    
    function initAudioContext(serverSampleRate) {
        if (audioContext && audioContext.state !== 'closed' && audioContext.sampleRate !== serverSampleRate) {
            console.warn(`Sample rate mismatch. Closing old AudioContext (SR: ${audioContext.sampleRate}) and creating new (SR: ${serverSampleRate}).`);
            audioContext.close().catch(e => console.warn("Error closing previous AudioContext:", e));
            audioContext = null;
        }
        if (!audioContext || audioContext.state === 'closed') {
            audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: serverSampleRate
            });
            console.log(`AudioContext initialized/reinitialized with sample rate: ${audioContext.sampleRate}`);
        }
        nextPlayTime = audioContext.currentTime;
        audioSourceNodes = []; // Reset antrian node
    }

    function pcmInt16ToFloat32(inputInt16Array) {
        const outputFloat32Array = new Float32Array(inputInt16Array.length);
        for (let i = 0; i < inputInt16Array.length; i++) {
            outputFloat32Array[i] = inputInt16Array[i] / 32768.0;
        }
        return outputFloat32Array;
    }

    function scheduleChunkPlayback(pcmInt16Data) {
        if (!audioContext || audioContext.state === 'closed' || pcmInt16Data.length === 0) return;

        const float32AudioData = pcmInt16ToFloat32(pcmInt16Data);
        const audioBuffer = audioContext.createBuffer(1, float32AudioData.length, audioContext.sampleRate);
        audioBuffer.getChannelData(0).set(float32AudioData);

        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);

        const currentTime = audioContext.currentTime;
        if (nextPlayTime < currentTime) {
            nextPlayTime = currentTime;
        }
        
        source.start(nextPlayTime);
        audioSourceNodes.push(source); // Simpan untuk kemungkinan stop
        nextPlayTime += audioBuffer.duration;
    }

    function stopCurrentStream() {
        if (abortController) {
            abortController.abort();
            console.log("Fetch aborted.");
        }
        if (audioContext && audioSourceNodes.length > 0) {
            audioSourceNodes.forEach(node => {
                try {
                    node.stop();
                } catch (e) { /* Mungkin sudah berhenti atau error lain, abaikan */ }
            });
            console.log("Scheduled audio sources stopped.");
        }
        audioSourceNodes = [];
        isPlaying = false;
        playButton.disabled = false;
        stopButton.disabled = true;
        playButton.textContent = "Play Stream";
        statusDiv.textContent = "Stream stopped.";
    }

    playButton.addEventListener('click', async () => {
        if (isPlaying) return; 

        // Reset dan sembunyikan metrik performa sebelum memulai
        performanceMetricsDiv.style.display = 'none';
        metricTtfb.textContent = '-';
        metricChunks.textContent = '-';
        metricBytes.textContent = '-';
        metricAudioDuration.textContent = '-';
        metricTotalTime.textContent = '-';

        const text = textInput.value.trim();
        const spk_id = parseInt(spkIdInput.value, 10);
        const speed = parseFloat(speedInput.value);
        const volume = parseFloat(volumeInput.value);
        const requested_sr = parseInt(sampleRateInput.value, 10);

        if (!text) {
            statusDiv.textContent = "Please enter some text.";
            return;
        }

        isPlaying = true;
        playButton.disabled = true;
        stopButton.disabled = false;
        playButton.textContent = "Processing...";
        
        if (!sampleRateFromServer || (audioContext && audioContext.sampleRate !== sampleRateFromServer && audioContext.state !== 'closed')) {
            sampleRateFromServer = await getStreamingSampleRate();
        } else if (audioContext && audioContext.state === 'closed'){
             sampleRateFromServer = await getStreamingSampleRate(); // Re-fetch if context was closed
        }


        if (!sampleRateFromServer) { 
            isPlaying = false;
            playButton.disabled = false;
            stopButton.disabled = true;
            playButton.textContent = "Play Stream";
            return;
        }
        initAudioContext(sampleRateFromServer);

        const payload = {
            text: text,
            spk_id: spk_id,
            speed: speed,
            volume: volume,
            sample_rate: requested_sr 
        };

        statusDiv.textContent = "Requesting TTS stream...";
        console.log("Requesting TTS stream with payload:", payload);
        abortController = new AbortController(); 

        try {
            const response = await fetch(`${PADDLESPEECH_API_BASE_URL}/paddlespeech/tts/streaming`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', },
                body: JSON.stringify(payload),
                signal: abortController.signal 
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }

            if (!response.body) {
                throw new Error("ReadableStream not available.");
            }

            statusDiv.textContent = "Receiving audio stream...";
            playButton.textContent = "Playing...";
            const reader = response.body.getReader();

            // Performance metrics initialization
            let firstChunkReceivedTime = null;
            let lastChunkTime = performance.now(); // Initialize lastChunkTime here
            let totalAudioBytes = 0;
            let chunkCount = 0;
            const streamStartTime = performance.now(); // Initialize streamStartTime here
            
            while (true) {
                if (!isPlaying) { 
                    if (!reader.closed) await reader.cancel("Stream stopped by user");
                    console.log("Streaming loop exited due to stop signal.");
                    break;
                }

                const { done, value } = await reader.read();
                if (done) {
                    const streamEndTime = performance.now();
                    console.log("Stream finished.");
                    statusDiv.textContent = "Stream finished.";
                    
                    const totalStreamDuration = (streamEndTime - streamStartTime) / 1000;
                    // Pastikan sampleRateFromServer valid sebelum pembagian
                    const audioDuration = (sampleRateFromServer > 0 && totalAudioBytes > 0) ? totalAudioBytes / (sampleRateFromServer * 2) : 0; // 2 bytes per sample (Int16)
                    
                    console.log(`Total chunks received: ${chunkCount}`);
                    console.log(`Total raw audio bytes: ${totalAudioBytes}`);
                    console.log(`Estimated audio duration: ${audioDuration.toFixed(3)}s`);
                    console.log(`Total streaming time (network + processing): ${totalStreamDuration.toFixed(3)}s`);
                    
                    // Update metrik di UI
                    performanceMetricsDiv.style.display = 'block';
                    metricChunks.textContent = chunkCount;
                    metricBytes.textContent = totalAudioBytes;
                    metricAudioDuration.textContent = audioDuration.toFixed(3);
                    metricTotalTime.textContent = totalStreamDuration.toFixed(3);
                    
                    if (firstChunkReceivedTime) {
                         // Waktu dari awal request hingga chunk terakhir diterima (bukan dari chunk pertama)
                         const timeToLastChunkFromStart = (streamEndTime - streamStartTime) / 1000; // Ini sama dengan totalStreamDuration
                         console.log(`Time from request to last chunk: ${timeToLastChunkFromStart.toFixed(3)}s`);
                    }
                    break;
                }
                
                const currentTime = performance.now();
                const timeSinceLastChunk = (currentTime - lastChunkTime) / 1000;
                lastChunkTime = currentTime;
                chunkCount++;

                if (firstChunkReceivedTime === null) {
                    firstChunkReceivedTime = currentTime;
                    const timeToFirstChunk = (firstChunkReceivedTime - streamStartTime) / 1000;
                    console.log(`Time to first audio chunk: ${timeToFirstChunk.toFixed(3)}s`);
                    statusDiv.textContent = `Receiving audio (First chunk: ${timeToFirstChunk.toFixed(3)}s)...`;
                    
                    // Tampilkan TTFB di UI
                    performanceMetricsDiv.style.display = 'block'; // Tampilkan box metrik
                    metricTtfb.textContent = timeToFirstChunk.toFixed(3);
                }
                
                try {
                    const base64ChunkString = new TextDecoder().decode(value); // value is Uint8Array
                    const binaryString = atob(base64ChunkString); 
                    const len = binaryString.length;
                    const bytes = new Uint8Array(len);
                    for (let i = 0; i < len; i++) {
                        bytes[i] = binaryString.charCodeAt(i);
                    }
                    
                    totalAudioBytes += bytes.byteLength;
                    const pcmInt16 = new Int16Array(bytes.buffer);

                    console.log(`Chunk ${chunkCount}: Received ${value.byteLength} bytes (base64), Decoded to ${bytes.byteLength} PCM bytes. Time since last: ${timeSinceLastChunk.toFixed(3)}s`);
                    scheduleChunkPlayback(pcmInt16);

                } catch (e) {
                    console.error("Error processing chunk:", e, "Chunk as text:", new TextDecoder().decode(value, {stream: true}));
                    statusDiv.textContent = `Error processing audio chunk: ${e.message}`;
                    if (!reader.closed) await reader.cancel("Error processing chunk");
                    break; 
                }
            }
        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('Fetch aborted by user.');
                statusDiv.textContent = 'Stream stopped by user.';
            } else {
                console.error("Streaming TTS failed:", error);
                statusDiv.textContent = `Error: ${error.message}`;
            }
        } finally {
            if (isPlaying) { 
                isPlaying = false;
                playButton.disabled = false;
                stopButton.disabled = true;
                playButton.textContent = "Play Stream";
            }
            abortController = null; 
        }
    });

    stopButton.addEventListener('click', () => {
        stopCurrentStream();
    });

    // Ambil sample rate server saat halaman dimuat
    getStreamingSampleRate().then(sr => {
        sampleRateFromServer = sr;
    });
});
