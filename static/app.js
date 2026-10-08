let spectrogramRendered = false;
    /* =========================================================
       CANVAS SETUP
    ========================================================= */
    const waveformCanvas = document.getElementById("waveform-canvas");
    const waveformCtx = waveformCanvas.getContext("2d");
    const spectrogramCanvas = document.getElementById("spectrogram-canvas");
    const spectrogramCtx = spectrogramCanvas.getContext("2d");
    let visualizerActive = false;

    /* =========================================================
       REAL AUDIO & SPECTROGRAM DATA STORAGE
    ========================================================= */
    let realWaveformData = [];
    let realSpectrogramData = [];
    let audioDataPolling = null;
    // Seconds of audio the backend has actually recorded, drives the playhead
    let liveElapsed = 0;
    let liveDuration = 5;

    /* =========================================================
       CANVAS SIZING HELPER (Supervisor's standard)
    ========================================================= */
    function resizeCanvas(canvas) {
        const r = canvas.getBoundingClientRect();
        const d = devicePixelRatio || 1;
        canvas.width = Math.max(1, Math.round(r.width * d));
        canvas.height = Math.max(1, Math.round(r.height * d));
    }

    function resizeCanvases() {
        resizeCanvas(waveformCanvas);
        resizeCanvas(spectrogramCanvas);
        paintWave();
        paintSpectrogram();
    }
    window.addEventListener("resize", resizeCanvases);
    resizeCanvases();

    /* =========================================================
       FETCH REAL AUDIO DATA FROM BACKEND (/audio-data)
    ========================================================= */
    async function updateAudioData() {
        try {
            const response = await fetch("/audio-data", { cache: "no-store" });
            if (!response.ok) return;
            const data = await response.json();

            liveElapsed = Number(data.elapsed) || 0;
            liveDuration = Number(data.duration) || 5;

            if (Array.isArray(data.waveform) && data.waveform.length > 0) {
                realWaveformData = data.waveform;
                paintWave();
            }

            if (Array.isArray(data.spectrogram) && data.spectrogram.length > 0 &&
                Array.isArray(data.spectrogram[0]) && data.spectrogram[0].length > 0) {
                realSpectrogramData = data.spectrogram;
                paintSpectrogram();
            }
        } catch (error) {
            console.log("Audio data connection error:", error);
        }
    }

    function startAudioDataPolling() {
        if (audioDataPolling) return;
        updateAudioData();
        audioDataPolling = setInterval(updateAudioData, 100);
    }

    function stopAudioDataPolling() {
        if (audioDataPolling) {
            clearInterval(audioDataPolling);
            audioDataPolling = null;
        }
    }

/* =========================================================
       SOLID WAVEFORM
    ========================================================= */
    function paintWave() {
        const w = waveformCanvas.width;
        const h = waveformCanvas.height;
        
        // Background gradient
        const gradient = waveformCtx.createLinearGradient(0, 0, 0, h);
        gradient.addColorStop(0, "#F9FCFA");
        gradient.addColorStop(0.5, "#EEF7F1");
        gradient.addColorStop(1, "#E6F2EA");
        waveformCtx.fillStyle = gradient;
        waveformCtx.fillRect(0, 0, w, h);

        // Center line
        waveformCtx.strokeStyle = "#d8e6da";
        waveformCtx.lineWidth = 1;
        waveformCtx.beginPath();
        waveformCtx.moveTo(0, h / 2);
        waveformCtx.lineTo(w, h / 2);
        waveformCtx.stroke();

        const isActive = typeof visualizerActive !== 'undefined' ? visualizerActive : false;

        if (typeof realWaveformData === 'undefined' || !realWaveformData || realWaveformData.length === 0) {
            waveformCtx.font = "600 11px Inter, sans-serif";
            waveformCtx.fillStyle = "#789083";
            waveformCtx.fillText(isActive ? "LIVE AUDIO RECORDING" : "", 30, 24);
            return;
        }

        const data = realWaveformData;
        const totalColumns = Math.floor(w / 4); // Creates a dense, solid column layout
        const centerY = h / 2;

        waveformCtx.fillStyle = '#1a7548';

        for (let x = 0; x < totalColumns; x++) {
            let maxVal = 0;
            // The points span the whole recording time; map each column onto them
            const first = Math.floor(x * data.length / totalColumns);
            const last = Math.max(first + 1, Math.floor((x + 1) * data.length / totalColumns));
            for (let i = first; i < Math.min(data.length, last); i++) {
                const val = Math.abs(Number(data[i]) || 0);
                if (val > maxVal) maxVal = val;
            }

            const heightFactor = Math.max(maxVal * centerY * 0.85, 2);
            const px = x * (w / totalColumns);
            const barWidth = Math.max(1, (w / totalColumns) - 1);
            
            // Draw solid vertical blocks
            waveformCtx.fillRect(px, centerY - heightFactor, barWidth, heightFactor * 2);
        }

        // Playhead = how much audio the microphone has recorded so far,
        // so it moves in step with the sound from the speakers
        const currentProgress = liveDuration > 0 ? Math.min(1, liveElapsed / liveDuration) : 0;

        const curPx = currentProgress * w;
        waveformCtx.fillStyle = '#22794b22';
        waveformCtx.fillRect(0, 0, curPx, h);

        waveformCtx.strokeStyle = '#ed8b40';
        waveformCtx.lineWidth = 3 * (devicePixelRatio || 1);
        waveformCtx.beginPath();
        waveformCtx.moveTo(curPx, 0);
        waveformCtx.lineTo(curPx, h);
        waveformCtx.stroke();

        waveformCtx.font = "600 11px Inter, sans-serif";
        waveformCtx.fillStyle = isActive ? "$155C35" : "#789083";
        waveformCtx.fillText(isActive ? "LIVE AUDIO RECORDING" : "", 30, 24);
    }


/* =========================================================
        STATIC SPECTROGRAM 
    ========================================================= */
    function paintSpectrogram() {
        if (!spectrogramCanvas || !spectrogramCtx) return;
        
        const w = spectrogramCanvas.width;
        const h = spectrogramCanvas.height;

        // Rich dark background matching supervisor theme
        spectrogramCtx.fillStyle = "#0c1812";
        spectrogramCtx.fillRect(0, 0, w, h);

        if (!realSpectrogramData || realSpectrogramData.length === 0 || !realSpectrogramData[0]) {
            spectrogramCtx.font = "600 11px Inter, sans-serif";
            spectrogramCtx.fillStyle = "#5a7a69";
            spectrogramCtx.fillText("LIVE SPECTRAL ACTIVITY", 30, 24);
            return;
        }

        const cols = realSpectrogramData.length;
        const rows = realSpectrogramData[0].length;
        if (rows === 0) return;

        // Create an offscreen buffer for the raw data matrix
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = cols;
        tempCanvas.height = rows;
        const tempCtx = tempCanvas.getContext('2d');
        const imgData = tempCtx.createImageData(cols, rows);
        const pixels = imgData.data;

        for (let x = 0; x < cols; x++) {
            const colData = realSpectrogramData[x] || [];
            for (let y = 0; y < rows; y++) {
                const val = Number(colData[y]) || 0;
                // Invert y so low frequencies sit at the bottom
                const targetRow = rows - 1 - y;
                const index = (targetRow * cols + x) * 4;

                const intensity = Math.min(1, Math.max(0, val));

                if (intensity > 0.2) {
                    // Smooth green gradient mapping (no hard squares)
                    pixels[index]     = Math.floor(15 + intensity * 55);   // Red
                    pixels[index + 1] = Math.floor(90 + intensity * 165);  // Green
                    pixels[index + 2] = Math.floor(35 + intensity * 45);   // Blue
                    pixels[index + 3] = Math.floor(intensity * 255);       // Alpha
                } else {
                    // Seamless dark background blend
                    pixels[index]     = 12;
                    pixels[index + 1] = 24;
                    pixels[index + 2] = 18;
                    pixels[index + 3] = 255;
                }
            }
        }

        tempCtx.putImageData(imgData, 0, 0);

        // Turn on browser high-quality smoothing to completely melt away grid squares
        spectrogramCtx.imageSmoothingEnabled = true;
        spectrogramCtx.imageSmoothingQuality = 'high';

        // Position it statically at the bottom portion of the canvas (like supervisor style)
        const activeHeight = h * 0.42; 
        const targetY = h - activeHeight;
        
        spectrogramCtx.drawImage(tempCanvas, 0, 0, cols, rows, 0, targetY, w, activeHeight);

        // Sleek Status Label Overlay
        spectrogramCtx.font = "600 11px Inter, sans-serif";
        spectrogramCtx.fillStyle = "#7ca38f";
        spectrogramCtx.fillText("LIVE SPECTRAL ACTIVITY", 30, 24);
    }



    /* =========================================================
       VISUALIZER STATE
    ========================================================= */
    function setVisualizerState(active) {
        visualizerActive = active;
        const waveStatus = document.getElementById("waveLiveStatus");
        const spectStatus = document.getElementById("spectLiveStatus");
        const signalDot = document.getElementById("signalDot");
        const signalStatusText = document.getElementById("signalStatusText");

        if (active) {
            if (waveStatus) waveStatus.classList.add("active");
            if (spectStatus) spectStatus.classList.add("active");
            if (signalDot) signalDot.style.background = "#278A4F";
            if (signalStatusText) signalStatusText.innerText = "MIC / RECORDING";
            startAudioDataPolling();
        } else {
            if (waveStatus) waveStatus.classList.remove("active");
            if (spectStatus) spectStatus.classList.remove("active");
            if (signalDot) signalDot.style.background = "#718B7D";
            if (signalStatusText) signalStatusText.innerText = "MIC / AUDIO";
            stopAudioDataPolling();
            realWaveformData = [];
            realSpectrogramData = [];
            paintWave();
            paintSpectrogram();
        }
    }

    /* =========================================================
       AI PREDICTION UI STATE & ANIMATION
    ========================================================= */
    let predictionRunning = false;
    let analysisTimer = null;
    let analysisSeconds = 15;
    const predictionLoading = document.getElementById("predictionLoading");
    const probabilityResults = document.getElementById("probabilityResults");
    const analysisMessage = document.getElementById("analysisMessage");
    const analysisCount = document.getElementById("analysisCount");

    function showAnalysisAnimation() {
        predictionRunning = true;
        analysisSeconds = 5;
        if (predictionLoading) predictionLoading.classList.add("active");
        if (probabilityResults) probabilityResults.classList.remove("visible");
        if (analysisMessage) analysisMessage.innerText = "Listening to the horse sound...";
        if (analysisCount) analysisCount.innerText = "5 seconds remaining";
        
        if (analysisTimer) clearInterval(analysisTimer);
        analysisTimer = setInterval(() => {
            analysisSeconds--;
            if (analysisSeconds > 0) {
                if (analysisMessage) analysisMessage.innerText = analysisSeconds > 2 ? "Processing acoustic signal..." : "AI classification in progress...";
                if (analysisCount) analysisCount.innerText = analysisSeconds + (analysisSeconds === 1 ? " second remaining" : " seconds remaining");
            } else {
                if (analysisMessage) analysisMessage.innerText = "Finalising AI prediction...";
                if (analysisCount) analysisCount.innerText = "Waiting for model result...";
            }
        }, 1000);
    }

    function showPredictionResults() {
        predictionRunning = false;
        if (analysisTimer) {
            clearInterval(analysisTimer);
            analysisTimer = null;
        }
        if (predictionLoading) predictionLoading.classList.remove("active");
        if (probabilityResults) probabilityResults.classList.add("visible");
    }

    function resetPredictionResults() {
        predictionRunning = false;
        if (analysisTimer) {
            clearInterval(analysisTimer);
            analysisTimer = null;
        }
        if (predictionLoading) predictionLoading.classList.remove("active");
        if (probabilityResults) probabilityResults.classList.remove("visible");
        analysisSeconds = 5;
        if (analysisMessage) analysisMessage.innerText = "Listening to the horse sound...";
        if (analysisCount) analysisCount.innerText = "5 seconds remaining";
        setConcealed(false);
    }

    /* =========================================================
       REVEAL OVERLAY
       The result is hidden behind an overlay until the audience
       has guessed; the button or the Space key reveals it.
    ========================================================= */
    const resultStack = document.getElementById("resultStack");
    const revealOverlay = document.getElementById("revealOverlay");
    const revealBtn = document.getElementById("revealBtn");
    let resultConcealed = false;
    let swallowSpaceKeyup = false;

    function setConcealed(concealed) {
        resultConcealed = concealed;
        if (resultStack) resultStack.classList.toggle("concealed", concealed);
        if (revealOverlay) revealOverlay.setAttribute("aria-hidden", concealed ? "false" : "true");
    }

    function concealResult() {
        predictionRunning = false;
        if (analysisTimer) {
            clearInterval(analysisTimer);
            analysisTimer = null;
        }
        if (predictionLoading) predictionLoading.classList.remove("active");
        if (probabilityResults) probabilityResults.classList.remove("visible");
        setConcealed(true);
        if (revealBtn) revealBtn.focus({ preventScroll: true });
    }

    function revealResult() {
        if (!resultConcealed) return;
        setConcealed(false);
        showPredictionResults();
    }

    if (revealBtn) revealBtn.addEventListener("click", revealResult);

    document.addEventListener("keydown", (event) => {
        if (event.code !== "Space" || !resultConcealed) return;
        // Keep Space from scrolling or clicking whatever has focus
        event.preventDefault();
        swallowSpaceKeyup = true;
        if (!event.repeat) revealResult();
    });

    document.addEventListener("keyup", (event) => {
        if (event.code === "Space" && swallowSpaceKeyup) {
            event.preventDefault();
            swallowSpaceKeyup = false;
        }
    });

    /* =========================================================
       PLAY BUTTON
    ========================================================= */
    const playBtn = document.getElementById("playBtn");
    const replayBtn = document.getElementById("replayBtn");
    const listenAgainBtn = document.getElementById("listenAgainBtn");
    const PLAY_LABEL = '<i class="fa-solid fa-play me-2"></i> Play Sound &amp; Test';
    const REPLAY_LABEL = '<i class="fa-solid fa-rotate-right me-2"></i> Replay Sound';
    let testRunning = false;
    let replayRunning = false;
    let soundPlayed = false;

    function updateButtons() {
        if (playBtn) {
            playBtn.disabled = testRunning || replayRunning;
            if (!testRunning) playBtn.innerHTML = PLAY_LABEL;
        }
        if (replayBtn) {
            replayBtn.disabled = testRunning || replayRunning || !soundPlayed;
            replayBtn.innerHTML = replayRunning
                ? '<i class="fa-solid fa-volume-high me-2"></i> Playing...'
                : REPLAY_LABEL;
        }
        if (listenAgainBtn) listenAgainBtn.disabled = testRunning || replayRunning;
    }

    function finishTest() {
        testRunning = false;
        updateButtons();
    }

    if (playBtn) {
        playBtn.addEventListener("click", () => {
            realWaveformData = [];
            realSpectrogramData = [];
            liveElapsed = 0;
            paintWave();
            paintSpectrogram();
            setVisualizerState(true);
            setConcealed(false);
            showAnalysisAnimation();
            // Space is the reveal key; don't let it re-trigger this button
            playBtn.blur();

            testRunning = true;
            updateButtons();
            playBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-2"></i> Recording &amp; Analysing...';

            fetch("/trigger/play")
                .then(response => response.json().then(data => {
                    if (!response.ok) throw new Error(data.error || response.statusText);
                    console.log("Horse sound test triggered:", data);
                }))
                .catch(error => {
                    console.error("Trigger error:", error);
                    resetPredictionResults();
                    setVisualizerState(false);
                    finishTest();
                });
        });
    }

    /* =========================================================
       REPLAY: play the last sound again through the speakers
    ========================================================= */
    function replaySound(event) {
        if (event && event.currentTarget) event.currentTarget.blur();
        if (replayRunning || testRunning) return;
        replayRunning = true;
        updateButtons();

        // Returns once the sound has finished playing
        fetch("/trigger/replay")
            .then(response => response.json().then(data => {
                if (!response.ok) throw new Error(data.error || response.statusText);
            }))
            .catch(error => console.error("Replay error:", error))
            .finally(() => {
                replayRunning = false;
                updateButtons();
            });
    }

    if (replayBtn) replayBtn.addEventListener("click", replaySound);
    if (listenAgainBtn) listenAgainBtn.addEventListener("click", replaySound);

    /* =========================================================
       STATUS POLLING
    ========================================================= */
let lastActionStatus = "";    
setInterval(() => {
        fetch("/status")
            .then(response => response.json())
            .then(data => {
                const actionStatus = document.getElementById("actionStatus");
                const predictionText = document.getElementById("predictionText");
                const confidenceText = document.getElementById("confidenceText");

                if (actionStatus) actionStatus.innerText = data.action;
                if (predictionText) predictionText.innerText = data.prediction;
                if (confidenceText) confidenceText.innerText = data.confidence;

                if (data.action === lastActionStatus) {
                      return;
                }
                lastActionStatus = data.action;

                if (data.action === "PLAYING & RECORDING") {
                    soundPlayed = true;
                    setVisualizerState(true);
                } else if (data.action === "COMPLETED") {
                    stopAudioDataPolling();
                    concealResult();
                    soundPlayed = true;
                    finishTest();
                } else if (data.action === "ERROR") {
                    // Show the error instead of waiting forever
                    stopAudioDataPolling();
                    resetPredictionResults();
                    finishTest();
                } else if (data.action === "System Ready") {
                    resetPredictionResults();
                    finishTest();
                }

                if (data.probabilities) {
                    const probabilities = data.probabilities;
                    const classes = ["drinking", "eating", "neigh", "horse_kick", "rattle", "snort", "nicker"];
                    classes.forEach(cls => {
                        const value = Number(probabilities[cls] || 0);
                        const rounded = value.toFixed(1);
                        const valueElement = document.getElementById("value-" + cls);
                        const barElement = document.getElementById("bar-" + cls);
                        if (valueElement) valueElement.innerText = rounded + "%";
                        if (barElement) barElement.style.width = Math.min(100, value) + "%";
                    });
                }
            })
            .catch(error => console.log("Status connection error:", error));
    }, 250);
