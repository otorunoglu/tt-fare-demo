# Label Studio Setup

## Installation

To install this you need to make sure to install the requiremets, which should be situated in folders parallel to label-studio-setup:
- uv pip install -e ../smartstablemodel
- uv pip install -e ../ml_audio_core

## Label Studio Configuration

### Project Setup
- Create a new Project and paste html code from LABELING_INTERFACE.md into custom template under Labeling Setup.
- Setup the Cloud Storage in project settings:
   - Under Cloud Storage, select Add Source Storage
      - Storage Type: Local files
      - Title: audio
      - Absolute local path: /label-studio/data/audio
      - Set "Treat every bucket object as a source file" to true
      - Click Add and then *DO NOT!* click Sync Storage 
         - Sync Storage would create a task for every audio file, but we want a task with audio and video together
   - Repeat same process for video
   - Now add task files, they look like this:
```
[
  {
    "audio": "/data/local-files/?d=label-studio/data/audio/horse123.wav",
    "video": "/data/local-files/?d=label-studio/data/video/horse123_stereo.mp4",
    "stable": "Farm 42",
    "stall": 12
  }
]
```

### ML Backend Setup
1. Open Label Studio in your browser
2. Go to **Settings → Model**
3. Click **Connect Model**
4. Enter Name: `THE NAME OF THE ML MODEL` or `soundscape_detector`
5. Enter URL: `http://ml-backend:9090` or whatever backend you use
6. Click **Validate and Save**
7. Add personal token to .env file as LABEL_STUDIO_PERSONAL_TOKEN=your_token or just put it in the call and get API token:
```bash
docker exec ml-backend curl -X POST label-studio:8080/api/token/refresh \
-H "Content-Type: application/json" \
-d '{"refresh": $LABEL_STUDIO_PERSONAL_TOKEN}'
```
8. Add API token in .env file: LABEL_STUDIO_API_TOKEN=your_token
9. Test API token:
```bash
docker exec ml-backend curl -v \
  -H "Authorization: Token $LABEL_STUDIO_API_TOKEN" \
  http://label-studio:8080/api/projects/
```

## Training Workflow
Pretrain the model on some prelabeled data (or the artificial data)
1. **Label the data** currently (26.06.2025) with three categories:
   - **Kick**: Actual kick sounds
   - **Normal**: Explicitly normal barn sounds (this includes everything not in the other categories)
   - **Abnormal**: Leave this for the CAE to predict (don't manually label)

2. **Manual Training**: Go to the project → **Model** tab, right click → **Start Training**
   - This trains on ALL labeled data in the project
   - Recommended approach for stable, consistent training
   - Do not mix stabel training data

### 3. Network Configuration Notes
- **ML Backend URL**: Use `http://ml-backend:9090` (Docker service name)
- **No webhooks needed**: Manual training is more reliable

## API Endpoints

- **`/predict`** - Get predictions for audio files
- **`/webhook`** - Webhook handler for Label Studio training events (uses SDK)
- **`/train`** - Direct training endpoint (Label studio is supposed to call this, but through a bug it calls /webhook instead, this endpoint will cry out if label studio should fix this in the future!)
- **`/health`** - Health check
- **`/setup`** - Setup information
- **`/training-status`** - Check training status
- **`/model-status`** - Check what models are loaded
- **`/reset-models`** - Reset to original models


## Training Workflows

### Initial Training (Command Line)
First, label some data in Label Studio and export it, then run initial training:

```bash
# Train kick detector from Label Studio export
python smartstablemodel/horse_kick_monitor.py \
    --train-from-ls-export /path/to/your-export.json \
    --audio-base-path /data/local-files \
    --epochs 50 \
    --retrain-cae \
    --verbose
```

### CAE Training (Command Line)
Train CAE only on explicitly labeled "Normal" segments:

```bash
# Train CAE on normal labels only
python smartstablemodel/horse_kick_monitor.py \
    --train-cae-normal-labels /path/to/your-export.json \
    --audio-base-path /data/local-files \
    --epochs 50 \
    --verbose
```

### Ongoing Training (Label Studio UI)
- Continue labeling in Label Studio
- Click **"Start Training"** in the Model tab when you want to update the model
- This will retrain on ALL your labeled data

### Training Status
Monitor training progress and status:
```bash
curl http://localhost:9090/training-status
```

## Training Behavior

- **Initial Training**: Full training on exported Label Studio data (command line)
- **Manual Training**: Full training with all labeled data (Label Studio UI)
- **Model Preservation**: Uses low learning rates to preserve existing knowledge
- **Background Processing**: Training doesn't block predictions

## Troubleshooting

### Model Files Missing
If you see "Detector not initialized" errors:

1. **Check if model files exist**:
   ```bash
   ls smartstablemodel/models/
   ```

2. **Run initial training**:
   ```bash
   python smartstablemodel/horse_kick_monitor.py --train-from-ls-export /path/to/export.json
   ```

### Port Conflicts on Server
When deploying on a shared server, change ports to avoid conflicts:

```bash
# Edit docker-compose.yml to use different ports
# Change 8080:8080 to 8081:8080 for Label Studio
# Change 9090:9090 to 9091:9090 for ML Backend
# Update ML backend URL accordingly: http://ml-backend:9091
```

### Training Not Working
1. **Check training logs**: `docker logs ml-backend`
2. **Verify data format**: Check that annotations have proper labels ("Kick", "Normal")
3. **Check file paths**: Ensure audio files are accessible in the container

## Using the deploy script

Remember to edit your WSL SSH config (~/.ssh/config)::
```bash
Host smartstable
  HostName 10.212.11.40
  User johannesgeisler
  ControlMaster auto
  ControlPath ~/.ssh/smartstable-%r@%h:%p
  ControlPersist 5m
```
