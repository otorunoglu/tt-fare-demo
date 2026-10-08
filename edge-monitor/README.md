# stable-edge-monitor

High-performance, low-latency audio monitoring for horse stables, written in Rust.

## Build & Run

### Prerequisites
-   Rust toolchain
-   Models in `data/models/`

### Compiling from source
Raspberry Pi must be at least on Debian13 and also take care its on 64 bit!
- (OPTIONAL) Upgrade system fully to newer Debian version:
```bash
# OPTIONAL! THINK BEFORE COPY PASTING IT!
sudo sed -i 's/bookworm/trixie/g' /etc/apt/sources.list
sudo apt update
sudo apt full-upgrade
sudo reboot
```

- Install Rust and Cargo: 
```bash
curl https://sh.rustup.rs -sSf | sh
# Dont forget to source bash after install or open new terminal
```
- Then install dependencies:
```bash
sudo apt update
sudo apt-get install libssl-dev libasound2-dev
```

- Then check if all goes well
```bash
cargo check
```

- Dont forget to create env file with a path to the model weigths:
```bash
openssl req -x509 -newkey rsa:2048 -keyout ./cert/key.pem -out ./cert/cert.pem -days 365 -nodes -subj "/CN=localhost"
```

### Build
```bash
cargo build --release
```

### Run Realtime (Microphone)
```bash
cargo run --release -- \
  --model-path data/models/full_pipeline.onnx \
  --label-mapping-path data/models/label_mapping.json \
  --config-path smart_stable_model_config.yaml \
  --stable-id stable01 \
  --stall-id stall01
```

### Run Simulation (File)
```bash
cargo run --release -- \
  --simulate-file data/audio/stable03_stall01_horse01_20251214_000008_mic.flac \
  --config-path smart_stable_model_config.yaml
```

### Configuration
Environment variables can also be used:
```bash
export STABLE_ID=stable02
export STALL_ID=stall05
cargo run --release
```

## Testing

To test against the databases and testfiles defined in `config/edge_monitor_config.toml`, run:
```bash
cargo run -p inference -- verify-all
```

## Python Bridge (`audio_preprocessing` crate)

> **⚠️ Do not remove `../../libs/audio_preprocessing/`!** It is shared between this repo and the Python ML backend.

The `audio_preprocessing` crate provides PyO3 bindings so the Python backend (`label-studio-setup`) uses the **exact same Rust code** for audio loading and resampling as the Raspberry Pi edge monitor. This guarantees that training and inference see audio identically.

- **Rust side** (edge monitor): uses the crate as a normal Rust dependency
- **Python side** (ML backend): built into a `.whl` via `maturin` during `docker compose build`

### Updating the bridge

Changes to `../../libs/audio_preprocessing/` require a **Docker image rebuild** — a simple `docker compose restart` is not sufficient because the wheel is compiled at build time:

```bash
cd label-studio-setup
docker compose build smart-stable-ml-backend
docker compose up -d smart-stable-ml-backend
```

## Webserver Config:

### How to run:
Build rust webserver:
```bash
cargo build --release -p webserver
```
#### run daemon:
```bash
sudo systemctl enable stable-edge-monitor-webserver.service
```
#### Show logs:
```bash
journalctl -u stable-edge-monitor-webserver.service -f
```
#### Stop daemon:
```bash
sudo systemctl stop stable-edge-monitor-webserver.service
```
#### Changes to the daemon:
```bash
sudo nano /etc/systemd/system/stable-edge-monitor-webserver.service
```

### Privilege model for remote updates

The webserver itself does not need to run as root in the normal case:
- it binds to port `3000`, not a privileged port
- audio capture should work as an unprivileged service user as long as that user has access to the ALSA devices, typically via the `audio` group

For remote updates, keep the build unprivileged and delegate only the service control step:

1. Copy `scripts/update_edge_monitor.sh` to the deployed checkout and make it executable.
2. Copy `config/stable-edge-monitor-update.service.example` to `/etc/systemd/system/stable-edge-monitor-update.service` and adjust the `User`, `Group`, and `WorkingDirectory` values.
3. Copy `config/stable-edge-monitor-update.sudoers.example` to `/etc/sudoers.d/stable-edge-monitor-update` via `visudo -f`.
4. Reload systemd with `sudo systemctl daemon-reload`.

The intended runtime flow is:

1. The HTTP endpoint runs as the `edge-monitor` service user.
2. It triggers `sudo /bin/systemctl start stable-edge-monitor-update.service`.
3. The updater service runs `git fetch`, hard-resets to `origin/dev`, builds `webserver`, and restarts `stable-edge-monitor-webserver.service`.

Once that wiring is in place, the webserver endpoint `POST /api/admin/update` will trigger the updater unit and return immediately with `202 Accepted`.

Notes:
- No sudo password needs to be stored anywhere.
- The deployment checkout should be dedicated to the edge service because the updater does a hard reset.
- `git fetch` plus `git reset --hard origin/dev` is used instead of `git pull` to avoid merge state during unattended deploys.


# Steps to setup new raspberry pi (Tested):
Raspberry Pi must be at least on Debian13 `cat /etc/os-release` and also take care its on 64 bit!
- (OPTIONAL) Upgrade system fully to newer Debian version:
```bash
# OPTIONAL! THINK BEFORE COPY PASTING IT!
sudo sed -i 's/bookworm/trixie/g' /etc/apt/sources.list
sudo apt update
sudo apt full-upgrade
sudo reboot
```

- Create a new user in usergroup audio and add to system-journal group: 
```bash
sudo useradd -m -G audio edge-monitor
sudo usermod -aG systemd-journal edge-monitor
sudo passwd edge-monitor
su edge-monitor
```

- Setup new ssh key and clone rep:
```bash
ssh-keygen -t ed25519 -C "edge-monitor@pi" -f ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
git clone git@github.com:Geisler-at-Savonia/equine-ai-monitor.git
git switch dev
```

- install cargo and dependencies:
```bash
curl https://sh.rustup.rs -sSf | sh
sudo apt update 
sudo apt-get install libssl-dev libasound2-dev
cd ~/equine-ai-monitor/apps/edge-monitor && cargo build --release
```
- Dont forget to create env file (e.g. copy example one):
```bash
cp /home/edge-monitor/equine-ai-monitor/apps/edge-monitor/.env.example /home/edge-monitor/equine-ai-monitor/apps/edge-monitor/.env
```
- Generate and add cert files and folder:
```bash
mkdir -p /home/edge-monitor/equine-ai-monitor/apps/edge-monitor/cert && openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout /home/edge-monitor/equine-ai-monitor/apps/edge-monitor/cert/key.pem -out cert/cert.pem -days 365 \
  -subj "/CN=$(hostname).local"
```

- Add webserver service
```bash
sudo cp /home/edge-monitor/equine-ai-monitor/apps/edge-monitor/config/stable-edge-monitor-webserver.service.example /etc/systemd/system/stable-edge-monitor-webserver.service
sudo cp /home/edge-monitor/equine-ai-monitor/apps/edge-monitor/config/stable-edge-monitor-update.service.example /etc/systemd/system/stable-edge-monitor-update.service
sudo cp /home/edge-monitor/equine-ai-monitor/apps/edge-monitor/config/stable-edge-monitor-update.sudoers.example /etc/sudoers.d/stable-edge-monitor-update

sudo systemctl daemon-reload
sudo systemctl start stable-edge-monitor-webserver.service
```
- Download model files

- general ownership command:
```bash
sudo chown -R edge-monitor:edge-monitor /home/edge-monitor/equine-ai-monitor
sudo chmod -R u+rwX,go-rwx /home/edge-monitor/equine-ai-monitor
```


### Copy model files using scp:
`scp -r ~/Downloads/champion_v31_onnx/* edge-monitor@pi135-ssh.horsemonitor.win:~/equine-ai-monitor/apps/edge-monitor/data/models`

### Add gdrive support:
- install rclone:
```bash
curl https://rclone.org/install.sh | sudo bash
```
- Copy rclone config file to raspy:
`scp dgx:~/.config/rclone/rclone.conf edge-monitor@pi135-ssh.horsemonitor.win:~/.config/rclone/rclone.conf`

- Original rclone config (just use config file instead!)
```bash
rclone config        # pick "drive" (or "s3" → Cloudflare R2)
n # for new remote
name> model_weights
storgae> drive # Or find number of Google Drive in the list
client_id> # Get from goolgle console secret thing 
client_secret> # Get from goolgle console secret thing 
scope> 1 # Full access
service__account_file> # press enter
Edit advanced config? n
use auto config? n
Use web browser to automatically authenticate rclone with remote? n

Now copy paste rclone command to device with webbrwoser, something like
rclone authorize "drive" "SOMEID"
// paste token
config as shared drive? n
keep this remote? y
q # Quit rclone
```

### Debugging:
- if some error with tokio shutting down, check if you have created the certfiles!
