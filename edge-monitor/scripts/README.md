# # Get Chamption Script Setup

Similar to Push Alert Script below.

- First

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/weights_key -C "weights-pull-<stable>"
cat ~/.ssh/weights_key.pub 
```

Then on the server add this to the `~/.ssh/authorized_keys` file:
```bash 
command="rrsync -ro /data/model_cache",restrict PASTE_THE_KEY_HERE
```

# # Push Alert Script — Setup

Pushes a single alert clip from a Raspberry Pi to the DGX under
`/data/audio/alerts/<stable>_<stall>/`.

**Server endpoint:** `root@20.91.249.84` port `50024` (routes into the `horse-data` container).

---

## 1. Generate a key on the NEW Pi

``bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/alert_key -C "alert-push-<stable>"
cat ~/.ssh/alert_key.pub      # copy this whole single line
```

`-N ""` = no passphrase (required for unattended pushes from the webserver).

---

## 2. Authorize that key on the server

ssh into dgx and navigate to /MNT/horseData/horse-data/root (or the container's home directory) and edit `~/.ssh/authorized_keys`:


```bash
sudo nano /MNT/horseData/horse-data/root/.ssh/authorized_keys

cat ~/.ssh/authorized_keys    # confirm BOTH keys are present, original unchanged
```



---

## 3. Test the connection from the NEW Pi

```bash
ssh -i ~/.ssh/alert_key -p 50024 -o StrictHostKeyChecking=accept-new \
  root@20.91.249.84 'echo ok'
```

Should print `ok` with **no password prompt**.


## Notes

- **Key path / ownership:** the script reads `~/.ssh/alert_key`. It must be
  `chmod 600` and owned by the user the Rust server runs as. Adjust the `KEY=`
  line in `push_alert.sh` if the key lives elsewhere.
- **Filename safety:** `stable`/`stall` are validated against `^[A-Za-z0-9_-]+$`
  to block path escapes (`..`, `/`). Also validate the incoming filename in Rust
  so the endpoint can't push arbitrary files off the Pi.

