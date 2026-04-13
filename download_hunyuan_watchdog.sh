#!/bin/bash
# Watchdog for HunyuanVideo download.
# Kills and restarts download if no disk progress for 3 minutes.
# Run on VM host: nohup bash /tmp/download_hunyuan_watchdog.sh > /tmp/hunyuan_watchdog.log 2>&1 &

CONTAINER="flux-server-flux-server-1"
CACHE_PATH="/app/model_cache/models--hunyuanvideo-community--HunyuanVideo"
SCRIPT="/tmp/download_hunyuan_v2.py"
STALL_LIMIT=6   # 6 x 30s = 3 min stall before restart
DL_PID=""

start_download() {
    echo "[$(date '+%H:%M:%S')] Starting download..."
    sudo docker exec -d "$CONTAINER" python3 "$SCRIPT"
    DL_PID=$(sudo docker top "$CONTAINER" | awk '/download_hunyuan_v2/{print $2}' | head -1)
    echo "[$(date '+%H:%M:%S')] Download PID (inside container): $DL_PID"
}

get_cache_bytes() {
    sudo docker exec "$CONTAINER" du -sb "$CACHE_PATH" 2>/dev/null | cut -f1 || echo 0
}

is_complete() {
    local n
    n=$(sudo docker exec "$CONTAINER" find "$CACHE_PATH" -name '*.incomplete' 2>/dev/null | wc -l)
    [ "$n" -eq 0 ]
}

# Kill any existing download scripts inside the container
sudo docker exec "$CONTAINER" pkill -f download_hunyuan 2>/dev/null || true
sleep 2

start_download

PREV_SIZE=0
STALL_COUNT=0
RUN=0

while true; do
    sleep 30
    RUN=$((RUN+1))

    if is_complete; then
        echo "[$(date '+%H:%M:%S')] DONE — all files downloaded, no .incomplete blobs."
        break
    fi

    CURR_SIZE=$(get_cache_bytes)
    INCOMPLETE=$(sudo docker exec "$CONTAINER" find "$CACHE_PATH" -name '*.incomplete' 2>/dev/null | wc -l)

    if [ "$CURR_SIZE" -gt "$PREV_SIZE" ]; then
        DELTA=$((CURR_SIZE - PREV_SIZE))
        echo "[$(date '+%H:%M:%S')] Progress: ${CURR_SIZE} bytes (+${DELTA}B in 30s) | incomplete=${INCOMPLETE}"
        STALL_COUNT=0
    else
        STALL_COUNT=$((STALL_COUNT+1))
        echo "[$(date '+%H:%M:%S')] No progress (stall #${STALL_COUNT}/${STALL_LIMIT}) | size=${CURR_SIZE} | incomplete=${INCOMPLETE}"
        if [ $STALL_COUNT -ge $STALL_LIMIT ]; then
            echo "[$(date '+%H:%M:%S')] STALL DETECTED — killing and restarting download"
            sudo docker exec "$CONTAINER" pkill -f download_hunyuan 2>/dev/null || true
            sleep 3
            start_download
            STALL_COUNT=0
        fi
    fi

    PREV_SIZE=$CURR_SIZE
done
