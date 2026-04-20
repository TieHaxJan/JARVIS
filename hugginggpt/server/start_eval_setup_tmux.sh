#!/usr/bin/env bash
set -euo pipefail

SESSION="jarvis"
WORKDIR="/workspace/JARVIS/hugginggpt/server"
CONFIG="configs/config.localllama.yaml"

# Avoid duplicate session names
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists."
  echo "Attach with: tmux attach -t $SESSION"
  exit 1
fi

# Create new detached session
tmux new-session -d -s "$SESSION" -n main

# Split into 3 panes
tmux split-window -h -t "$SESSION":0
tmux split-window -v -t "$SESSION":0.1

tmux select-layout -t "$SESSION":0 tiled

# Pane 0: models server
tmux send-keys -t "$SESSION":0.0 "cd $WORKDIR" C-m
tmux send-keys -t "$SESSION":0.0 "source env_models/bin/activate" C-m
tmux send-keys -t "$SESSION":0.0 "python models_server.py --config $CONFIG" C-m

# Pane 1: awesome_chat server, wait 30s first
tmux send-keys -t "$SESSION":0.1 "cd $WORKDIR" C-m
tmux send-keys -t "$SESSION":0.1 "source env_chat/bin/activate" C-m
tmux send-keys -t "$SESSION":0.1 "sleep 30 && python awesome_chat.py --config $CONFIG --mode server" C-m

# Pane 2: run batch
tmux send-keys -t "$SESSION":0.2 "cd $WORKDIR" C-m
tmux send-keys -t "$SESSION":0.2 "source env_chat/bin/activate" C-m
tmux send-keys -t "$SESSION":0.2 "sleep 45 && python run_batch.py" C-m

# Attach to session
tmux attach -t "$SESSION"