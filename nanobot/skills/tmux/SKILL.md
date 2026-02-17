---
name: tmux
description: Remote-control tmux sessions for interactive CLIs by sending keystrokes and scraping pane output.
metadata: {"nanobot":{"emoji":"🧵","os":["darwin","linux"],"requires":{"bins":["tmux"]}}}
---

# tmux Skill

Use tmux only when you need an interactive TTY. Prefer exec background mode for long-running, non-interactive tasks.

## Quickstart (isolated socket, exec tool)

```bash
SOCKET_DIR="${NANOBOT_TMUX_SOCKET_DIR:-${TMPDIR:-/tmp}/nanobot-tmux-sockets}"
mkdir -p "$SOCKET_DIR"
SOCKET="$SOCKET_DIR/nanobot.sock"
SESSION=nanobot-python

tmux -S "$SOCKET" new -d -s "$SESSION" -n shell
tmux -S "$SOCKET" send-keys -t "$SESSION":0.0 -- 'PYTHON_BASIC_REPL=1 python3 -q' Enter
tmux -S "$SOCKET" capture-pane -p -J -t "$SESSION":0.0 -S -200
```

After starting a session, always print monitor commands:

```
To monitor:
  tmux -S "$SOCKET" attach -t "$SESSION"
  tmux -S "$SOCKET" capture-pane -p -J -t "$SESSION":0.0 -S -200
```

## Socket convention

- For the current user's default tmux server, use: `SOCKET="/tmp/tmux-$(id -u)/default"`.
- `NANOBOT_TMUX_SOCKET_DIR` is used by `find-sessions.sh --all`; default is `/tmp/tmux-$(id -u)`.
- When running nanobot via systemd, set `PrivateTmp=false` or the service may see a different `/tmp`.

## Targeting panes and naming

- Target format: `session:window.pane` (defaults to `:0.0`).
- Keep names short; avoid spaces.
- Inspect: `tmux -S "$SOCKET" list-sessions`, `tmux -S "$SOCKET" list-panes -a`.

## Finding sessions

- List sessions on one socket: `nanobot/skills/tmux/scripts/find-sessions.sh -S "$SOCKET"`.
- Scan current user's tmux socket directory: `nanobot/skills/tmux/scripts/find-sessions.sh --all`.
- Optional diagnostic: scan all users (`root` usually required): `nanobot/skills/tmux/scripts/find-sessions.sh --all-users`.

## Sending input safely

- Prefer literal sends: `tmux -S "$SOCKET" send-keys -t target -l -- "$cmd"`.
- Control keys: `tmux -S "$SOCKET" send-keys -t target C-c`.
- For interactive TUI apps (Claude Code/Codex), do not send text+Enter in one rapid command.
  Send text first, then send `Enter` separately after a short delay:

```bash
tmux -S "$SOCKET" send-keys -t target -l -- "$cmd" && sleep 0.1 && tmux -S "$SOCKET" send-keys -t target Enter
```

- For interactive prompts, always use this wrapper for stability:

```bash
nanobot/skills/tmux/scripts/send-command.sh -S "$SOCKET" -t session:0.0 -c "$cmd"
```

## Watching output

- Capture recent history: `tmux -S "$SOCKET" capture-pane -p -J -t target -S -200`.
- Wait for prompts: `nanobot/skills/tmux/scripts/wait-for-text.sh -S "$SOCKET" -t session:0.0 -p 'pattern'`.
- Attaching is OK; detach with `Ctrl+b d`.

## Spawning processes

- For python REPLs, set `PYTHON_BASIC_REPL=1` (non-basic REPL breaks send-keys flows).

## Windows / WSL

- tmux is supported on macOS/Linux. On Windows, use WSL and install tmux inside WSL.
- This skill is gated to `darwin`/`linux` and requires `tmux` on PATH.

## Orchestrating Coding Agents (Codex, Claude Code)

tmux excels at running multiple coding agents in parallel:

```bash
SOCKET="${TMPDIR:-/tmp}/codex-army.sock"

# Create multiple sessions
for i in 1 2 3 4 5; do
  tmux -S "$SOCKET" new-session -d -s "agent-$i"
done

# Launch agents in different workdirs (safer submit wrapper)
nanobot/skills/tmux/scripts/send-command.sh -S "$SOCKET" -t agent-1:0.0 -c "cd /tmp/project1 && codex --yolo 'Fix bug X'"
nanobot/skills/tmux/scripts/send-command.sh -S "$SOCKET" -t agent-2:0.0 -c "cd /tmp/project2 && codex --yolo 'Fix bug Y'"

# Poll for completion (check if prompt returned)
for sess in agent-1 agent-2; do
  if tmux -S "$SOCKET" capture-pane -p -t "$sess" -S -3 | grep -q "❯"; then
    echo "$sess: DONE"
  else
    echo "$sess: Running..."
  fi
done

# Get full output from completed session
tmux -S "$SOCKET" capture-pane -p -t agent-1 -S -500
```

**Tips:**
- Use separate git worktrees for parallel fixes (no branch conflicts)
- `pnpm install` first before running codex in fresh clones
- Check for shell prompt (`❯` or `$`) to detect completion
- Codex needs `--yolo` or `--full-auto` for non-interactive fixes

## Cleanup

- Kill a session: `tmux -S "$SOCKET" kill-session -t "$SESSION"`.
- Kill all sessions on a socket: `tmux -S "$SOCKET" list-sessions -F '#{session_name}' | xargs -r -n1 tmux -S "$SOCKET" kill-session -t`.
- Remove everything on the private socket: `tmux -S "$SOCKET" kill-server`.

## Helper: wait-for-text.sh

`nanobot/skills/tmux/scripts/wait-for-text.sh` polls a pane for a regex (or fixed string) with a timeout.

```bash
nanobot/skills/tmux/scripts/wait-for-text.sh -S "$SOCKET" -t session:0.0 -p 'pattern' [-F] [-T 20] [-i 0.5] [-l 2000]
```

- `-t`/`--target` pane target (required)
- `-p`/`--pattern` regex to match (required); add `-F` for fixed string
- `-L`/`--socket` tmux socket name or `-S`/`--socket-path` socket path
- `-T` timeout seconds (integer, default 15)
- `-i` poll interval seconds (default 0.5)
- `-l` history lines to search (integer, default 1000)

## Helper: send-command.sh

`nanobot/skills/tmux/scripts/send-command.sh` sends one command to a pane with safer submit semantics (type first, Enter second, optional retries).

```bash
nanobot/skills/tmux/scripts/send-command.sh -S "$SOCKET" -t session:0.0 -c 'echo hello'
```

- `-t`/`--target` pane target (required)
- `-c`/`--command` command text (required)
- `-L`/`--socket` or `-S`/`--socket-path` to select socket
- `-d` delay seconds between typing and Enter (default 0.12)
- `-r` typing retries (default 2; retry 2+ uses paste-buffer fallback)
- `-n` skip Enter
- `--no-echo-check` skip text-echo verification (useful for hidden-input prompts)
- `--enter-key` submit key token (default `Enter`, alternative `C-m`)
