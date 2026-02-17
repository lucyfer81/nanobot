#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: send-command.sh -t target -c command [-L socket-name|-S socket-path] [options]

Send one command to a tmux pane in a safer two-step way:
1) send literal text
2) send Enter after a short delay

Options:
  -t, --target       tmux target (session:window.pane), required
  -c, --command      command text to type, required
  -L, --socket       tmux socket name (passed to tmux -L)
  -S, --socket-path  tmux socket path (passed to tmux -S)
  -d, --delay        seconds between text and Enter (default: 0.12)
  -r, --retries      typing retries before giving up (default: 2)
  -n, --no-enter     do not send Enter
      --no-echo-check  skip checking whether typed text appears in pane
      --enter-key    key token for submit (default: Enter, e.g. C-m)
  -h, --help         show this help
USAGE
}

target=""
command_text=""
socket_name=""
socket_path=""
delay="0.12"
retries=2
send_enter=true
echo_check=true
enter_key="Enter"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--target) target="${2-}"; shift 2 ;;
    -c|--command) command_text="${2-}"; shift 2 ;;
    -L|--socket) socket_name="${2-}"; shift 2 ;;
    -S|--socket-path) socket_path="${2-}"; shift 2 ;;
    -d|--delay) delay="${2-}"; shift 2 ;;
    -r|--retries) retries="${2-}"; shift 2 ;;
    -n|--no-enter) send_enter=false; shift ;;
    --no-echo-check) echo_check=false; shift ;;
    --enter-key) enter_key="${2-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$target" || -z "$command_text" ]]; then
  echo "target and command are required" >&2
  usage
  exit 1
fi

if [[ -n "$socket_name" && -n "$socket_path" ]]; then
  echo "Use either -L or -S, not both" >&2
  exit 1
fi

if ! [[ "$retries" =~ ^[1-9][0-9]*$ ]]; then
  echo "retries must be a positive integer" >&2
  exit 1
fi

if ! [[ "$delay" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "delay must be numeric (seconds)" >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found in PATH" >&2
  exit 1
fi

tmux_cmd=(tmux)
if [[ -n "$socket_name" ]]; then
  tmux_cmd+=(-L "$socket_name")
elif [[ -n "$socket_path" ]]; then
  tmux_cmd+=(-S "$socket_path")
fi

if ! "${tmux_cmd[@]}" display-message -p -t "$target" '#{pane_id}' >/dev/null 2>&1; then
  echo "tmux target not found: $target" >&2
  exit 1
fi

pane_has_text() {
  local needle="$1"
  local pane_text
  pane_text="$("${tmux_cmd[@]}" capture-pane -p -J -t "$target" -S -120 2>/dev/null || true)"
  printf '%s\n' "$pane_text" | grep -F -- "$needle" >/dev/null 2>&1
}

type_with_send_keys() {
  "${tmux_cmd[@]}" send-keys -t "$target" -l -- "$command_text"
}

type_with_paste_buffer() {
  "${tmux_cmd[@]}" set-buffer -- "$command_text"
  "${tmux_cmd[@]}" paste-buffer -d -t "$target"
}

typed=false
attempt=1
while (( attempt <= retries )); do
  if (( attempt == 1 )); then
    type_with_send_keys
  else
    type_with_paste_buffer
  fi

  sleep "$delay"

  if [[ "$echo_check" == false ]] || pane_has_text "$command_text"; then
    typed=true
    break
  fi

  attempt=$((attempt + 1))
done

if [[ "$typed" == false ]]; then
  echo "Warning: command text was not observed in pane after ${retries} attempt(s)." >&2
  echo "Hint: try --no-echo-check for hidden-input prompts." >&2
fi

if [[ "$send_enter" == true ]]; then
  "${tmux_cmd[@]}" send-keys -t "$target" "$enter_key"
fi

echo "Sent command to $target (enter=$send_enter, method=$([[ "$typed" == true ]] && echo ok || echo uncertain))"
