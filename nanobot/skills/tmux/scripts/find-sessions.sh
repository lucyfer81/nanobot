#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: find-sessions.sh [-L socket-name|-S socket-path|-A|-U] [-q pattern]

List tmux sessions on a socket (default tmux socket if none provided).

Options:
  -L, --socket       tmux socket name (passed to tmux -L)
  -S, --socket-path  tmux socket path (passed to tmux -S)
  -A, --all          scan socket files under NANOBOT_TMUX_SOCKET_DIR
                     (default: /tmp/tmux-<uid>)
  -U, --all-users    scan /tmp/tmux-* for all users (best-effort)
  -q, --query        case-insensitive substring to filter session names
  -h, --help         show this help
USAGE
}

socket_name=""
socket_path=""
query=""
scan_all=false
scan_all_users=false
default_socket_dir="/tmp/tmux-$(id -u)"
socket_dir="${NANOBOT_TMUX_SOCKET_DIR:-$default_socket_dir}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -L|--socket)      socket_name="${2-}"; shift 2 ;;
    -S|--socket-path) socket_path="${2-}"; shift 2 ;;
    -A|--all)         scan_all=true; shift ;;
    -U|--all-users)   scan_all_users=true; shift ;;
    -q|--query)       query="${2-}"; shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$scan_all" == true && "$scan_all_users" == true ]]; then
  echo "Cannot combine --all with --all-users" >&2
  exit 1
fi

if [[ ( "$scan_all" == true || "$scan_all_users" == true ) && ( -n "$socket_name" || -n "$socket_path" ) ]]; then
  echo "Cannot combine --all/--all-users with -L or -S" >&2
  exit 1
fi

if [[ -n "$socket_name" && -n "$socket_path" ]]; then
  echo "Use either -L or -S, not both" >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found in PATH" >&2
  exit 1
fi

list_sessions() {
  local label="$1"; shift
  local tmux_cmd=(tmux "$@")
  local format=$'#{session_name}\t#{session_attached}\t#{session_created_string}'

  if ! sessions="$("${tmux_cmd[@]}" list-sessions -F "$format" 2>/dev/null)"; then
    echo "No tmux server found on $label" >&2
    return 1
  fi

  if [[ -n "$query" ]]; then
    sessions="$(printf '%s\n' "$sessions" | grep -i -- "$query" || true)"
  fi

  if [[ -z "$sessions" ]]; then
    echo "No sessions found on $label"
    return 0
  fi

  echo "Sessions on $label:"
  printf '%s\n' "$sessions" | while IFS=$'\t' read -r name attached created; do
    attached_label=$([[ "$attached" == "1" ]] && echo "attached" || echo "detached")
    printf '  - %s (%s, started %s)\n' "$name" "$attached_label" "$created"
  done
}

scan_socket_dir() {
  local dir="$1"
  local label_prefix="$2"
  local sockets=()

  if [[ ! -d "$dir" ]]; then
    echo "Socket directory not found: $dir" >&2
    return 1
  fi

  while IFS= read -r sock; do
    sockets+=("$sock")
  done < <(find "$dir" -mindepth 1 -maxdepth 1 -type s 2>/dev/null | sort)

  if [[ "${#sockets[@]}" -eq 0 ]]; then
    echo "No socket files found under $dir" >&2
    return 1
  fi

  exit_code=0
  for sock in "${sockets[@]}"; do
    list_sessions "$label_prefix '$sock'" -S "$sock" || exit_code=$?
  done
  return "$exit_code"
}

if [[ "$scan_all" == true ]]; then
  scan_socket_dir "$socket_dir" "socket path" || exit $?
  exit 0
fi

if [[ "$scan_all_users" == true ]]; then
  exit_code=0
  found_user_dirs=false
  while IFS= read -r dir; do
    found_user_dirs=true
    owner="$(stat -c '%U' "$dir" 2>/dev/null || echo unknown)"
    if ! scan_socket_dir "$dir" "user '$owner' socket path"; then
      exit_code=$?
    fi
  done < <(find /tmp -mindepth 1 -maxdepth 1 -type d -name 'tmux-*' 2>/dev/null | sort)

  if [[ "$found_user_dirs" == false ]]; then
    echo "No tmux user directories found under /tmp" >&2
    exit 1
  fi
  exit "$exit_code"
fi

tmux_cmd=(tmux)
socket_label="default socket"

if [[ -n "$socket_name" ]]; then
  tmux_cmd+=(-L "$socket_name")
  socket_label="socket name '$socket_name'"
elif [[ -n "$socket_path" ]]; then
  tmux_cmd+=(-S "$socket_path")
  socket_label="socket path '$socket_path'"
fi

list_sessions "$socket_label" "${tmux_cmd[@]:1}"
