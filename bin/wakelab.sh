#!/usr/bin/env bash
#
# Author: Clive Bostock
# Date: 15-May-2026
# Purpose: Launch the WakeLab desktop application from Poetry or a local .venv.
# Usage: ./bin/wakelab.sh [--list-themes] [--set-theme THEME] [--appearance dark|light] [wakelab-args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
THEMES_DIR="$PROJECT_ROOT/src/orac_wake_lab/themes"
THEME_SELECTION_FILE="$THEMES_DIR/set_theme.txt"
DEFAULT_THEME_NAME="NightTrain"
DEFAULT_APPEARANCE_MODE="Dark"

list_themes() {
  find "$THEMES_DIR" -maxdepth 1 -type f -name '*.json' \
    -printf '%f\n' | sed 's/\.json$//' | sort
}

normalize_theme() {
  local theme_value="${1%.json}"
  local theme_path="$THEMES_DIR/$theme_value.json"
  if [[ ! -f "$theme_path" ]]; then
    echo "Error: theme '$1' was not found in $THEMES_DIR." >&2
    echo "Available themes:" >&2
    list_themes | sed 's/^/  /' >&2
    exit 2
  fi
  printf '%s\n' "$theme_value"
}

normalize_appearance() {
  case "${1,,}" in
    dark)
      printf 'Dark\n'
      ;;
    light)
      printf 'Light\n'
      ;;
    *)
      echo "Error: appearance '$1' must be 'dark' or 'light'." >&2
      exit 2
      ;;
  esac
}

current_theme() {
  local theme_name="$DEFAULT_THEME_NAME"
  if [[ -f "$THEME_SELECTION_FILE" ]]; then
    local raw_value
    raw_value="$(<"$THEME_SELECTION_FILE")"
    raw_value="${raw_value%%:*}"
    raw_value="${raw_value%.json}"
    if [[ -f "$THEMES_DIR/$raw_value.json" ]]; then
      theme_name="$raw_value"
    fi
  fi
  printf '%s\n' "$theme_name"
}

current_appearance() {
  local appearance="$DEFAULT_APPEARANCE_MODE"
  if [[ -f "$THEME_SELECTION_FILE" ]]; then
    local raw_value mode_value
    raw_value="$(<"$THEME_SELECTION_FILE")"
    mode_value="${raw_value#*:}"
    if [[ "$raw_value" == *:* ]]; then
      case "${mode_value,,}" in
        dark)
          appearance="Dark"
          ;;
        light)
          appearance="Light"
          ;;
      esac
    fi
  fi
  printf '%s\n' "$appearance"
}

write_theme_selection() {
  printf '%s:%s\n' "$1" "$2" > "$THEME_SELECTION_FILE"
}

APP_ARGS=()
REQUESTED_THEME=""
REQUESTED_APPEARANCE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list-themes)
      list_themes
      exit 0
      ;;
    --set-theme)
      if [[ $# -lt 2 ]]; then
        echo "Error: --set-theme requires a theme name." >&2
        exit 2
      fi
      REQUESTED_THEME="$(normalize_theme "$2")"
      shift 2
      ;;
    --set-theme=*)
      THEME_VALUE="${1#--set-theme=}"
      if [[ -z "$THEME_VALUE" ]]; then
        echo "Error: --set-theme requires a theme name." >&2
        exit 2
      fi
      REQUESTED_THEME="$(normalize_theme "$THEME_VALUE")"
      shift
      ;;
    --appearance)
      if [[ $# -lt 2 ]]; then
        echo "Error: --appearance requires 'dark' or 'light'." >&2
        exit 2
      fi
      REQUESTED_APPEARANCE="$(normalize_appearance "$2")"
      shift 2
      ;;
    --appearance=*)
      APPEARANCE_VALUE="${1#--appearance=}"
      if [[ -z "$APPEARANCE_VALUE" ]]; then
        echo "Error: --appearance requires 'dark' or 'light'." >&2
        exit 2
      fi
      REQUESTED_APPEARANCE="$(normalize_appearance "$APPEARANCE_VALUE")"
      shift
      ;;
    *)
      APP_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$REQUESTED_THEME" || -n "$REQUESTED_APPEARANCE" ]]; then
  THEME_NAME="${REQUESTED_THEME:-$(current_theme)}"
  APPEARANCE_MODE="${REQUESTED_APPEARANCE:-$(current_appearance)}"
  write_theme_selection "$THEME_NAME" "$APPEARANCE_MODE"
fi

if command -v poetry >/dev/null 2>&1; then
  exec poetry --project "$PROJECT_ROOT" run wakelab "${APP_ARGS[@]}"
fi

VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [[ -x "$VENV_PYTHON" ]]; then
  export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
  exec "$VENV_PYTHON" -m orac_wake_lab.app "${APP_ARGS[@]}"
fi

cat >&2 <<EOF
Error: unable to launch WakeLab.

Neither Poetry nor a project-local virtual environment was found.
Install Poetry or create a virtual environment at:
  $PROJECT_ROOT/.venv
EOF
exit 1
