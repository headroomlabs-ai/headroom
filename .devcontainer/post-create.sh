#!/usr/bin/env bash
set -euo pipefail

profile="${1:-default}"
project_env="${UV_PROJECT_ENVIRONMENT:-/home/vscode/.venvs/headroom}"
project_env_root="$(dirname "$project_env")"
cache_root="${HOME}/.cache"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd "$script_dir/.." && pwd)"
git_profile_script="/etc/profile.d/headroom-worktree-git.sh"
sync_extras=(--extra dev)

if [[ "$profile" == "memory-stack" ]]; then
  sync_extras+=(--extra memory-stack)
fi

cd "$workspace_root"

configure_worktree_git_env() {
  sudo rm -f "$git_profile_script"

  if git rev-parse --show-toplevel >/dev/null 2>&1; then
    return 0
  fi

  if [[ ! -f .git ]]; then
    return 1
  fi

  local gitdir_spec repo_root translated_git_dir translated_common_dir
  gitdir_spec="$(sed -n 's/^gitdir: //p' .git)"

  if [[ -z "$gitdir_spec" || "$gitdir_spec" != *"/.git/worktrees/"* ]]; then
    return 1
  fi

  repo_root="${gitdir_spec%/.git/worktrees/*}"

  if [[ "$gitdir_spec" =~ ^[A-Za-z]:/ ]]; then
    translated_git_dir="/workspaces-host/${gitdir_spec#?:/}"
    translated_common_dir="/workspaces-host/${repo_root#?:/}/.git"
  elif [[ "$gitdir_spec" == /* ]]; then
    translated_git_dir="/workspaces-host${gitdir_spec}"
    translated_common_dir="/workspaces-host${repo_root}/.git"
  else
    return 1
  fi

  if [[ ! -d "$translated_git_dir" || ! -d "$translated_common_dir" ]]; then
    return 1
  fi

  sudo tee "$git_profile_script" >/dev/null <<EOF
export GIT_DIR="$translated_git_dir"
export GIT_COMMON_DIR="$translated_common_dir"
export GIT_WORK_TREE="$workspace_root"
EOF

  # shellcheck disable=SC1091
  source "$git_profile_script"
  git rev-parse --show-toplevel >/dev/null 2>&1
}

sudo mkdir -p "$project_env_root" "$cache_root/uv" "$cache_root/pip" "$cache_root/pre-commit"
sudo chown -R "$(id -u):$(id -g)" "$project_env_root" "$cache_root"

uv sync --frozen "${sync_extras[@]}" --link-mode copy

if configure_worktree_git_env; then
  uv run pre-commit install
else
  echo "Skipping pre-commit install because git metadata is not available inside this container."
fi

echo "Headroom devcontainer is ready."
if [[ "$profile" == "memory-stack" ]]; then
  echo "Memory stack sidecars are available at qdrant:6333 and neo4j://neo4j:7687."
fi
echo "Run checks with:"
echo "  uv run ruff check ."
echo "  uv run ruff format --check ."
echo "  uv run mypy headroom --ignore-missing-imports"
echo "  uv run pytest -v --tb=short"
