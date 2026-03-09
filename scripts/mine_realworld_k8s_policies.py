import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

API_BASE = "https://api.github.com"
NETWORKPOLICY_RE = re.compile(r"kind\s*:\s*NetworkPolicy", re.IGNORECASE)


def _headers(token: Optional[str]) -> Dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request_json(url: str, *, token: Optional[str], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    resp = requests.get(url, headers=_headers(token), params=params, timeout=30)
    if resp.status_code >= 400:
        raise RuntimeError(f"GitHub API error {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected GitHub API response")
    return data


def _search_repos(query: str, *, token: Optional[str], max_repos: int, sleep_s: float) -> List[Dict[str, Any]]:
    repos: List[Dict[str, Any]] = []
    page = 1
    per_page = 100
    while len(repos) < max_repos:
        params = {"q": query, "per_page": min(per_page, max_repos - len(repos)), "page": page}
        data = _request_json(f"{API_BASE}/search/repositories", token=token, params=params)
        items = data.get("items") if isinstance(data.get("items"), list) else []
        if not items:
            break
        for it in items:
            if not isinstance(it, dict):
                continue
            owner = (it.get("owner") or {}).get("login")
            name = it.get("name")
            if isinstance(owner, str) and isinstance(name, str):
                repos.append(
                    {
                        "owner": owner,
                        "repo": name,
                        "stars": it.get("stargazers_count"),
                        "default_branch": it.get("default_branch"),
                        "html_url": it.get("html_url"),
                    }
                )
            if len(repos) >= max_repos:
                break
        page += 1
        if page > 10:
            break
        time.sleep(sleep_s)
    return repos


def _search_code(query: str, *, token: Optional[str], max_items: int, sleep_s: float) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    page = 1
    per_page = 100
    while len(items) < max_items:
        params = {"q": query, "per_page": min(per_page, max_items - len(items)), "page": page}
        data = _request_json(f"{API_BASE}/search/code", token=token, params=params)
        xs = data.get("items") if isinstance(data.get("items"), list) else []
        if not xs:
            break
        for it in xs:
            if not isinstance(it, dict):
                continue
            repo = it.get("repository") if isinstance(it.get("repository"), dict) else {}
            owner = (repo.get("owner") or {}).get("login") if isinstance(repo.get("owner"), dict) else None
            name = repo.get("name")
            path = it.get("path")
            if not (isinstance(owner, str) and isinstance(name, str) and isinstance(path, str)):
                continue
            items.append(
                {
                    "owner": owner,
                    "repo": name,
                    "path": path,
                    "branch": repo.get("default_branch") if isinstance(repo.get("default_branch"), str) else None,
                    "stars": repo.get("stargazers_count") if isinstance(repo.get("stargazers_count"), int) else None,
                }
            )
            if len(items) >= max_items:
                break
        page += 1
        if page > 10:
            break
        time.sleep(sleep_s)
    return items


def _load_repo_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            out: List[Dict[str, Any]] = []
            for it in data:
                if isinstance(it, str) and "/" in it:
                    owner, repo = it.split("/", 1)
                    out.append({"owner": owner, "repo": repo})
                elif isinstance(it, dict) and it.get("owner") and it.get("repo"):
                    out.append({"owner": it.get("owner"), "repo": it.get("repo")})
            return out
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out = []
    for ln in lines:
        if "/" in ln:
            owner, repo = ln.split("/", 1)
            out.append({"owner": owner, "repo": repo})
    return out


def _get_repo_info(owner: str, repo: str, *, token: Optional[str]) -> Dict[str, Any]:
    return _request_json(f"{API_BASE}/repos/{owner}/{repo}", token=token)


def _repo_stars(
    owner: str,
    repo: str,
    *,
    token: Optional[str],
    cache: Dict[Tuple[str, str], Optional[int]],
) -> Optional[int]:
    key = (owner, repo)
    if key in cache:
        return cache[key]
    try:
        info = _get_repo_info(owner, repo, token=token)
    except Exception:
        cache[key] = None
        return None
    stars = info.get("stargazers_count")
    cache[key] = int(stars) if isinstance(stars, int) else None
    return cache[key]


def _get_repo_tree(owner: str, repo: str, branch: str, *, token: Optional[str]) -> List[Dict[str, Any]]:
    data = _request_json(f"{API_BASE}/repos/{owner}/{repo}/git/trees/{branch}", token=token, params={"recursive": 1})
    tree = data.get("tree") if isinstance(data.get("tree"), list) else []
    out: List[Dict[str, Any]] = []
    for it in tree:
        if not isinstance(it, dict):
            continue
        if it.get("type") != "blob":
            continue
        path = it.get("path")
        if not isinstance(path, str):
            continue
        if not (path.lower().endswith(".yml") or path.lower().endswith(".yaml")):
            continue
        out.append(it)
    return out


def _download_raw(owner: str, repo: str, branch: str, path: str, *, token: Optional[str]) -> Optional[str]:
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    resp = requests.get(raw_url, headers=_headers(token), timeout=30)
    if resp.status_code != 200:
        return None
    return resp.text


def _save_raw(out_dir: Path, owner: str, repo: str, rel_path: str, content: str) -> Path:
    local_path = out_dir / "raw" / owner / repo / rel_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(content, encoding="utf-8")
    return local_path


def _maybe_sleep(sleep_s: float) -> None:
    if sleep_s > 0:
        time.sleep(sleep_s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=".\\data\\realworld_k8s_sources")
    ap.add_argument("--mode", type=str, default="code", choices=["code", "repo"])
    ap.add_argument("--query", type=str, default='"kind: NetworkPolicy"')
    ap.add_argument("--max_repos", type=int, default=50, help="Only used in --mode repo")
    ap.add_argument("--max_files", type=int, default=500, help="Max YAML files to download that contain NetworkPolicy")
    ap.add_argument("--min_stars", type=int, default=100)
    ap.add_argument("--max_files_per_repo", type=int, default=200)
    ap.add_argument("--max_file_bytes", type=int, default=500_000)
    ap.add_argument("--repo_list", type=str, default=None)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--token", type=str, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    token = args.token or os.getenv("GITHUB_TOKEN")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stars_cache: Dict[Tuple[str, str], Optional[int]] = {}

    if args.repo_list:
        repos = _load_repo_list(Path(args.repo_list))
    elif str(args.mode).strip().lower() == "repo":
        query = f"NetworkPolicy kubernetes stars:>{int(args.min_stars)}"
        repos = _search_repos(query, token=token, max_repos=int(args.max_repos), sleep_s=float(args.sleep))
    else:
        repos = []

    code_items: List[Dict[str, Any]] = []
    if not args.repo_list and str(args.mode).strip().lower() == "code":
        q_base = str(args.query).strip() or '"kind: NetworkPolicy"'
        for ext in ["yaml", "yml"]:
            q = f"{q_base} extension:{ext}"
            try:
                if bool(args.verbose):
                    print({"code_search": q, "max_items": int(args.max_files) * 2}, file=sys.stderr)
                got = _search_code(q, token=token, max_items=int(args.max_files) * 2, sleep_s=float(args.sleep))
                if bool(args.verbose):
                    print({"code_search_items": len(got)}, file=sys.stderr)
                code_items.extend(got)
            except RuntimeError as e:
                print({"code_search_error": str(e), "query": q}, file=sys.stderr)
                break

    seen_files: Set[Tuple[str, str, str]] = set()

    index: List[Dict[str, Any]] = []
    if code_items:
        for it in code_items:
            if len(index) >= int(args.max_files):
                break
            owner = it.get("owner")
            repo = it.get("repo")
            path = it.get("path")
            if not (isinstance(owner, str) and isinstance(repo, str) and isinstance(path, str)):
                continue

            key = (owner, repo, path)
            if key in seen_files:
                continue
            seen_files.add(key)

            stars = it.get("stars")
            if int(args.min_stars) > 0:
                if not isinstance(stars, int):
                    stars = _repo_stars(owner, repo, token=token, cache=stars_cache)
                if isinstance(stars, int) and int(stars) < int(args.min_stars):
                    continue

            branch = it.get("branch") if isinstance(it.get("branch"), str) and it.get("branch") else None
            if not branch:
                info = _get_repo_info(owner, repo, token=token)
                branch = info.get("default_branch") if isinstance(info.get("default_branch"), str) else "main"

            local_path = out_dir / "raw" / owner / repo / path
            if args.resume and local_path.exists():
                index.append(
                    {
                        "owner": owner,
                        "repo": repo,
                        "branch": branch,
                        "path": path,
                        "raw_url": f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}",
                        "local_path": str(local_path),
                        "size": None,
                        "stars": stars,
                    }
                )
                continue

            text = _download_raw(owner, repo, branch, path, token=token)
            if not text:
                continue
            if not NETWORKPOLICY_RE.search(text):
                continue

            if len(text.encode("utf-8")) > int(args.max_file_bytes):
                continue

            saved = _save_raw(out_dir, owner, repo, path, text)
            index.append(
                {
                    "owner": owner,
                    "repo": repo,
                    "branch": branch,
                    "path": path,
                    "raw_url": f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}",
                    "local_path": str(saved),
                    "size": None,
                    "stars": stars,
                }
            )
            _maybe_sleep(float(args.sleep))

    for repo_info in repos:
        if len(index) >= int(args.max_files):
            break
        owner = repo_info.get("owner")
        repo = repo_info.get("repo")
        if not owner or not repo:
            continue

        try:
            info = _get_repo_info(owner, repo, token=token)
        except Exception:
            continue
        stars = info.get("stargazers_count")
        if isinstance(stars, int) and int(stars) < int(args.min_stars):
            continue
        branch = info.get("default_branch") or repo_info.get("default_branch") or "main"

        try:
            tree = _get_repo_tree(owner, repo, branch, token=token)
        except Exception:
            continue

        file_count = 0
        for it in tree:
            if len(index) >= int(args.max_files):
                break
            path = it.get("path")
            size = it.get("size")
            if not isinstance(path, str):
                continue
            if isinstance(size, int) and size > int(args.max_file_bytes):
                continue
            file_count += 1
            if file_count > int(args.max_files_per_repo):
                break

            key = (owner, repo, path)
            if key in seen_files:
                continue
            seen_files.add(key)

            local_path = out_dir / "raw" / owner / repo / path
            if args.resume and local_path.exists():
                continue

            text = _download_raw(owner, repo, branch, path, token=token)
            if not text:
                continue
            if not NETWORKPOLICY_RE.search(text):
                continue

            saved = _save_raw(out_dir, owner, repo, path, text)
            index.append(
                {
                    "owner": owner,
                    "repo": repo,
                    "branch": branch,
                    "path": path,
                    "raw_url": f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}",
                    "local_path": str(saved),
                    "size": size,
                    "stars": stars,
                }
            )
            _maybe_sleep(float(args.sleep))

        _maybe_sleep(float(args.sleep))

    index_path = out_dir / "index.json"
    if bool(args.resume) and index_path.exists():
        try:
            prev = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            prev = []
        if isinstance(prev, list):
            merged: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
            for it in prev:
                if not isinstance(it, dict):
                    continue
                o = it.get("owner")
                r = it.get("repo")
                p = it.get("path")
                if isinstance(o, str) and isinstance(r, str) and isinstance(p, str):
                    merged[(o, r, p)] = it
            for it in index:
                if not isinstance(it, dict):
                    continue
                o = it.get("owner")
                r = it.get("repo")
                p = it.get("path")
                if isinstance(o, str) and isinstance(r, str) and isinstance(p, str):
                    merged[(o, r, p)] = it
            index = list(merged.values())

    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    repo_set = {(it.get("owner"), it.get("repo")) for it in index if isinstance(it, dict)}
    repo_set = {x for x in repo_set if isinstance(x, tuple) and len(x) == 2 and x[0] and x[1]}
    print({"repos": len(repo_set), "files": len(index), "out": str(index_path)})


if __name__ == "__main__":
    main()
