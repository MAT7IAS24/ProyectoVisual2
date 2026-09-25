#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parent
REMOTE_NAME = "upstream"
REMOTE_URL = "https://github.com/uexternadojz/pulso-transmi-sdk.git"
DEFAULT_BRANCH = "main"


def run(cmd: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, cwd=str(cwd or ROOT), text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{result.stderr.strip()}")
    return result.stdout.strip()


def sql_literal(value):
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("'", "''")
    return "'" + text + "'"


def build_snapshot() -> dict:
    run(["git", "fetch", REMOTE_NAME, "--prune"], cwd=ROOT)

    commit_info = run(
        ["git", "log", "-1", "--format=%H%x1f%an%x1f%ae%x1f%s%x1f%cI", f"{REMOTE_NAME}/{DEFAULT_BRANCH}"],
        cwd=ROOT,
    )
    sha, author_name, author_email, message, committed_at = commit_info.split("\x1f")

    try:
        readme = run(["git", "show", f"{REMOTE_NAME}/{DEFAULT_BRANCH}:README.md"], cwd=ROOT)
        readme_excerpt = "\n".join(readme.splitlines()[:25])
    except RuntimeError:
        readme_excerpt = "README.md not found in upstream branch"

    files = run(["git", "ls-tree", "-r", "--name-only", f"{REMOTE_NAME}/{DEFAULT_BRANCH}"], cwd=ROOT)
    file_count = len([line for line in files.splitlines() if line.strip()])

    metadata = {
        "remote_name": REMOTE_NAME,
        "remote_url": REMOTE_URL,
        "default_branch": DEFAULT_BRANCH,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "files": file_count,
        "readme_excerpt": readme_excerpt,
    }

    return {
        "repo_name": "pulso-transmi-sdk",
        "repo_url": REMOTE_URL,
        "default_branch": DEFAULT_BRANCH,
        "commit_sha": sha,
        "short_sha": sha[:12],
        "author_name": author_name,
        "author_email": author_email,
        "commit_message": message,
        "committed_at": committed_at,
        "readme_excerpt": readme_excerpt,
        "file_count": file_count,
        "metadata_json": metadata,
    }


def upsert_snapshot(payload: dict) -> str:
    metadata_json = json.dumps(payload["metadata_json"], ensure_ascii=False)
    sql = """
    INSERT INTO public.upstream_repo_snapshots (
        repo_name,
        repo_url,
        default_branch,
        commit_sha,
        short_sha,
        author_name,
        author_email,
        commit_message,
        committed_at,
        readme_excerpt,
        file_count,
        metadata_json
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    ON CONFLICT (repo_url, commit_sha)
    DO UPDATE SET
        repo_name = EXCLUDED.repo_name,
        default_branch = EXCLUDED.default_branch,
        short_sha = EXCLUDED.short_sha,
        author_name = EXCLUDED.author_name,
        author_email = EXCLUDED.author_email,
        commit_message = EXCLUDED.commit_message,
        committed_at = EXCLUDED.committed_at,
        readme_excerpt = EXCLUDED.readme_excerpt,
        file_count = EXCLUDED.file_count,
        metadata_json = EXCLUDED.metadata_json,
        fetched_at = NOW()
    RETURNING snapshot_id;
    """

    db_url = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
    if db_url:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        payload["repo_name"],
                        payload["repo_url"],
                        payload["default_branch"],
                        payload["commit_sha"],
                        payload["short_sha"],
                        payload["author_name"],
                        payload["author_email"],
                        payload["commit_message"],
                        payload["committed_at"],
                        payload["readme_excerpt"][:2000],
                        payload["file_count"],
                        json.loads(metadata_json),
                    ),
                )
                row = cur.fetchone()
                return json.dumps({"snapshot_id": row[0]}) if row else "No output"

    fallback_sql = """
    INSERT INTO public.upstream_repo_snapshots (
        repo_name,
        repo_url,
        default_branch,
        commit_sha,
        short_sha,
        author_name,
        author_email,
        commit_message,
        committed_at,
        readme_excerpt,
        file_count,
        metadata_json
    ) VALUES (
        {repo_name},
        {repo_url},
        {default_branch},
        {commit_sha},
        {short_sha},
        {author_name},
        {author_email},
        {commit_message},
        {committed_at},
        {readme_excerpt},
        {file_count},
        '{metadata_json}'::jsonb
    )
    ON CONFLICT (repo_url, commit_sha)
    DO UPDATE SET
        repo_name = EXCLUDED.repo_name,
        default_branch = EXCLUDED.default_branch,
        short_sha = EXCLUDED.short_sha,
        author_name = EXCLUDED.author_name,
        author_email = EXCLUDED.author_email,
        commit_message = EXCLUDED.commit_message,
        committed_at = EXCLUDED.committed_at,
        readme_excerpt = EXCLUDED.readme_excerpt,
        file_count = EXCLUDED.file_count,
        metadata_json = EXCLUDED.metadata_json,
        fetched_at = NOW()
    RETURNING snapshot_id;
    """.format(
        repo_name=sql_literal(payload["repo_name"]),
        repo_url=sql_literal(payload["repo_url"]),
        default_branch=sql_literal(payload["default_branch"]),
        commit_sha=sql_literal(payload["commit_sha"]),
        short_sha=sql_literal(payload["short_sha"]),
        author_name=sql_literal(payload["author_name"]),
        author_email=sql_literal(payload["author_email"]),
        commit_message=sql_literal(payload["commit_message"]),
        committed_at=sql_literal(payload["committed_at"]),
        readme_excerpt=sql_literal(payload["readme_excerpt"][:2000]),
        file_count=sql_literal(payload["file_count"]),
        metadata_json=metadata_json.replace("'", "''"),
    )

    query = [
        "supabase",
        "db",
        "query",
        "--linked",
        "--experimental",
        fallback_sql,
    ]
    result = subprocess.run(query, cwd=str(ROOT), text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Supabase insert failed:\n{result.stderr}\nSQL:\n{fallback_sql}")
    stdout = result.stdout.strip()
    if not stdout:
        return "No output"
    return stdout


def main() -> None:
    snapshot = build_snapshot()
    print(json.dumps({
        "repo": snapshot["repo_url"],
        "branch": snapshot["default_branch"],
        "commit": snapshot["commit_sha"],
        "author": snapshot["author_name"],
        "files": snapshot["file_count"],
        "committed_at": snapshot["committed_at"],
    }, indent=2))

    response = upsert_snapshot(snapshot)
    print("\nPostgres insert result:")
    print(response)


if __name__ == "__main__":
    main()
