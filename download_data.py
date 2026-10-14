"""Auto-download Elliptic Bitcoin Dataset from Kaggle and preprocess graph.

Called at dashboard startup on Streamlit Cloud (or any fresh environment).
Skips all work if data already exists locally.

Requires ONE of these in Streamlit secrets:

  Option A – new single-token format (recommended):
    [kaggle]
    api_token = "KGAT_529a4f9ebf890efecd8788574cf3efee"

  Option B – legacy username+key format:
    [kaggle]
    username = "your_kaggle_username"
    key      = "your_kaggle_api_key"
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (mirrors config.py without importing it to stay dependency-light)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GRAPH_CACHE = PROCESSED_DIR / "graph_data.pt"

FEATURES_CSV = RAW_DIR / "elliptic_txs_features.csv"
CLASSES_CSV  = RAW_DIR / "elliptic_txs_classes.csv"
EDGELIST_CSV = RAW_DIR / "elliptic_txs_edgelist.csv"

KAGGLE_DATASET = "ellipticco/elliptic-data-set"


def _ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _kaggle_credentials_from_streamlit() -> str | tuple[str, str] | None:
    """Read Kaggle creds from st.secrets.

    Returns:
        str  – new-style single API token  (KGAT_...)
        tuple – legacy (username, key) pair
        None  – no credentials found
    """
    try:
        import streamlit as st
        secrets = st.secrets.get("kaggle", {})
        # New single-token format
        if "api_token" in secrets:
            return secrets["api_token"]
        # Legacy username+key
        if "username" in secrets and "key" in secrets:
            return secrets["username"], secrets["key"]
    except Exception:
        pass
    return None


def _setup_kaggle_env(creds: str | tuple[str, str]) -> None:
    """Inject Kaggle credentials into the environment so kaggle-cli finds them."""
    if isinstance(creds, str):
        # New single-token format
        os.environ["KAGGLE_API_TOKEN"] = creds
    else:
        username, key = creds
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"]      = key


def _download_elliptic() -> None:
    """Download and unzip the Elliptic dataset into data/raw/."""
    print("📥 Downloading Elliptic dataset from Kaggle …")
    import subprocess
    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-p", str(RAW_DIR),
            "--unzip",
            "--force",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed:\n{result.stderr}\n{result.stdout}"
        )
    print("✅ Elliptic dataset downloaded.")


def _build_graph_cache() -> None:
    """Run graph_builder to create the processed_graph.pt cache."""
    print("⚙️  Building graph cache (this takes ~2 min on first run) …")
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.preprocessing.graph_builder import build_graph
    build_graph()
    print("✅ Graph cache created.")


def ensure_data_ready(streamlit_ui: bool = True) -> bool:
    """
    Main entry point. Returns True if data is ready (existed or just downloaded).
    Returns False if credentials are missing and data isn't present.
    """
    _ensure_dirs()

    # ---- Fast path: everything already exists --------------------------------
    if GRAPH_CACHE.exists() and FEATURES_CSV.exists():
        return True

    # ---- Need to download ---------------------------------------------------
    creds = _kaggle_credentials_from_streamlit()
    if creds is None:
        # Fallback: try environment variables already set
        token = os.environ.get("KAGGLE_API_TOKEN")
        if token:
            creds = token
        else:
            username = os.environ.get("KAGGLE_USERNAME")
            key      = os.environ.get("KAGGLE_KEY")
            if username and key:
                creds = (username, key)

    if creds is None:
        if streamlit_ui:
            import streamlit as st
            st.error(
                "⚠️ **Data not found and Kaggle credentials not configured.**\n\n"
                "Add your Kaggle API token to the app's Streamlit secrets:\n"
                "```toml\n[kaggle]\napi_token = \"KGAT_xxxxxxxxxxxxxxxxxxxx\"\n```\n"
                "Then click **Manage app → Reboot app**."
            )
            st.info(
                "💡 Get your token at [kaggle.com/settings](https://www.kaggle.com/settings) "
                "→ API → Create New Token"
            )
            st.stop()  # Halt — don't fall through to load_graph()
        return False

    _setup_kaggle_env(creds)

    # Install kaggle CLI if not present
    try:
        import kaggle  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])

    if streamlit_ui:
        import streamlit as st
        if not FEATURES_CSV.exists():
            with st.spinner("📥 Downloading Elliptic dataset from Kaggle (~658 MB, ~3 min)…"):
                _download_elliptic()
        if not GRAPH_CACHE.exists():
            with st.spinner("⚙️ Building graph cache (~2 min on first run)…"):
                _build_graph_cache()
    else:
        if not FEATURES_CSV.exists():
            _download_elliptic()
        if not GRAPH_CACHE.exists():
            _build_graph_cache()

    return True


# ---------------------------------------------------------------------------
# Allow running directly: python download_data.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download & preprocess Elliptic dataset")
    parser.add_argument("--kaggle-token",    default=os.environ.get("KAGGLE_API_TOKEN", ""),
                        help="New-style single API token (KGAT_...)")
    parser.add_argument("--kaggle-username", default=os.environ.get("KAGGLE_USERNAME", ""),
                        help="Legacy: Kaggle username")
    parser.add_argument("--kaggle-key",      default=os.environ.get("KAGGLE_KEY", ""),
                        help="Legacy: Kaggle API key")
    args = parser.parse_args()

    if args.kaggle_token:
        _setup_kaggle_env(args.kaggle_token)
    elif args.kaggle_username and args.kaggle_key:
        _setup_kaggle_env((args.kaggle_username, args.kaggle_key))

    ok = ensure_data_ready(streamlit_ui=False)
    sys.exit(0 if ok else 1)
