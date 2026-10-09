"""streamlit_app_api.py

Proof-of-concept entry point. It runs the same app as streamlit_app.py, but in API mode: the
form is sent to the stroke prediction API deployed on Render instead of running the model
inside the app. STROKE_API_URL, if already set, takes precedence over the address below.

    streamlit run app/streamlit_app_api.py
"""

import os
import runpy
from pathlib import Path

os.environ.setdefault("STROKE_API_URL", "https://stroke-mortality-model.onrender.com")
runpy.run_path(str(Path(__file__).with_name("streamlit_app.py")), run_name="__main__")
