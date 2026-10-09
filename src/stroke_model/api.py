"""FastAPI service exposing model inference and health endpoints."""

from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Literal, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from stroke_model.predict import StrokePredictor

try:
    # Software version, read from the package metadata set in pyproject.toml
    API_VERSION = package_version("stroke-model")
except PackageNotFoundError:
    from stroke_model import __version__ as API_VERSION

# Global predictor instance loaded on startup
predictor: Optional[StrokePredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifact on application startup."""
    global predictor
    try:
        predictor = StrokePredictor()
    except Exception as exc:
        print(f"Warning: Failed to load model artifact at startup: {exc}")
        predictor = None
    yield


app = FastAPI(
    title="Stroke 14-Day Mortality Prediction Service",
    description="Research demonstration for acute stroke mortality risk assessment. Not for clinical use.",
    version=API_VERSION,
    lifespan=lifespan
)


# ==============================================================================
# PATIENT INPUT SCHEMA
#
# Rule: Authored/reviewed by the domain expert (medical doctor/health data scientist).
# Each field defines data type, clinical unit, and allowed range.
# ==============================================================================

class PatientRecord(BaseModel):
    """Clinical predictors for 14-day post-stroke mortality assessment."""

    # Numeric features
    age: float = Field(
        ...,
        ge=0,
        le=120,
        description="Age in years (0 to 120). The training data covers ages 16 to 98",
        examples=[72.0]
    )
    sbp: float = Field(
        ...,
        ge=50,
        le=300,
        description="Systolic blood pressure at randomisation in mmHg (50 to 300). The training data covers 70 to 295",
        examples=[150.0]
    )
    delay: float = Field(
        ...,
        ge=0,
        le=100,
        description="Delay between stroke onset and randomisation in hours (0 to 100). The training data covers 1 to 48",
        examples=[14.0]
    )

    # Demographic & Clinical categoricals
    gender: Literal["M", "F"] = Field(
        ...,
        description="Sex ('M' = male, 'F' = female)",
        examples=["F"]
    )
    consc: Literal["D", "F"] = Field(
        ...,
        description="Conscious state at randomisation ('F' = fully alert, 'D' = drowsy). Patients recorded as unconscious are not in the training data",
        examples=["F"]
    )
    subtype: Literal["LACS", "OTH", "PACS", "POCS", "TACS"] = Field(
        ...,
        description="Stroke subtype (Oxford Community Stroke Project classification): 'LACS' lacunar, 'PACS' partial anterior circulation, 'POCS' posterior circulation, 'TACS' total anterior circulation, 'OTH' other",
        examples=["PACS"]
    )
    treat1: Literal["L", "M", "N"] = Field(
        ...,
        description="Heparin allocated in the trial ('N' = none, 'L' = low dose, 'M' = medium dose)",
        examples=["N"]
    )
    treat2: Literal["N", "Y"] = Field(
        ...,
        description="Aspirin allocated in the trial ('Y' = aspirin, 'N' = no aspirin)",
        examples=["N"]
    )

    # Clinical flags ('Y' / 'N')
    wakesym: Literal["Y", "N"] = Field(..., description="Symptoms noted on waking ('Y'/'N')", examples=["N"])
    atrial: Literal["Y", "N"] = Field(..., description="Atrial fibrillation ('Y'/'N')", examples=["N"])
    CT: Literal["Y", "N"] = Field(..., description="CT scan before randomisation ('Y'/'N')", examples=["Y"])
    Infarc: Literal["Y", "N"] = Field(..., description="Infarct visible on CT ('Y'/'N')", examples=["Y"])
    hep24: Literal["Y", "N"] = Field(..., description="Heparin in the 24 hours before randomisation ('Y'/'N')", examples=["N"])
    asp3: Literal["Y", "N"] = Field(..., description="Aspirin in the 3 days before randomisation ('Y'/'N')", examples=["N"])

    # Neurological deficits at randomisation ('Y' / 'N'); symptom1 to symptom8 are IST variables RDEF1 to RDEF8
    symptom1: Literal["Y", "N"] = Field(..., description="Face deficit at randomisation ('Y'/'N')", examples=["Y"])
    symptom2: Literal["Y", "N"] = Field(..., description="Arm or hand deficit at randomisation ('Y'/'N')", examples=["Y"])
    symptom3: Literal["Y", "N"] = Field(..., description="Leg or foot deficit at randomisation ('Y'/'N')", examples=["N"])
    symptom4: Literal["Y", "N"] = Field(..., description="Dysphasia at randomisation ('Y'/'N')", examples=["N"])
    symptom5: Literal["Y", "N"] = Field(..., description="Hemianopia at randomisation ('Y'/'N')", examples=["N"])
    symptom6: Literal["Y", "N"] = Field(..., description="Visuospatial disorder at randomisation ('Y'/'N')", examples=["N"])
    symptom7: Literal["Y", "N"] = Field(..., description="Brainstem or cerebellar signs at randomisation ('Y'/'N')", examples=["N"])
    symptom8: Literal["Y", "N"] = Field(..., description="Other deficit at randomisation ('Y'/'N')", examples=["N"])


class PredictionResponse(BaseModel):
    """Standardised prediction output structure."""
    mortality_probability: float = Field(..., description="Estimated probability of 14-day mortality (0.0 to 1.0)")
    high_risk_flag: bool = Field(..., description="True if the estimated probability is at or above the Youden classification threshold")
    threshold_applied: float = Field(..., description="Decision boundary threshold selected via Youden's J statistic")
    model_version: str = Field(..., description="Version of the serialised model pipeline")
    disclaimer: str = Field(..., description="Regulatory safety disclaimer")


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/", include_in_schema=False)
def root():
    """Redirect the bare URL to the interactive documentation page."""
    return RedirectResponse(url="/docs")


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Verify service availability and model loading status."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded."
        )
    return {
        "status": "healthy",
        "model_version": predictor.model_version,
        "threshold": predictor.threshold
    }


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_mortality(patient: PatientRecord):
    """Estimate 14-day mortality risk for a single validated patient record."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not initialised."
        )
    result = predictor.predict_record(patient.model_dump())
    return result
