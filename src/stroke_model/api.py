"""FastAPI service exposing model inference and health endpoints."""

from contextlib import asynccontextmanager
from typing import Literal, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from stroke_model.predict import StrokePredictor

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
    version="0.1.0",
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
        description="Patient age in years (0 to 120)",
        examples=[72.0]
    )
    sbp: float = Field(
        ...,
        ge=50,
        le=300,
        description="Systolic blood pressure in mmHg (50 to 300)",
        examples=[150.0]
    )
    delay: float = Field(
        ...,
        ge=0,
        le=100,
        description="Time from symptom onset to hospital presentation in hours (0 to 100)",
        examples=[14.0]
    )

    # Demographic & Clinical categoricals
    gender: Literal["M", "F"] = Field(
        ...,
        description="Biological sex ('M' = Male, 'F' = Female)",
        examples=["F"]
    )
    consc: Literal["D", "F"] = Field(
        ...,
        description="Consciousness level ('D' = Drowsy, 'F' = Fully alert)",
        examples=["F"]
    )
    subtype: Literal["LACS", "OTH", "PACS", "POCS", "TACS"] = Field(
        ...,
        description="Oxford Community Stroke Project clinical classification ('LACS', 'OTH', 'PACS', 'POCS', 'TACS')",
        examples=["PACS"]
    )
    treat1: Literal["L", "M", "N"] = Field(
        ...,
        description="Treatment code 1 ('L', 'M', 'N')",
        examples=["N"]
    )
    treat2: Literal["N", "Y"] = Field(
        ...,
        description="Treatment code 2 ('N', 'Y')",
        examples=["N"]
    )

    # Clinical flags ('Y' / 'N')
    wakesym: Literal["Y", "N"] = Field(..., description="Symptoms present upon waking ('Y'/'N')", examples=["N"])
    atrial: Literal["Y", "N"] = Field(..., description="History of atrial fibrillation ('Y'/'N')", examples=["N"])
    CT: Literal["Y", "N"] = Field(..., description="CT scan performed prior to admission ('Y'/'N')", examples=["Y"])
    Infarc: Literal["Y", "N"] = Field(..., description="Cerebral infarction confirmed ('Y'/'N')", examples=["Y"])
    hep24: Literal["Y", "N"] = Field(..., description="Heparin administered within 24 hours ('Y'/'N')", examples=["N"])
    asp3: Literal["Y", "N"] = Field(..., description="Aspirin administered within 3 days ('Y'/'N')", examples=["N"])

    # Neurological deficit symptoms ('Y' / 'N')
    symptom1: Literal["Y", "N"] = Field(..., description="Symptom 1 deficit flag", examples=["Y"])
    symptom2: Literal["Y", "N"] = Field(..., description="Symptom 2 deficit flag", examples=["Y"])
    symptom3: Literal["Y", "N"] = Field(..., description="Symptom 3 deficit flag", examples=["N"])
    symptom4: Literal["Y", "N"] = Field(..., description="Symptom 4 deficit flag", examples=["N"])
    symptom5: Literal["Y", "N"] = Field(..., description="Symptom 5 deficit flag", examples=["N"])
    symptom6: Literal["Y", "N"] = Field(..., description="Symptom 6 deficit flag", examples=["N"])
    symptom7: Literal["Y", "N"] = Field(..., description="Symptom 7 deficit flag", examples=["N"])
    symptom8: Literal["Y", "N"] = Field(..., description="Symptom 8 deficit flag", examples=["N"])


class PredictionResponse(BaseModel):
    """Standardised prediction output structure."""
    mortality_probability: float = Field(..., description="Estimated probability of 14-day mortality (0.0 to 1.0)")
    high_risk_flag: bool = Field(..., description="True if estimated probability exceeds the Youden classification threshold")
    threshold_applied: float = Field(..., description="Decision boundary threshold selected via Youden's J statistic")
    model_version: str = Field(..., description="Version of the serialised model pipeline")
    disclaimer: str = Field(..., description="Regulatory safety disclaimer")


# ==============================================================================
# ENDPOINTS
# ==============================================================================

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
