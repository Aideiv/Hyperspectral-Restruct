"""
API module for Hyperspectral-Restruct Nitrogen Prediction.

Provides endpoints for predicting soil nitrogen content from hyperspectral cubes
and downloading resulting GeoTIFF maps.
"""

import logging
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
import rasterio
from rasterio.transform import from_origin
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse

from configs.constants import API_DEFAULTS, NITROGEN_SWIR_BANDS

logger = logging.getLogger(__name__)

# Global session storage for ONNX runtime
_ort_session: Optional[ort.InferenceSession] = None
_latest_tif_path: Optional[Path] = None


def get_model_path() -> str:
    """Get the ONNX model path from environment or use default."""
    return os.getenv("MODEL_PATH", API_DEFAULTS["model_path"])


def load_onnx_session(model_path: str) -> ort.InferenceSession:
    """
    Load ONNX runtime session with error handling.
    
    Args:
        model_path: Path to the ONNX model file
        
    Returns:
        ort.InferenceSession: Loaded inference session
        
    Raises:
        HTTPException: If model file doesn't exist or fails to load
    """
    if not Path(model_path).exists():
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found: {model_path}. Please ensure the model is trained and exported to ONNX."
        )
    
    try:
        # Use CPU execution provider for edge deployment
        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)
        logger.info(f"Loaded ONNX model from: {model_path}")
        return session
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load ONNX model: {str(e)}"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    global _ort_session
    # Startup: Load model
    model_path = get_model_path()
    _ort_session = load_onnx_session(model_path)
    yield
    # Shutdown: Cleanup
    _ort_session = None


app = FastAPI(
    title=API_DEFAULTS["title"],
    version=API_DEFAULTS["version"],
    lifespan=lifespan
)

def validate_cube_shape(cube: np.ndarray) -> np.ndarray:
    """
    Validate and normalize hyperspectral cube shape.
    
    Args:
        cube: Input array with shape (bands, H, W) or (1, bands, H, W)
        
    Returns:
        Normalized array with shape (1, bands, H, W)
        
    Raises:
        HTTPException: If shape is invalid
    """
    if len(cube.shape) not in [3, 4]:
        raise HTTPException(
            400,
            f"Invalid cube shape: {cube.shape}. Expected (bands, H, W) or (1, bands, H, W)"
        )
    
    # Ensure 4D shape (batch, bands, H, W)
    if len(cube.shape) == 3:
        cube = cube[np.newaxis, ...]
    
    # Validate dimensions
    if cube.shape[0] != 1:
        raise HTTPException(
            400,
            f"Invalid batch dimension: {cube.shape[0]}. Expected 1 for single sample."
        )
    
    return cube


def parse_geotransform(geotransform: Optional[str]) -> tuple[float, float, float, float]:
    """
    Parse geotransform string into components.
    
    Args:
        geotransform: Comma-separated string "ulx,uly,resx,resy" or None
        
    Returns:
        Tuple of (ulx, uly, resx, resy)
        
    Raises:
        HTTPException: If parsing fails
    """
    if geotransform:
        try:
            parts = geotransform.split(",")
            if len(parts) != 4:
                raise ValueError("Expected 4 comma-separated values")
            return tuple(float(p.strip()) for p in parts)
        except ValueError as e:
            raise HTTPException(
                400,
                f"Invalid geotransform format: '{geotransform}'. Expected 'ulx,uly,resx,resy'. Error: {e}"
            )
    return 0.0, 0.0, 1.0, -1.0


@app.post("/predict_nitrogen")
async def predict_nitrogen(
    hsi_cube: UploadFile = File(..., description=".npy file containing hyperspectral cube (bands x H x W)"),
    geotransform: Optional[str] = None,
) -> JSONResponse:
    """
    Predict soil nitrogen from hyperspectral cube.
    
    Returns JSON metrics and stores a GeoTIFF false-color map that can be
    downloaded via the /download_map endpoint.
    
    Args:
        hsi_cube: Numpy array file (.npy) with hyperspectral data
        geotransform: Optional "ulx,uly,resx,resy" for GeoTIFF georeferencing
        
    Returns:
        JSONResponse with prediction statistics and download URL
    """
    global _latest_tif_path, _ort_session
    
    # Validate file extension
    if not hsi_cube.filename or not hsi_cube.filename.endswith(".npy"):
        raise HTTPException(400, "Upload must be a .npy hyperspectral cube file")
    
    # Check model is loaded
    if _ort_session is None:
        raise HTTPException(500, "Model not loaded. Please try again later.")

    # Create temp file for uploaded cube
    cube_path: Optional[str] = None
    tif_path: Optional[str] = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
            shutil.copyfileobj(hsi_cube.file, tmp)
            cube_path = tmp.name

        # Load and validate cube
        try:
            cube = np.load(cube_path).astype(np.float32)
        except Exception as e:
            raise HTTPException(400, f"Failed to load .npy file: {str(e)}")
        
        cube = validate_cube_shape(cube)

        # Run ONNX inference
        input_name = _ort_session.get_inputs()[0].name
        outputs = _ort_session.run(None, {input_name: cube})
        n_map = outputs[0].squeeze()  # (H, W) nitrogen map

        # Compute statistics
        mean_n = float(n_map.mean())
        max_n = float(n_map.max())
        min_n = float(n_map.min())

        # Generate GeoTIFF
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_tif:
            tif_path = tmp_tif.name

        ulx, uly, resx, resy = parse_geotransform(geotransform)

        try:
            with rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                height=n_map.shape[0],
                width=n_map.shape[1],
                count=1,
                dtype=n_map.dtype,
                crs="EPSG:4326",
                transform=from_origin(ulx, uly, resx, resy),
            ) as dst:
                dst.write(n_map, 1)
        except Exception as e:
            raise HTTPException(500, f"Failed to create GeoTIFF: {str(e)}")

        # Store path for download endpoint
        _latest_tif_path = Path(tif_path)

        return JSONResponse({
            "status": "success",
            "mean_nitrogen_percent": round(mean_n, 3),
            "min_nitrogen": round(min_n, 3),
            "max_nitrogen": round(max_n, 3),
            "map_shape": list(n_map.shape),
            "swir_bands_used": NITROGEN_SWIR_BANDS,
            "download_url": "/download_map"
        })

    finally:
        # Cleanup input temp file
        if cube_path:
            Path(cube_path).unlink(missing_ok=True)

@app.get("/download_map")
async def download_map() -> FileResponse:
    """
    Download the latest generated GeoTIFF nitrogen map.
    
    Returns:
        FileResponse with the GeoTIFF file
        
    Raises:
        HTTPException: If no map has been generated yet
    """
    global _latest_tif_path
    
    if _latest_tif_path is None or not _latest_tif_path.exists():
        raise HTTPException(
            404,
            "No nitrogen map available. Please run /predict_nitrogen first."
        )
    
    return FileResponse(
        path=str(_latest_tif_path),
        media_type="image/tiff",
        filename="nitrogen_map.tif"
    )


@app.get("/health")
async def health_check() -> Dict:
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "model_loaded": _ort_session is not None,
        "version": API_DEFAULTS["version"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", API_DEFAULTS["host"]),
        port=int(os.getenv("API_PORT", API_DEFAULTS["port"]))
    )
