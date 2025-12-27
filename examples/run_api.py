"""Example script to run the Load Shift Optimizer API server.

This script starts the FastAPI server for the load-shift-optimizer.
Once running, you can:
- Access the API at http://localhost:8000
- View Swagger documentation at http://localhost:8000/docs
- View ReDoc documentation at http://localhost:8000/redoc
"""

import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print("Starting Load Shift Optimizer API Server")
    print("=" * 70)
    print()
    print("Server will be available at:")
    print("  • API Base URL:        http://localhost:8000")
    print("  • Swagger UI:          http://localhost:8000/docs")
    print("  • ReDoc:               http://localhost:8000/redoc")
    print("  • OpenAPI Schema:      http://localhost:8000/openapi.json")
    print()
    print("Available endpoints:")
    print("  • GET  /health                           - Health check")
    print("  • POST /api/v1/optimize                  - Basic optimization")
    print("  • POST /api/v1/optimize/moving-horizon   - Moving horizon optimization")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    print()

    uvicorn.run(
        "loadshift.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info",
    )
