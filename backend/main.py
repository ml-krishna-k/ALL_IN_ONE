from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import os
import logging
import traceback
from datetime import datetime
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EverydayAI Backend API",
    description="AI-powered fitness, recipe, and task planning API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - Update origins for production
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:3001", 
    "https://your-frontend-domain.com"  # Replace with your actual frontend domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize OpenAI client with error handling
try:
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    
    client = OpenAI(
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1"
    )
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    client = None

# Request models
class FitnessRequest(BaseModel):
    age: str
    weight: str
    height: str
    fitness_goal: str
    fitness_level: str
    available_days: str

class RecipeRequest(BaseModel):
    query: str

class TaskRequest(BaseModel):
    user_name: str
    tasks: List[str]

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )

# Health check endpoint
@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "EverydayAI Backend API",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    """Detailed health check endpoint"""
    try:
        # Check if OpenAI client is available
        client_status = "connected" if client else "disconnected"
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "api": "operational",
                "openai_client": client_status
            },
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

# Fitness endpoint
@app.post("/fitness")
def fitness_plan(req: FitnessRequest):
    try:
        if not client:
            raise HTTPException(status_code=503, detail="AI service unavailable")
        
        logger.info(f"Fitness plan request for age: {req.age}, weight: {req.weight}")
        
        from fitness import generate_fitness_plan
        result = generate_fitness_plan(
            req.age, req.weight, req.height, 
            req.fitness_goal, req.fitness_level, req.available_days
        )
        
        logger.info("Fitness plan generated successfully")
        return {"result": result, "timestamp": datetime.utcnow().isoformat()}
        
    except Exception as e:
        logger.error(f"Fitness plan error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate fitness plan: {str(e)}")

# Recipe endpoint
@app.post("/recipe")
def recipe(req: RecipeRequest):
    try:
        if not client:
            raise HTTPException(status_code=503, detail="AI service unavailable")
        
        logger.info(f"Recipe request for query: {req.query[:50]}...")
        
        from recipie import generate_recipe
        result = generate_recipe(req.query)
        
        logger.info("Recipe generated successfully")
        return {"result": result, "timestamp": datetime.utcnow().isoformat()}
        
    except Exception as e:
        logger.error(f"Recipe generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recipe: {str(e)}")

# Task planner endpoint
@app.post("/taskplan")
def task_plan(req: TaskRequest):
    try:
        if not client:
            raise HTTPException(status_code=503, detail="AI service unavailable")
        
        logger.info(f"Task plan request for user: {req.user_name}, tasks: {len(req.tasks)}")
        
        from taskplanner import generate_task_plan
        result = generate_task_plan(req.user_name, req.tasks)
        
        logger.info("Task plan generated successfully")
        return {"result": result, "timestamp": datetime.utcnow().isoformat()}
        
    except Exception as e:
        logger.error(f"Task plan error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate task plan: {str(e)}")

# Update the run block for both local and production
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)