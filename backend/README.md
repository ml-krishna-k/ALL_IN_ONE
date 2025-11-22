# EverydayAI Backend API

A FastAPI-based backend service that provides AI-powered fitness planning, recipe generation, and task planning using Groq's OpenAI-compatible API.

## Features

- **Fitness Planning**: Generate personalized weekly fitness and meal plans
- **Recipe Generation**: Create detailed recipes with Tamil Nadu/Indian cuisine focus
- **Task Planning**: Generate structured daily/weekly task plans
- **Health Monitoring**: Built-in health check endpoints
- **Error Handling**: Comprehensive error handling and logging
- **Production Ready**: Optimized for deployment on Render

## API Endpoints

### Health Check

- `GET /` - Basic health status
- `GET /health` - Detailed health check with service status

### AI Services

- `POST /fitness` - Generate fitness plans
- `POST /recipe` - Generate recipes
- `POST /taskplan` - Generate task plans

## Request/Response Examples

### Fitness Plan

```json
POST /fitness
{
  "age": "25",
  "weight": "70",
  "height": "175",
  "fitness_goal": "lose weight",
  "fitness_level": "beginner",
  "available_days": "3"
}
```

### Recipe Generation

```json
POST /recipe
{
  "query": "chicken curry with coconut milk"
}
```

### Task Planning

```json
POST /taskplan
{
  "user_name": "John",
  "tasks": ["Prepare presentation", "Gym workout", "Buy groceries"]
}
```

## Environment Variables

Create a `.env` file with the following variables:

```env
GROQ_API_KEY=your_groq_api_key_here
PORT=8000
PYTHONPATH=.
LOG_LEVEL=INFO
```

## Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables (copy `.env.example` to `.env`)
4. Run the application:
   ```bash
   python main.py
   ```
5. Access the API documentation at `http://localhost:8000/docs`

## Deployment on Render

1. Connect your GitHub repository to Render
2. Set the following environment variables in Render:
   - `GROQ_API_KEY`: Your Groq API key
3. Deploy using the provided `render.yaml` configuration

The application will automatically:

- Install dependencies
- Start with Gunicorn and Uvicorn workers
- Handle production traffic efficiently

## API Documentation

Once deployed, access the interactive API documentation at:

- Swagger UI: `https://your-app.onrender.com/docs`
- ReDoc: `https://your-app.onrender.com/redoc`

## Error Handling

The API includes comprehensive error handling:

- Global exception handler for unexpected errors
- Service availability checks
- Detailed error logging
- Proper HTTP status codes

## Logging

The application logs:

- Request/response information
- Error details with stack traces
- Service health status
- AI service interactions

## Dependencies

- FastAPI 0.104.1 - Web framework
- Uvicorn 0.24.0 - ASGI server
- Gunicorn 21.2.0 - WSGI server for production
- OpenAI 1.3.5 - AI client library
- Pydantic 2.5.0 - Data validation
- Python-dotenv 1.0.0 - Environment variable management

## Production Considerations

- Configured for Render deployment
- Optimized Gunicorn settings for production
- Proper CORS configuration
- Health check endpoints for monitoring
- Comprehensive error handling and logging
- Environment variable validation

## Support

For issues or questions, please check the logs and ensure all environment variables are properly configured.
