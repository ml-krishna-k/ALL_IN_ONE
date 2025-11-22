# Deployment Checklist for Render

## Pre-Deployment Checklist

### ✅ Code Quality

- [x] All Python files have proper error handling
- [x] Logging is implemented throughout the application
- [x] Environment variables are properly validated
- [x] CORS is configured for production
- [x] Health check endpoints are implemented
- [x] API documentation is available at `/docs`

### ✅ Configuration Files

- [x] `render.yaml` - Render deployment configuration
- [x] `requirements.txt` - Python dependencies with versions
- [x] `runtime.txt` - Python version specification
- [x] `Procfile` - Alternative deployment method
- [x] `.env.example` - Environment variables template

### ✅ Production Optimizations

- [x] Gunicorn configuration optimized for production
- [x] Proper worker count and timeout settings
- [x] Request limits and jitter configured
- [x] Error handling and logging implemented
- [x] Health monitoring endpoints

## Deployment Steps

### 1. Environment Variables Setup

In your Render dashboard, set these environment variables:

```
GROQ_API_KEY=your_actual_groq_api_key
PORT=10000
PYTHONPATH=.
```

### 2. Deploy to Render

1. Connect your GitHub repository to Render
2. Select "Web Service"
3. Use the following settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120 --keep-alive 2 --max-requests 1000 --max-requests-jitter 100`
   - **Python Version**: 3.11.9

### 3. Post-Deployment Verification

After deployment, test these endpoints:

#### Health Checks

```bash
curl https://your-app.onrender.com/
curl https://your-app.onrender.com/health
```

#### API Endpoints

```bash
# Fitness Plan
curl -X POST https://your-app.onrender.com/fitness \
  -H "Content-Type: application/json" \
  -d '{"age":"25","weight":"70","height":"175","fitness_goal":"lose weight","fitness_level":"beginner","available_days":"3"}'

# Recipe Generation
curl -X POST https://your-app.onrender.com/recipe \
  -H "Content-Type: application/json" \
  -d '{"query":"chicken curry"}'

# Task Planning
curl -X POST https://your-app.onrender.com/taskplan \
  -H "Content-Type: application/json" \
  -d '{"user_name":"Test","tasks":["task1","task2"]}'
```

#### API Documentation

- Swagger UI: `https://your-app.onrender.com/docs`
- ReDoc: `https://your-app.onrender.com/redoc`

## Monitoring and Maintenance

### Logs

- Monitor application logs in Render dashboard
- Check for any error patterns
- Verify API response times

### Performance

- Monitor response times
- Check memory usage
- Verify worker processes are healthy

### Security

- Ensure GROQ_API_KEY is properly secured
- Verify CORS settings for your frontend domain
- Monitor for any unauthorized access attempts

## Troubleshooting

### Common Issues

1. **502 Bad Gateway**: Check if GROQ_API_KEY is set correctly
2. **Timeout Errors**: Verify Groq API is responding
3. **CORS Issues**: Update allowed origins in main.py
4. **Memory Issues**: Monitor worker memory usage

### Debug Commands

```bash
# Check health status
curl https://your-app.onrender.com/health

# Test with verbose output
curl -v https://your-app.onrender.com/

# Check specific endpoint
curl -X POST https://your-app.onrender.com/fitness \
  -H "Content-Type: application/json" \
  -d '{"age":"25","weight":"70","height":"175","fitness_goal":"test","fitness_level":"beginner","available_days":"3"}' \
  -v
```

## Success Criteria

- [ ] All health check endpoints return 200 OK
- [ ] All API endpoints respond correctly
- [ ] API documentation is accessible
- [ ] No errors in application logs
- [ ] Response times are acceptable (< 5 seconds)
- [ ] CORS is working for your frontend domain

## Next Steps After Deployment

1. Update your frontend to use the new API URL
2. Configure monitoring and alerting
3. Set up automated backups if needed
4. Plan for scaling if traffic increases
5. Consider adding rate limiting for production use
