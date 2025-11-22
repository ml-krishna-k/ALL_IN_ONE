from openai import OpenAI
import os
import json
import logging
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
# Walk up the directory tree to find .env file
script_path = Path(__file__).resolve()
current_dir = script_path.parent

# Search up to 3 levels for .env file
for _ in range(3):
    env_file = current_dir / '.env'
    if env_file.exists():
        load_dotenv(env_file, override=True)
        break
    parent = current_dir.parent
    if parent == current_dir:  # Reached root
        break
    current_dir = parent

# Also try default location (current working directory)
load_dotenv(override=True)

logger = logging.getLogger(__name__)

# Initialize client with error handling
try:
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("GROQ_API_KEY not found in environment variables")
        raise ValueError("GROQ_API_KEY environment variable is not set")
    
    # Verify API key is loaded (show first and last 4 chars for security)
    logger.info(f"API key loaded: {groq_api_key[:4]}...{groq_api_key[-4:]}")
    
    client = OpenAI(
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1"
    )
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client in taskplanner module: {e}")
    client = None

system_prompt = '''
You are an expert Task Planner and Productivity Assistant. Your goal is to generate a personalized daily or weekly task plan based on user input. Output must always be in **JSON format**, ready for a frontend application.

JSON Format:
{
  "user_name": "string",
  "date": "YYYY-MM-DD",
  "tasks": [
    {
      "task_name": "string",
      "priority": "High / Medium / Low",
      "deadline": "YYYY-MM-DD or time",
      "duration": "estimated time in minutes or hours",
      "category": "Work / Study / Personal / Fitness / Other",
      "notes": "optional",
      "status": "Pending / In Progress / Completed"
    }
  ],
  "general_tips": ["array of productivity tips"]
}

Rules:
1. If some task info is missing (priority, deadline, duration, category), infer intelligently.
2. Suggest priorities, durations, and categories if not provided.
3. Provide practical productivity tips.
4. Use a friendly and motivating tone.
5. Do not include explanations outside the JSON.

Examples:

**User:** I have tasks: Prepare presentation, Gym workout, Buy groceries  

**System JSON Output:** 
{
  "user_name": "Krishna",
  "date": "2025-10-23",
  "tasks": [
    {"task_name": "Prepare presentation", "priority": "High", "deadline": "2025-10-25", "duration": "2 hours", "category": "Work", "notes": "Focus on key points", "status": "Pending"},
    {"task_name": "Gym workout", "priority": "Medium", "deadline": "2025-10-23", "duration": "1 hour", "category": "Fitness", "notes": "", "status": "Pending"},
    {"task_name": "Buy groceries", "priority": "Medium", "deadline": "2025-10-23", "duration": "30 minutes", "category": "Personal", "notes": "", "status": "Pending"}
  ],
  "general_tips": ["Prioritize high-impact tasks first.", "Break large tasks into smaller steps.", "Use time blocks for focused work."]
}

Always generate task plans in the same JSON format.
'''

def generate_task_plan(user_name, tasks):
    if not client:
        raise Exception("OpenAI client is not available")
    
    try:
        logger.info(f"Generating task plan for user: {user_name}, tasks: {len(tasks)}")
        
        user_input = f"User: My name is {user_name}. I have tasks: {', '.join(tasks)}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        response = client.chat.completions.create(
            model=os.environ.get("MODEL", "openai/gpt-oss-20b"),
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )

        logger.info("Task plan generated successfully")
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating task plan: {str(e)}")
        raise Exception(f"Failed to generate task plan: {str(e)}")

if __name__ == "__main__":
    user_name = input("Enter your name: ")
    print("Enter your tasks one by one. Type 'done' when finished.")
    
    tasks = []
    while True:
        task_name = input("Task name: ")
        if task_name.lower() == "done":
            break
        tasks.append(task_name)
    
    print("\nYour Task Plan:\n")
    print(generate_task_plan(user_name, tasks))