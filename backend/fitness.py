from openai import OpenAI
import os
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
    logger.error(f"Failed to initialize OpenAI client in fitness module: {e}")
    client = None

def generate_fitness_plan(user_age, user_weight, user_height, user_fitness_goal, user_fitness_level, user_available_days):
    if not client:
        raise Exception("OpenAI client is not available")
    
    try:
        logger.info(f"Generating fitness plan for age: {user_age}, weight: {user_weight}")
        
        system_prompt = '''
You are a world-class Fitness Coach and Personal Trainer. Generate **personalized weekly fitness and meal plans** in **JSON format**. 
You may ask the user clarifying questions if any information is missing, but **do not finalize the plan until you have all necessary details**. 
Once all info is provided, generate **three distinct variations** of weekly plans in JSON format.

JSON Format:
{
  "user_goal": "string",
  "weekly_schedule": [
    {
      "day": "string (Monday, Tuesday, etc.)",
      "workout": [
        {
          "exercise": "string",
          "sets": "number or string",
          "reps": "number or string",
          "duration": "string (if applicable)",
          "notes": "string (optional)"
        }
      ]
    }
  ],
  "general_tips": ["array of strings for motivation, recovery, or nutrition"]
}

Rules:
- Include Tamil Nadu / Indian cuisine in meal suggestions where appropriate.
- Match workouts to the user's fitness level and goal.
- Use a friendly and motivating tone.
- **Only output JSON at the final step**, after all clarifications are gathered.
'''

        user_prompt = f"""
I am {user_age} years old, weigh {user_weight} kg, and am {user_height} cm tall. 
My fitness goal is to {user_fitness_goal}. 
My fitness level is {user_fitness_level}, and I can work out {user_available_days} days per week.
If any information is missing, make intelligent assumptions and NEVER ask follow-up questions.
Always produce final output in strict JSON format.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = client.chat.completions.create(
            model=os.environ.get("MODEL", "openai/gpt-oss-20b"),
            messages=messages,
            temperature=0.7,
            max_tokens=2500
        )

        logger.info("Fitness plan generated successfully")
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating fitness plan: {str(e)}")
        raise Exception(f"Failed to generate fitness plan: {str(e)}")

if __name__ == "__main__":
    user_age = input("Enter your age: ")
    user_weight = input("Enter your weight (in kg): ")
    user_height = input("Enter your height (in cm): ")
    user_fitness_goal = input("Enter your fitness goal (e.g., lose weight, build muscle, improve endurance): ")
    user_fitness_level = input("Enter your fitness level (beginner, intermediate, advanced): ")
    user_available_days = input("How many days per week can you work out? ")
    
    print("\nYour FITNESS PLANS (3 Variations, JSON):\n")
    print(generate_fitness_plan(user_age, user_weight, user_height, user_fitness_goal, user_fitness_level, user_available_days))