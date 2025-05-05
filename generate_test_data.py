"""
Generate sample test data for the CrossMatch application.

This script creates sample user data for Instagram, Twitter, and Facebook
to test the user matching functionality without requiring real API access.
"""

import os
import sys
import json
import random
import datetime
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set DATABASE_URL environment variable before importing Flask app
os.environ['DATABASE_URL'] = 'postgresql://postgres:postgres@localhost:5432/crossmatch'

# Import project modules
import utils
from main import app, db, User, Platform, MatchResult

# Sample usernames for each platform
SAMPLE_USERS = {
    "instagram": [
        {"username": "photo_enthusiast", "display_name": "Photography Lover", "bio": "Capturing moments | Nature lover | Travel addict"},
        {"username": "fitness_journey", "display_name": "Fitness Journey", "bio": "Documenting my fitness journey | Healthy lifestyle | Workout tips"},
        {"username": "tech_geek_101", "display_name": "Tech Geek", "bio": "Tech reviews | Coding | Gaming | Science enthusiast"},
        {"username": "foodie_adventures", "display_name": "Food Explorer", "bio": "Food blogger | Recipe creator | Restaurant reviews"},
        {"username": "travel_diaries", "display_name": "Travel Diaries", "bio": "Exploring the world one country at a time | Adventure seeker"}
    ],
    "twitter": [
        {"username": "photo_lover", "display_name": "Photography Enthusiast", "bio": "I capture moments | Nature photography | Travel pics"},
        {"username": "fit_life_journey", "display_name": "Fitness Life", "bio": "My fitness journey | Health tips | Workout motivation"},
        {"username": "tech_reviews", "display_name": "Tech Reviews", "bio": "Technology | Coding | Gaming | Science news"},
        {"username": "food_critic", "display_name": "Food Critic", "bio": "Food reviews | Recipes | Culinary adventures"},
        {"username": "wanderlust_soul", "display_name": "Wanderlust", "bio": "Traveler | Adventure seeker | World explorer"}
    ],
    "facebook": [
        {"username": "photography.passion", "display_name": "Photography Passion", "bio": "Professional photographer | Nature and wildlife | Travel photography"},
        {"username": "fitness.lifestyle", "display_name": "Fitness Lifestyle", "bio": "Personal trainer | Fitness tips | Healthy recipes"},
        {"username": "tech.enthusiast", "display_name": "Technology Enthusiast", "bio": "Software developer | Tech reviewer | Gaming | Science"},
        {"username": "culinary.adventures", "display_name": "Culinary Adventures", "bio": "Chef | Food blogger | Restaurant reviewer | Recipe creator"},
        {"username": "global.traveler", "display_name": "Global Traveler", "bio": "Travel blogger | World explorer | Adventure seeker"}
    ]
}

# Ground truth mappings between users on different platforms
GROUND_TRUTH = {
    "instagram_twitter": {
        "photo_enthusiast": "photo_lover",
        "fitness_journey": "fit_life_journey",
        "tech_geek_101": "tech_reviews",
        "foodie_adventures": "food_critic",
        "travel_diaries": "wanderlust_soul"
    },
    "instagram_facebook": {
        "photo_enthusiast": "photography.passion",
        "fitness_journey": "fitness.lifestyle",
        "tech_geek_101": "tech.enthusiast",
        "foodie_adventures": "culinary.adventures",
        "travel_diaries": "global.traveler"
    },
    "twitter_facebook": {
        "photo_lover": "photography.passion",
        "fit_life_journey": "fitness.lifestyle",
        "tech_reviews": "tech.enthusiast",
        "food_critic": "culinary.adventures",
        "wanderlust_soul": "global.traveler"
    }
}

def generate_post_data(username, platform, num_posts=20):
    """Generate sample post data for a user."""
    posts = []
    
    # Post templates for different types of content
    templates = {
        "photo_enthusiast": [
            "Check out this amazing {subject} I photographed today! #photography #nature",
            "The lighting was perfect for this {subject} shot. #photooftheday",
            "Experimenting with {technique} photography today. Thoughts?",
            "New camera lens test with this {subject} shot! #photography"
        ],
        "fitness_journey": [
            "Completed a {workout} workout today! Feeling great! #fitness",
            "New personal record on my {exercise}! #fitnessmotivation",
            "Healthy {meal} recipe that's perfect post-workout. #healthyeating",
            "Day {day} of my fitness challenge. Progress is showing! #transformation"
        ],
        "tech_geek_101": [
            "Just reviewed the new {device}. Check it out! #tech #review",
            "My thoughts on the latest {software} update. #technology",
            "Coding project update: Building a {project} with {language}. #coding",
            "Gaming session with {game} tonight! #gaming #tech"
        ],
        "foodie_adventures": [
            "Tried this amazing {dish} at {restaurant}. Absolutely delicious! #foodie",
            "My homemade {dish} recipe. Let me know if you try it! #cooking",
            "Food tour in {location} - so many amazing flavors! #foodblogger",
            "The perfect {beverage} pairing for {dish}. #culinary #foodie"
        ],
        "travel_diaries": [
            "Exploring the beautiful {location} today! #travel #adventure",
            "The view from {landmark} was breathtaking! #travelphotography",
            "Cultural experience: Learning about {culture} traditions in {location}.",
            "Travel tip: Best way to experience {location} is {tip}. #travelblogger"
        ]
    }
    
    # Map similar usernames to the same templates
    template_mapping = {
        "photo_lover": "photo_enthusiast",
        "photography.passion": "photo_enthusiast",
        "fit_life_journey": "fitness_journey",
        "fitness.lifestyle": "fitness_journey",
        "tech_reviews": "tech_geek_101",
        "tech.enthusiast": "tech_geek_101",
        "food_critic": "foodie_adventures",
        "culinary.adventures": "foodie_adventures",
        "wanderlust_soul": "travel_diaries",
        "global.traveler": "travel_diaries"
    }
    
    # Get the right template category
    template_category = username
    if username in template_mapping:
        template_category = template_mapping[username]
    elif template_category not in templates:
        template_category = random.choice(list(templates.keys()))
    
    # Content fillers
    fillers = {
        "subject": ["sunset", "mountain", "wildlife", "cityscape", "portrait", "ocean", "forest", "architecture"],
        "technique": ["long exposure", "macro", "portrait", "night", "HDR", "black and white", "landscape"],
        "workout": ["HIIT", "strength", "cardio", "yoga", "pilates", "crossfit", "running"],
        "exercise": ["deadlift", "squat", "bench press", "5K run", "plank", "pull-up"],
        "meal": ["protein bowl", "smoothie", "salad", "meal prep", "protein pancakes"],
        "day": [str(i) for i in range(1, 31)],
        "device": ["iPhone 15", "Samsung Galaxy S24", "Google Pixel 8", "MacBook Pro", "Surface Laptop"],
        "software": ["iOS 17", "Android 14", "Windows 11", "macOS Ventura", "Ubuntu 22.04"],
        "project": ["web app", "mobile app", "game", "data visualization", "AI model"],
        "language": ["Python", "JavaScript", "React", "Flutter", "Swift", "Kotlin"],
        "game": ["Elden Ring", "Cyberpunk 2077", "Zelda", "Call of Duty", "Fortnite"],
        "dish": ["pasta carbonara", "sushi platter", "burger", "ramen", "tacos", "curry", "steak"],
        "restaurant": ["La Trattoria", "Sushi Palace", "Burger Joint", "Spice Garden", "Ocean View"],
        "location": ["Paris", "Tokyo", "New York", "Bali", "Barcelona", "Sydney", "Cape Town"],
        "landmark": ["Eiffel Tower", "Grand Canyon", "Taj Mahal", "Great Wall", "Machu Picchu"],
        "culture": ["Japanese", "Italian", "Indian", "Mexican", "Egyptian", "Thai"],
        "tip": ["taking a walking tour", "trying street food", "staying in local neighborhoods", "learning basic phrases"],
        "beverage": ["wine", "craft beer", "cocktail", "coffee", "tea"]
    }
    
    # Generate posts
    for i in range(num_posts):
        # Select a template
        if template_category in templates:
            template = random.choice(templates[template_category])
        else:
            # Fallback to a generic template
            template = "Just sharing my thoughts on {subject}. #{subject}"
        
        # Fill in the template
        for key, options in fillers.items():
            if "{" + key + "}" in template:
                template = template.replace("{" + key + "}", random.choice(options))
        
        # Create timestamp (random time in the last 30 days)
        days_ago = random.randint(0, 30)
        hours_ago = random.randint(0, 23)
        minutes_ago = random.randint(0, 59)
        timestamp = datetime.datetime.now() - datetime.timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
        
        # Create post object
        post = {
            "id": f"{platform}_{username}_post_{i}",
            "user_id": username,
            "platform": platform,
            "content": template,
            "timestamp": timestamp.isoformat(),
            "likes": random.randint(5, 500),
            "comments": random.randint(0, 50),
            "shares": random.randint(0, 30)
        }
        
        posts.append(post)
    
    return posts

def generate_platform_data():
    """Generate sample data for all platforms."""
    data_dir = Path("raw_data")
    data_dir.mkdir(exist_ok=True)
    
    all_posts = []
    
    # Generate data for each platform
    for platform, users in SAMPLE_USERS.items():
        platform_dir = data_dir / platform
        platform_dir.mkdir(exist_ok=True)
        
        platform_posts = []
        
        # Generate posts for each user
        for user_data in users:
            username = user_data["username"]
            posts = generate_post_data(username, platform)
            platform_posts.extend(posts)
            
            # Save user profile
            user_profile = {
                "id": username,
                "username": username,
                "display_name": user_data["display_name"],
                "bio": user_data["bio"],
                "follower_count": random.randint(100, 10000),
                "following_count": random.randint(50, 1000),
                "post_count": len(posts),
                "created_at": (datetime.datetime.now() - datetime.timedelta(days=random.randint(100, 1000))).isoformat()
            }
            
            with open(platform_dir / f"{username}_profile.json", "w") as f:
                json.dump(user_profile, f, indent=2)
                
            # Save user posts
            with open(platform_dir / f"{username}_posts.json", "w") as f:
                json.dump(posts, f, indent=2)
        
        # Save all platform posts
        with open(platform_dir / "all_posts.json", "w") as f:
            json.dump(platform_posts, f, indent=2)
            
        all_posts.extend(platform_posts)
    
    # Save ground truth mappings
    processed_dir = Path("processed_data")
    processed_dir.mkdir(exist_ok=True)
    
    with open(processed_dir / "ground_truth_mappings.json", "w") as f:
        json.dump(GROUND_TRUTH, f, indent=2)
    
    print(f"Generated sample data for {len(SAMPLE_USERS)} platforms")
    print(f"Total posts generated: {len(all_posts)}")
    print(f"Data saved to {data_dir} and {processed_dir}")

def add_platforms_to_user(user_id):
    """Add platform configurations to a user."""
    with app.app_context():
        # Check if user exists
        user = User.query.filter_by(id=user_id).first()
        if not user:
            print(f"User with ID {user_id} not found")
            return
        
        # Add Instagram platform
        instagram = Platform.query.filter_by(user_id=user_id, platform_name="instagram").first()
        if not instagram:
            instagram = Platform(
                user_id=user_id,
                platform_name="instagram",
                client_id="1615842599133380",
                client_secret="234e16b1a5c49ec3d04078c8f8c6140",
                access_token="sample_access_token"
            )
            db.session.add(instagram)
        
        # Add Twitter platform
        twitter = Platform.query.filter_by(user_id=user_id, platform_name="twitter").first()
        if not twitter:
            twitter = Platform(
                user_id=user_id,
                platform_name="twitter",
                api_key="sample_api_key",
                api_secret="sample_api_secret",
                bearer_token="sample_bearer_token"
            )
            db.session.add(twitter)
        
        # Add Facebook platform
        facebook = Platform.query.filter_by(user_id=user_id, platform_name="facebook").first()
        if not facebook:
            facebook = Platform(
                user_id=user_id,
                platform_name="facebook",
                client_id="sample_app_id",
                client_secret="sample_app_secret",
                access_token="sample_access_token"
            )
            db.session.add(facebook)
        
        db.session.commit()
        print(f"Added platform configurations for user {user.username} (ID: {user_id})")

def add_sample_match_results(user_id):
    """Add sample match results for a user."""
    with app.app_context():
        # Check if user exists
        user = User.query.filter_by(id=user_id).first()
        if not user:
            print(f"User with ID {user_id} not found")
            return
        
        # Clear existing match results
        MatchResult.query.filter_by(user_id=user_id).delete()
        
        # Add match results based on ground truth
        for platform_pair, mappings in GROUND_TRUTH.items():
            platforms = platform_pair.split("_")
            platform1, platform2 = platforms[0], platforms[1]
            
            for user1, user2 in mappings.items():
                # Create match result with high confidence
                match = MatchResult(
                    user_id=user_id,
                    platform1=platform1,
                    platform1_user_id=user1,
                    platform2=platform2,
                    platform2_user_id=user2,
                    confidence_score=random.uniform(0.85, 0.99),
                    timestamp=datetime.datetime.now()
                )
                db.session.add(match)
                
                # Create a few false matches with lower confidence
                for _ in range(2):
                    # Get a random user from platform2 that is not the correct match
                    false_users = [u["username"] for u in SAMPLE_USERS[platform2] if u["username"] != user2]
                    if false_users:
                        false_match = MatchResult(
                            user_id=user_id,
                            platform1=platform1,
                            platform1_user_id=user1,
                            platform2=platform2,
                            platform2_user_id=random.choice(false_users),
                            confidence_score=random.uniform(0.3, 0.7),
                            timestamp=datetime.datetime.now()
                        )
                        db.session.add(false_match)
        
        db.session.commit()
        print(f"Added sample match results for user {user.username} (ID: {user_id})")

if __name__ == "__main__":
    # Generate platform data
    generate_platform_data()
    
    # Add platforms and match results to admin user (ID: 1)
    add_platforms_to_user(1)
    add_sample_match_results(1)
    
    print("Sample test data generation complete!")
