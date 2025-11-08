#main Cat-> Sub Cat->Sub Sub Cat->Product Types->Product List

# pip install openai supabase fastapi uvicorn python-dotenv

from openai import OpenAI 
from supabase import create_client, Client
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
import re
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client
HUGGINGFACE_BASE_URL = os.getenv("HUGGINGFACE_BASE_URL")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not HUGGINGFACE_BASE_URL or not HUGGINGFACE_API_KEY:
	raise ValueError("HUGGINGFACE_BASE_URL and HUGGINGFACE_API_KEY must be set in environment variables")

client = OpenAI(
	base_url=HUGGINGFACE_BASE_URL,
	api_key=HUGGINGFACE_API_KEY
)

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
	raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize FastAPI app
app = FastAPI(title="Category Hierarchy API", version="1.0.0")

# Request/Response models
class CategoryRequest(BaseModel):
	category: str

class CategoryResponse(BaseModel):
	success: bool
	message: str
	category_id: int = None
	category_name: str = None
	sub_categories_count: int = None
	sub_sub_categories_count: int = None
	product_types_count: int = None
	summary: Dict[str, Any] = None


def get_category_hierarchy(main_category: str) -> dict:
	"""
	Generate a hierarchical category structure for a given main category.
	
	Args:
		main_category (str): The main category name (e.g., "Electronics", "Fashion", "Home & Garden")
	
	Returns:
		dict: A dictionary containing the category hierarchy in the format:
			{
				"category": "Main Category",
				"sub_categories": [
					{
						"sub_category": "Sub Category Name",
						"sub_subcategories": [
							{
								"name": "Sub Sub Category Name",
								"ProductTypes": ["Product Type 1", "Product Type 2", ...]
							},
							...
						]
					},
					...
				]
			}
	"""
	
	# Construct the prompt to get structured JSON output
	prompt = f"""You are an expert e-commerce shopping agent. When given a category name, you should provide a comprehensive hierarchical structure of sub-categories, sub-sub-categories, and product types using ONLY GENERIC CLASSIFICATIONS.

CRITICAL RULES - READ CAREFULLY:
1. NO BRAND NAMES in sub-categories or sub-sub-categories - these must be generic product groups and technology types
2. ProductTypes should be GENERIC classifications based on:
   - Use case (e.g., "Home Office Printers", "Gaming Laptops", "Business Desktops")
   - Price range (e.g., "Budget Smartphones", "High-End DSLRs", "Mid-Range Tablets")
   - Size/Variant (e.g., "Large Tablets", "Small Form Factor Computers", "Portable Printers")
   - Technology/Features (e.g., "Monochrome Laser Printers", "Color Laser Printers", "Wireless Printers")
   - Target market (e.g., "Entry-Level DSLRs", "Professional Cameras", "Consumer Electronics")
3. ONLY include brand names in ProductTypes when the brand name itself represents a distinct product category (e.g., "GoPro" for action cameras, "PlayStation 5" for gaming consoles - but these should be rare exceptions)
4. DO NOT use brand names like iPhone, Samsung Galaxy, MacBook, Dell XPS, HP, Lenovo, etc. in ProductTypes
5. Focus on generic, descriptive classifications that apply to all brands

Example of CORRECT structure (notice each sub-category has MANY sub-sub-categories):
{{
  "category": "Electronics",
  "sub_categories": [
    {{
      "sub_category": "Printers",
      "sub_subcategories": [
        {{ "name": "Inkjet Printers", "ProductTypes": ["Home Office Printers", "Photo Printers", "Large Format Printers", "Portable Printers"] }},
        {{ "name": "Laser Printers", "ProductTypes": ["Monochrome Laser Printers", "Color Laser Printers", "Wide Format Laser Printers"] }},
        {{ "name": "3D Printers", "ProductTypes": ["Fused Deposition Modeling (FDM) Printers", "Stereolithography (SLA) Printers", "Selective Laser Sintering (SLS) Printers"] }},
        {{ "name": "Label Printers", "ProductTypes": ["Direct Thermal Label Printers", "Thermal Transfer Label Printers", "Impact Label Printers"] }},
        {{ "name": "Dot Matrix Printers", "ProductTypes": ["Standard Dot Matrix Printers", "High-Speed Dot Matrix Printers", "Wide-Format Dot Matrix Printers"] }},
        {{ "name": "Plotter Printers", "ProductTypes": ["Flatbed Plotter Printers", "Roll-Based Plotter Printers", "Large Format Plotter Printers"] }},
        {{ "name": "Mobile Printers", "ProductTypes": ["Mobile Inkjet Printers", "Mobile Laser Printers", "Mobile 3D Printers"] }},
        {{ "name": "Specialty Printers", "ProductTypes": ["Wireless Printers", "All-in-One Printers", "Smart Printers"] }}
      ]
    }},
    {{
      "sub_category": "Laptops",
      "sub_subcategories": [
        {{ "name": "Ultraportable Laptops", "ProductTypes": ["Lightweight Ultraportables", "High-Performance Ultraportables", "Budget Ultraportables"] }},
        {{ "name": "Thin and Light Laptops", "ProductTypes": ["Business Thin and Lights", "Gaming Thin and Lights", "Home Thin and Lights"] }},
        {{ "name": "Gaming Laptops", "ProductTypes": ["High-Performance Gaming Laptops", "Budget Gaming Laptops", "Esports Gaming Laptops"] }},
        {{ "name": "Chromebooks", "ProductTypes": ["Home Chromebooks", "Business Chromebooks", "Budget Chromebooks"] }}
      ]
    }},
    {{
      "sub_category": "Smartphones",
      "sub_subcategories": [
        {{ "name": "Flagship Smartphones", "ProductTypes": ["High-End Flagships", "Mid-Range Flagships", "Budget Flagships"] }},
        {{ "name": "Mid-Range Smartphones", "ProductTypes": ["Home Mid-Ranges", "Business Mid-Ranges", "Gaming Mid-Ranges"] }},
        {{ "name": "Budget Smartphones", "ProductTypes": ["Entry-Level Budget Phones", "Mid-Range Budget Phones", "High-End Budget Phones"] }},
        {{ "name": "Feature Phones", "ProductTypes": ["Basic Feature Phones", "Advanced Feature Phones", "Rugged Feature Phones"] }}
      ]
    }}
  ]
}}

Example of INCORRECT structure (DO NOT DO THIS):
{{
  "sub_category": "Laptops",
  "sub_subcategories": [
    {{
      "name": "Laptops",
      "ProductTypes": ["MacBook", "Dell XPS", "Lenovo IdeaPad", "HP Pavilion"]
    }}
  ]
}}
NOTE: The above is WRONG because ProductTypes contain brand names. Use generic classifications instead!

Your task is to analyze the category "{main_category}" and provide a complete breakdown following these rules:

Requirements:
1. Provide at least 10-15 major sub-categories for the given category
2. CRITICAL: For each sub-category, provide 4-8 relevant sub-sub-categories (generic technology/types). DO NOT stop at just 2-3 sub-sub-categories - be comprehensive and include ALL major types/variants that exist in that sub-category. For example, "Printers" should have 8 sub-sub-categories (Inkjet, Laser, 3D, Label, Dot Matrix, Plotter, Mobile, Specialty), not just 2.
3. For each sub-sub-category, provide 2-5 product types (generic use cases, features, price ranges, sizes - NO brand names)
4. Be comprehensive and cover all major product areas within the category
5. Think about all the different types, technologies, and variants that exist in each sub-category - don't limit yourself to just a few
6. Use proper naming conventions (capitalize appropriately)
7. Return ONLY valid JSON, no additional text or explanations
8. REMEMBER: Generic classifications only - avoid brand names! And provide MANY sub-sub-categories for each sub-category!

Category to analyze: {main_category}

Return the JSON structure now:"""

	try:
		# Call the LLM
		chat_completion = client.chat.completions.create(
			model = 'Qwen/Qwen3-1.7B',
			messages = [
				{
					'role': 'user',
					'content': prompt
				}
			],
			stream = True,
			max_tokens = 30000,
			temperature = 0.7
		)
		
		# Collect the streaming response
		response_text = ""
		for message in chat_completion:
			if message.choices[0].delta.content:
				response_text += message.choices[0].delta.content
		
		# Save raw response for debugging (only if writable)
		try:
			with open("raw_llm_response.txt", "w", encoding="utf-8") as f:
				f.write(response_text)
		except (IOError, OSError):
			# File system might be read-only (e.g., Vercel serverless)
			pass
		
		# Clean and extract JSON from the response
		# First, remove all reasoning/thinking tags and their content
		response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL | re.IGNORECASE)
		response_text = re.sub(r'<reasoning>.*?</reasoning>', '', response_text, flags=re.DOTALL | re.IGNORECASE)
		response_text = re.sub(r'<thinking>.*?</thinking>', '', response_text, flags=re.DOTALL | re.IGNORECASE)
		response_text = re.sub(r'<thought>.*?</thought>', '', response_text, flags=re.DOTALL | re.IGNORECASE)
		
		# Remove any markdown code blocks if present
		response_text = response_text.strip()
		if response_text.startswith("```json"):
			response_text = response_text[7:]
		if response_text.startswith("```"):
			response_text = response_text[3:]
		if response_text.endswith("```"):
			response_text = response_text[:-3]
		response_text = response_text.strip()
		
		# Find the first JSON object start - skip any text before it
		first_brace = response_text.find('{')
		if first_brace > 0:
			# There's text before the JSON, remove it
			response_text = response_text[first_brace:]
		
		# Function to clean JSON text
		def clean_json_text(text):
			"""Clean JSON text by removing common issues."""
			# Remove single-line comments
			text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
			# Remove multi-line comments
			text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
			# Remove trailing commas before closing braces/brackets
			text = re.sub(r',(\s*[}\]])', r'\1', text)
			return text.strip()
		
		# Function to extract first complete JSON object
		def extract_json_object(text):
			"""Extract the first complete JSON object from text, handling nested braces."""
			start_idx = text.find('{')
			if start_idx == -1:
				return None
			
			brace_count = 0
			in_string = False
			escape_next = False
			
			for i in range(start_idx, len(text)):
				char = text[i]
				
				if escape_next:
					escape_next = False
					continue
				
				if char == '\\':
					escape_next = True
					continue
				
				if char == '"' and not escape_next:
					in_string = not in_string
					continue
				
				if not in_string:
					if char == '{':
						brace_count += 1
					elif char == '}':
						brace_count -= 1
						if brace_count == 0:
							# Found complete JSON object
							return text[start_idx:i+1]
			
			return None
		
		# Try multiple parsing strategies
		parsing_strategies = [
			# Strategy 1: Parse as-is
			lambda: json.loads(response_text),
			# Strategy 2: Clean and parse
			lambda: json.loads(clean_json_text(response_text)),
			# Strategy 3: Extract JSON object and parse
			lambda: json.loads(extract_json_object(response_text) or response_text),
			# Strategy 4: Extract JSON object, clean, and parse
			lambda: json.loads(clean_json_text(extract_json_object(response_text) or response_text)),
		]
		
		last_error = None
		for i, strategy in enumerate(parsing_strategies, 1):
			try:
				result = strategy()
				
				# Validate the structure
				if not isinstance(result, dict):
					raise ValueError("Result is not a dictionary")
				
				if "category" not in result:
					result["category"] = main_category
				if "sub_categories" not in result:
					raise ValueError("Response missing 'sub_categories' field")
				
				print(f"Successfully parsed JSON using strategy {i}")
				return result
				
			except (json.JSONDecodeError, ValueError, AttributeError) as e:
				last_error = e
				continue
		
		# If all strategies failed, try to manually extract and fix JSON
		json_text = extract_json_object(response_text)
		if json_text:
			# Try to fix common JSON issues
			json_text = clean_json_text(json_text)
			
			# Try parsing one more time
			try:
				result = json.loads(json_text)
				if "category" not in result:
					result["category"] = main_category
				if "sub_categories" not in result:
					raise ValueError("Response missing 'sub_categories' field")
				print("Successfully parsed JSON after manual extraction and cleaning")
				return result
			except Exception as e:
				last_error = e
		
		# If everything failed, raise error with helpful message
		error_msg = f"Failed to parse JSON response. Last error: {str(last_error)}"
		error_msg += f"\n\nFirst 500 chars of response:\n{response_text[:500]}"
		error_msg += f"\n\nRaw response saved to: raw_llm_response.txt"
		raise ValueError(error_msg)
	
	except Exception as e:
		raise Exception(f"Error generating category hierarchy: {str(e)}")


def store_category_hierarchy_in_supabase(hierarchy: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Store category hierarchy in Supabase database.
	
	Args:
		hierarchy (dict): The category hierarchy dictionary from get_category_hierarchy()
	
	Returns:
		dict: Summary of stored data with IDs
	"""
	try:
		category_name = hierarchy.get("category", "")
		sub_categories = hierarchy.get("sub_categories", [])
		
		if not category_name:
			raise ValueError("Category name is required")
		
		# Insert or get main category
		category_result = supabase.table("z_categories").select("id").eq("name", category_name).execute()
		
		if category_result.data:
			category_id = category_result.data[0]["id"]
			print(f"Category '{category_name}' already exists with ID: {category_id}")
		else:
			category_result = supabase.table("z_categories").insert({"name": category_name}).execute()
			category_id = category_result.data[0]["id"]
			print(f"Created category '{category_name}' with ID: {category_id}")
		
		summary = {
			"category_id": category_id,
			"category_name": category_name,
			"sub_categories": []
		}
		
		# Process each sub-category
		for sub_cat in sub_categories:
			sub_cat_name = sub_cat.get("sub_category", "")
			sub_subcategories = sub_cat.get("sub_subcategories", [])
			
			if not sub_cat_name:
				continue
			
			# Insert or get sub-category
			sub_cat_result = supabase.table("z_sub_categories").select("id").eq("category_id", category_id).eq("name", sub_cat_name).execute()
			
			if sub_cat_result.data:
				sub_cat_id = sub_cat_result.data[0]["id"]
			else:
				sub_cat_result = supabase.table("z_sub_categories").insert({
					"category_id": category_id,
					"name": sub_cat_name
				}).execute()
				sub_cat_id = sub_cat_result.data[0]["id"]
			
			sub_cat_summary = {
				"sub_category_id": sub_cat_id,
				"sub_category_name": sub_cat_name,
				"sub_subcategories": []
			}
			
			# Process each sub-sub-category
			for sub_sub_cat in sub_subcategories:
				sub_sub_cat_name = sub_sub_cat.get("name", "")
				product_types = sub_sub_cat.get("ProductTypes", [])
				
				if not sub_sub_cat_name:
					continue
				
				# Insert or get sub-sub-category
				sub_sub_cat_result = supabase.table("z_sub_sub_categories").select("id").eq("sub_category_id", sub_cat_id).eq("name", sub_sub_cat_name).execute()
				
				if sub_sub_cat_result.data:
					sub_sub_cat_id = sub_sub_cat_result.data[0]["id"]
				else:
					sub_sub_cat_result = supabase.table("z_sub_sub_categories").insert({
						"sub_category_id": sub_cat_id,
						"name": sub_sub_cat_name
					}).execute()
					sub_sub_cat_id = sub_sub_cat_result.data[0]["id"]
				
				sub_sub_cat_summary = {
					"sub_sub_category_id": sub_sub_cat_id,
					"sub_sub_category_name": sub_sub_cat_name,
					"product_types": []
				}
				
				# Process each product type
				for product_type_name in product_types:
					if not product_type_name:
						continue
					
					# Insert or get product type
					product_type_result = supabase.table("z_product_types").select("id").eq("sub_sub_category_id", sub_sub_cat_id).eq("name", product_type_name).execute()
					
					if product_type_result.data:
						product_type_id = product_type_result.data[0]["id"]
					else:
						product_type_result = supabase.table("z_product_types").insert({
							"sub_sub_category_id": sub_sub_cat_id,
							"name": product_type_name
						}).execute()
						product_type_id = product_type_result.data[0]["id"]
					
					sub_sub_cat_summary["product_types"].append({
						"product_type_id": product_type_id,
						"product_type_name": product_type_name
					})
				
				sub_cat_summary["sub_subcategories"].append(sub_sub_cat_summary)
			
			summary["sub_categories"].append(sub_cat_summary)
		
		return summary
		
	except Exception as e:
		raise Exception(f"Error storing category hierarchy in Supabase: {str(e)}")


# API Endpoints
@app.get("/")
async def root():
	"""Root endpoint - API information"""
	return {
		"message": "Category Hierarchy API",
		"version": "1.0.0",
		"endpoints": {
			"POST /generate": "Generate and store category hierarchy",
			"GET /health": "Health check"
		}
	}


@app.get("/health")
async def health_check():
	"""Health check endpoint"""
	return {"status": "healthy", "service": "Category Hierarchy API"}


@app.post("/generate", response_model=CategoryResponse)
async def generate_category_hierarchy(request: CategoryRequest):
	"""
	Generate category hierarchy and store in Supabase.
	
	Args:
		request: CategoryRequest with category name
	
	Returns:
		CategoryResponse with saved status and summary
	"""
	try:
		category_name = request.category.strip()
		
		if not category_name:
			raise HTTPException(status_code=400, detail="Category name cannot be empty")
		
		# Generate category hierarchy using LLM
		print(f"Generating category hierarchy for: {category_name}")
		result = get_category_hierarchy(category_name)
		
		# Store in Supabase
		print(f"Storing category hierarchy in Supabase...")
		summary = store_category_hierarchy_in_supabase(result)
		
		# Count total items stored
		total_sub_sub = sum(len(sub['sub_subcategories']) for sub in summary['sub_categories'])
		total_product_types = sum(
			len(sub_sub['product_types']) 
			for sub in summary['sub_categories'] 
			for sub_sub in sub['sub_subcategories']
		)
		
		# Return success response
		return CategoryResponse(
			success=True,
			message=f"Successfully generated and stored category hierarchy for '{category_name}'",
			category_id=summary['category_id'],
			category_name=summary['category_name'],
			sub_categories_count=len(summary['sub_categories']),
			sub_sub_categories_count=total_sub_sub,
			product_types_count=total_product_types,
			summary=summary
		)
		
	except HTTPException:
		raise
	except Exception as e:
		error_message = str(e)
		print(f"Error: {error_message}")
		raise HTTPException(
			status_code=500,
			detail=f"Error generating category hierarchy: {error_message}"
		)


# Example usage (for testing without API)
if __name__ == "__main__":
	import uvicorn
	import os
	
	# Get port from environment variable (Railway provides this) or default to 8000
	port = int(os.environ.get("PORT", 8000))
	
	# Run the API server
	# Usage: python main.py
	# Then make POST request to http://localhost:8000/generate
	# Example: curl -X POST "http://localhost:8000/generate" -H "Content-Type: application/json" -d '{"category": "Electronics"}'
	
	uvicorn.run(app, host="0.0.0.0", port=port)