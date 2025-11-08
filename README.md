# Category Hierarchy API - Railway Deployment

FastAPI application for generating and storing category hierarchies using LLM.

## Local Development

### Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
```

The API will be available at `http://localhost:8000`

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Generate category hierarchy
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"category": "Electronics"}'
```

## Docker Deployment

### Build Docker Image

```bash
docker build -t category-hierarchy-api .
```

### Run Docker Container

```bash
docker run -p 8000:8000 category-hierarchy-api
```

## Railway Deployment

### Prerequisites
- Railway account
- Railway CLI installed (`npm i -g @railway/cli`)

### Deploy to Railway

1. **Install Railway CLI** (if not already installed):
   ```bash
   npm i -g @railway/cli
   ```

2. **Login to Railway**:
   ```bash
   railway login
   ```

3. **Initialize Railway project**:
   ```bash
   railway init
   ```

4. **Deploy to Railway**:
   ```bash
   railway up
   ```

   Or link to existing project:
   ```bash
   railway link
   railway up
   ```

### Railway Configuration

The `railway.json` file is configured to:
- Use Dockerfile for building
- Run uvicorn with PORT environment variable
- Auto-restart on failure

### Environment Variables

Railway will automatically:
- Set `PORT` environment variable
- Expose the service on the assigned port

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /generate` - Generate and store category hierarchy

### Example Request

```bash
curl -X POST "https://your-app.railway.app/generate" \
     -H "Content-Type: application/json" \
     -d '{"category": "Electronics"}'
```

## Project Structure

```
zerothStep/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
├── railway.json        # Railway configuration
├── .dockerignore       # Docker ignore file
└── README.md           # This file
```
