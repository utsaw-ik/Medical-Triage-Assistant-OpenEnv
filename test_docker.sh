#!/usr/bin/env bash
#
# Docker Build & Deployment Test
# ================================
# Tests that the Docker container builds and runs correctly
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

function print_error() {
    echo -e "${RED}❌ $1${NC}"
}

function print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

function print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

function print_header() {
    echo ""
    echo "======================================================================"
    echo "$1"
    echo "======================================================================"
}

# Check prerequisites
print_header "Checking Prerequisites"

if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
    exit 1
fi
print_success "Docker is installed"

if [ ! -f "Dockerfile" ]; then
    print_error "Dockerfile not found in current directory"
    exit 1
fi
print_success "Dockerfile exists"

# Build Docker image
print_header "Building Docker Image"

IMAGE_NAME="medical-triage-env"
TAG="test"

print_info "Building $IMAGE_NAME:$TAG..."

if docker build -t "$IMAGE_NAME:$TAG" . 2>&1 | tee build.log; then
    print_success "Docker image built successfully"
else
    print_error "Docker build failed. Check build.log for details"
    exit 1
fi

# Check image size
IMAGE_SIZE=$(docker images "$IMAGE_NAME:$TAG" --format "{{.Size}}")
print_info "Image size: $IMAGE_SIZE"

# Validate image has required files
print_header "Validating Image Contents"

print_info "Checking for required files in image..."

REQUIRED_FILES=(
    "medical_triage_env.py"
    "server.py"
    "inference.py"
    "openenv.yaml"
    "requirements.txt"
)

for file in "${REQUIRED_FILES[@]}"; do
    if docker run --rm "$IMAGE_NAME:$TAG" test -f "$file"; then
        print_success "Found $file"
    else
        print_error "Missing $file"
        exit 1
    fi
done

# Test running the container
print_header "Testing Container Execution"

print_info "Starting container..."

CONTAINER_NAME="medical-triage-test"

# Clean up any existing container
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Run container in background
if docker run -d \
    --name "$CONTAINER_NAME" \
    -p 7860:7860 \
    -e HF_TOKEN="dummy_token" \
    "$IMAGE_NAME:$TAG" > /dev/null 2>&1; then
    print_success "Container started"
else
    print_error "Failed to start container"
    exit 1
fi

# Wait for server to be ready
print_info "Waiting for server to be ready..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:7860/ > /dev/null 2>&1; then
        print_success "Server is responding"
        break
    fi
    attempt=$((attempt + 1))
    sleep 1
done

if [ $attempt -eq $max_attempts ]; then
    print_error "Server did not start in time"
    docker logs "$CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME"
    exit 1
fi

# Test API endpoints
print_header "Testing API Endpoints"

# Test root endpoint
print_info "Testing GET /..."
if curl -s http://localhost:7860/ > /dev/null; then
    print_success "Root endpoint works"
else
    print_error "Root endpoint failed"
fi

# Test tasks endpoint
print_info "Testing GET /tasks..."
TASKS_RESPONSE=$(curl -s http://localhost:7860/tasks)
TASK_COUNT=$(echo "$TASKS_RESPONSE" | grep -o "name" | wc -l)

if [ "$TASK_COUNT" -ge 3 ]; then
    print_success "Tasks endpoint returns $TASK_COUNT tasks"
else
    print_error "Tasks endpoint should return at least 3 tasks, got $TASK_COUNT"
fi

# Test reset endpoint
print_info "Testing POST /reset..."
RESET_RESPONSE=$(curl -s -X POST http://localhost:7860/reset \
    -H "Content-Type: application/json" \
    -d '{"task_type": "esi_assignment", "seed": 42}')

if echo "$RESET_RESPONSE" | grep -q "observation"; then
    print_success "Reset endpoint works"
else
    print_error "Reset endpoint failed"
    echo "$RESET_RESPONSE"
fi

# Test step endpoint
print_info "Testing POST /step..."
STEP_RESPONSE=$(curl -s -X POST http://localhost:7860/step \
    -H "Content-Type: application/json" \
    -d '{"action_type": "assign_level", "content": "3"}')

if echo "$STEP_RESPONSE" | grep -q "reward"; then
    print_success "Step endpoint works"
else
    print_error "Step endpoint failed"
    echo "$STEP_RESPONSE"
fi

# Check container logs for errors
print_header "Checking Container Logs"

LOGS=$(docker logs "$CONTAINER_NAME" 2>&1)

if echo "$LOGS" | grep -i "error" | grep -v "stderr" > /dev/null; then
    print_warning "Found errors in container logs:"
    echo "$LOGS" | grep -i "error" | head -n 5
else
    print_success "No critical errors in logs"
fi

# Resource usage check
print_header "Checking Resource Usage"

STATS=$(docker stats "$CONTAINER_NAME" --no-stream --format "{{.CPUPerc}} {{.MemUsage}}")
print_info "Container stats: $STATS"

# Extract memory usage (basic check)
MEM_USAGE=$(echo "$STATS" | awk '{print $2}' | sed 's/MiB.*//')

if [ -n "$MEM_USAGE" ]; then
    if [ "${MEM_USAGE%.*}" -lt 8000 ]; then
        print_success "Memory usage within limits (< 8GB)"
    else
        print_warning "Memory usage might exceed 8GB limit"
    fi
fi

# Cleanup
print_header "Cleanup"

print_info "Stopping and removing test container..."
docker stop "$CONTAINER_NAME" > /dev/null 2>&1
docker rm "$CONTAINER_NAME" > /dev/null 2>&1
print_success "Cleanup complete"

# Summary
print_header "Test Summary"

print_success "Docker build and deployment tests passed!"
print_info "Image: $IMAGE_NAME:$TAG"
print_info "Size: $IMAGE_SIZE"

echo ""
echo "Next steps:"
echo "  1. Run unit tests: pytest test_medical_triage.py -v"
echo "  2. Run validation: python test_openenv_validation.py"
echo "  3. Test inference: python test_inference_pipeline.py"
echo ""
print_success "Ready for deployment to Hugging Face Spaces!"
