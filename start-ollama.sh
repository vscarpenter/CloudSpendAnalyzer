#!/bin/bash

# Ollama Startup Script
# Starts Ollama server and loads the gpt-oss:20b model in headless mode

set -e  # Exit on any error

# Configuration
MODEL_NAME="${MODEL_NAME:-gpt-oss:20b}"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
MAX_WAIT_TIME="${MAX_WAIT_TIME:-60}"
CHECK_INTERVAL=2
LOG_FILE="ollama-startup.log"

# Performance settings for 20B model
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_HOST="0.0.0.0:11434"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Error handling function
error_exit() {
    log "${RED}ERROR: $1${NC}"
    exit 1
}

# Check if ollama is installed
check_ollama_installed() {
    if ! command -v ollama &> /dev/null; then
        error_exit "Ollama is not installed. Please install it first: https://ollama.ai/"
    fi
    log "${GREEN}✓ Ollama found${NC}"
}

# Check if ollama server is already running
is_ollama_running() {
    curl -s "$OLLAMA_HOST/api/version" &> /dev/null
}

# Start ollama server in background
start_ollama_server() {
    if is_ollama_running; then
        log "${YELLOW}⚠ Ollama server is already running${NC}"
        return 0
    fi

    log "${BLUE}🚀 Starting Ollama server...${NC}"
    
    # Start ollama serve in background
    nohup ollama serve >> "$LOG_FILE" 2>&1 &
    OLLAMA_PID=$!
    
    # Wait for server to be ready
    local wait_time=0
    while [ $wait_time -lt $MAX_WAIT_TIME ]; do
        if is_ollama_running; then
            log "${GREEN}✓ Ollama server started successfully (PID: $OLLAMA_PID)${NC}"
            echo $OLLAMA_PID > .ollama.pid
            return 0
        fi
        
        sleep $CHECK_INTERVAL
        wait_time=$((wait_time + CHECK_INTERVAL))
        log "${BLUE}⏳ Waiting for Ollama server to start... ($wait_time/${MAX_WAIT_TIME}s)${NC}"
    done
    
    error_exit "Ollama server failed to start within $MAX_WAIT_TIME seconds"
}

# Check if model is available
is_model_available() {
    ollama list | grep -q "$MODEL_NAME"
}

# Check system resources
check_system_resources() {
    log "${BLUE}🔍 Checking system resources for $MODEL_NAME...${NC}"
    
    # Check available memory (requires at least 16GB for 20B model)
    if command -v free &> /dev/null; then
        AVAILABLE_MEM=$(free -g | awk '/^Mem:/{print $7}')
        if [ "$AVAILABLE_MEM" -lt 16 ]; then
            log "${YELLOW}⚠ Warning: Only ${AVAILABLE_MEM}GB available memory. 20B model may require 16GB+${NC}"
        fi
    elif command -v vm_stat &> /dev/null; then
        # macOS memory check
        FREE_PAGES=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
        FREE_GB=$((FREE_PAGES * 4096 / 1024 / 1024 / 1024))
        if [ "$FREE_GB" -lt 16 ]; then
            log "${YELLOW}⚠ Warning: Only ${FREE_GB}GB available memory. 20B model may require 16GB+${NC}"
        fi
    fi
    
    # Check disk space
    DISK_SPACE=$(df -h . | awk 'NR==2 {print $4}' | sed 's/G.*//')
    if [ "$DISK_SPACE" -lt 50 ]; then
        log "${YELLOW}⚠ Warning: Only ${DISK_SPACE}GB disk space available${NC}"
    fi
}

# Load the model
load_model() {
    if is_model_available; then
        log "${YELLOW}⚠ Model $MODEL_NAME is already available${NC}"
    else
        log "${BLUE}📥 Model $MODEL_NAME not found. Please install it first:${NC}"
        log "${BLUE}   ollama pull $MODEL_NAME${NC}"
        error_exit "Model $MODEL_NAME is not installed"
    fi
    
    check_system_resources
    
    log "${BLUE}🔄 Loading model $MODEL_NAME (this may take several minutes for 20B model)...${NC}"
    
    # Pre-load the model with a simple prompt to ensure it's in memory
    timeout 300 ollama run "$MODEL_NAME" "Hello" > /dev/null 2>> "$LOG_FILE" || {
        error_exit "Failed to load model $MODEL_NAME within 5 minutes"
    }
    
    log "${GREEN}✓ Model $MODEL_NAME loaded successfully${NC}"
}

# Health check function
health_check() {
    local test_prompt="Test"
    local response
    
    log "${BLUE}🏥 Running health check...${NC}"
    
    # Test model response
    response=$(timeout 30 ollama run "$MODEL_NAME" "$test_prompt" 2>/dev/null || echo "FAILED")
    
    if [ "$response" = "FAILED" ] || [ -z "$response" ]; then
        return 1
    fi
    
    log "${GREEN}✓ Model is responding correctly${NC}"
    return 0
}

# Verify everything is working
verify_setup() {
    log "${BLUE}🔍 Verifying setup...${NC}"
    
    # Test API endpoint
    if ! curl -s "$OLLAMA_HOST/api/tags" &> /dev/null; then
        error_exit "Ollama API is not responding"
    fi
    
    # Test model availability
    if ! ollama list | grep -q "$MODEL_NAME"; then
        error_exit "Model $MODEL_NAME is not available"
    fi
    
    # Run health check
    if ! health_check; then
        error_exit "Model health check failed"
    fi
    
    log "${GREEN}✅ Setup verification complete${NC}"
    log "${GREEN}🎉  backend is ready!${NC}"
    log "${BLUE}💡 Model: $MODEL_NAME is loaded and responding${NC}"
}

# Cleanup function
cleanup() {
    log "${YELLOW}🧹 Cleaning up...${NC}"
    if [ -f .ollama.pid ]; then
        local pid=$(cat .ollama.pid)
        if kill -0 "$pid" 2>/dev/null; then
            log "${BLUE}⏹ Stopping Ollama server (PID: $pid)...${NC}"
            kill "$pid"
            rm -f .ollama.pid
        fi
    fi
}

# Signal handlers
trap cleanup SIGINT SIGTERM EXIT

# Main execution
main() {
    log "${BLUE}🏠 Starting Ollama Backend${NC}"
    log "${BLUE}=================================================${NC}"
    
    # Create log file
    touch "$LOG_FILE"
    
    # Run checks and start services
    check_ollama_installed
    start_ollama_server
    sleep 5  # Give server more time to fully initialize for large models
    load_model
    verify_setup
    
    log "${GREEN}=================================================${NC}"
    log "${GREEN}🚀 Ready! Ollama is running with $MODEL_NAME loaded${NC}"
    log "${BLUE}📊 Server: $OLLAMA_HOST${NC}"
    log "${BLUE}📝 Logs: $LOG_FILE${NC}"
    log "${BLUE}🛑 To stop: Press Ctrl+C or run: ./stop-ollama.sh${NC}"
    
    # Keep script running to maintain the process
    if [ "$1" != "--daemon" ]; then
        log "${YELLOW}⌨️  Press Ctrl+C to stop the server${NC}"
        while true; do
            sleep 10
            if ! is_ollama_running; then
                error_exit "Ollama server stopped unexpectedly"
            fi
        done
    fi
}

# Handle command line arguments
case "$1" in
    --help|-h)
        echo "Ollama Startup Script"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --daemon    Run in daemon mode (background)"
        echo "  --help|-h   Show this help message"
        echo ""
        echo "Environment Variables:"
        echo "  MODEL_NAME     Model to load (default: $MODEL_NAME)"
        echo "  OLLAMA_HOST    Ollama server URL (default: $OLLAMA_HOST)"
        echo ""
        exit 0
        ;;
    --daemon)
        main --daemon
        ;;
    *)
        main
        ;;
esac
