#!/bin/bash

# Ollama Status Script
# Checks the status of Ollama server and loaded models

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODEL_NAME="${MODEL_NAME:-gpt-oss:20b}"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

# Logging function
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "${BLUE}📊 Ollama Status Check${NC}"
log "${BLUE}=========================================${NC}"

# Check if ollama server is running
if curl -s "$OLLAMA_HOST/api/version" &> /dev/null; then
    VERSION=$(curl -s "$OLLAMA_HOST/api/version" | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
    log "${GREEN}✓ Ollama server is running (version: $VERSION)${NC}"
    log "${BLUE}  Host: $OLLAMA_HOST${NC}"
    
    # Check PID file
    if [ -f .ollama.pid ]; then
        PID=$(cat .ollama.pid)
        if kill -0 "$PID" 2>/dev/null; then
            log "${GREEN}✓ Process ID: $PID${NC}"
        else
            log "${YELLOW}⚠ PID file exists but process not running${NC}"
        fi
    else
        log "${YELLOW}⚠ No PID file found${NC}"
    fi
    
    # List loaded models
    log "${BLUE}📋 Available models:${NC}"
    ollama list 2>/dev/null | tail -n +2 | while read -r line; do
        if [ -n "$line" ]; then
            model_name=$(echo "$line" | awk '{print $1}')
            if [ "$model_name" = "$MODEL_NAME" ]; then
                log "${GREEN}  ✓ $line${NC}"
            else
                log "${BLUE}    $line${NC}"
            fi
        fi
    done
    
    # Check if target model is loaded
    if ollama list | grep -q "$MODEL_NAME"; then
        log "${GREEN}✓ Target model $MODEL_NAME is available${NC}"
        
        # Test model response
        log "${BLUE}🧪 Testing model response...${NC}"
        response=$(timeout 10 ollama run "$MODEL_NAME" "Hello" 2>/dev/null || echo "TIMEOUT")
        
        if [ "$response" = "TIMEOUT" ]; then
            log "${YELLOW}⚠ Model response test timed out${NC}"
        elif [ -n "$response" ]; then
            log "${GREEN}✓ Model is responding correctly${NC}"
            log "${BLUE}  Response preview: $(echo "$response" | head -c 50)...${NC}"
        else
            log "${RED}✗ Model failed to respond${NC}"
        fi
    else
        log "${RED}✗ Target model $MODEL_NAME is not available${NC}"
    fi
    
    # Show resource usage
    if command -v ps &> /dev/null; then
        OLLAMA_PIDS=$(pgrep -f "ollama" || true)
        if [ -n "$OLLAMA_PIDS" ]; then
            log "${BLUE}💾 Resource usage:${NC}"
            ps -p "$OLLAMA_PIDS" -o pid,pcpu,pmem,rss,vsz,comm --no-headers | while read -r line; do
                log "${BLUE}  $line${NC}"
            done
        fi
    fi
    
else
    log "${RED}✗ Ollama server is not running${NC}"
    log "${BLUE}💡 Start with: ./start-ollama.sh${NC}"
fi

log "${BLUE}=========================================${NC}"
