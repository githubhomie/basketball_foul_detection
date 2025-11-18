#!/bin/bash
# Quick setup script for annotation tool

echo "🏀 NBA Foul Annotator Setup"
echo "============================"
echo ""

# Check if in correct directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Run this script from annotation_tool/ directory"
    exit 1
fi

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check .env file
if [ -f "../.env" ]; then
    echo "✓ Found .env file with AWS credentials"
else
    echo "⚠️  Warning: No .env file found in parent directory"
    echo "   Create foul_data_aws/.env with:"
    echo "   AWS_ACCESS_KEY_ID=..."
    echo "   AWS_SECRET_ACCESS_KEY=..."
    echo "   AWS_REGION=us-east-2"
    echo "   S3_BUCKET_NAME=nba-foul-dataset-oh"
fi

# Create annotations directory
echo ""
echo "📁 Creating annotations directory..."
mkdir -p ../data/annotations

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To start annotating:"
echo "   streamlit run app.py"
echo ""
