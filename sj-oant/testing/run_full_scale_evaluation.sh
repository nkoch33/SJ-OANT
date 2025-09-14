#!/bin/bash

# Full-Scale Evaluation Quick Start Script
# Run this script tomorrow to start the comprehensive evaluation

echo "🚀 SJ-OANT Full-Scale Evaluation"
echo "================================="
echo ""

# Check if API key is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  GEMINI_API_KEY not set. Please set it first:"
    echo "   export GEMINI_API_KEY='your-api-key-here'"
    echo ""
    echo "Or run: python setup_api_key.py"
    exit 1
fi

echo "✅ API key is configured"
echo ""

# Create results directory
mkdir -p results

echo "📊 Starting full-scale evaluation..."
echo "   - MultiWOZ: 200 conversations"
echo "   - SGD: 200 conversations" 
echo "   - Taskmaster: 200 conversations"
echo "   - MultiDoGO: 200 conversations"
echo ""

# Run the evaluation
python full_scale_evaluation.py

echo ""
echo "🎉 Full-scale evaluation completed!"
echo "📊 Check the results/ directory for detailed reports"
echo ""
echo "📈 Next steps:"
echo "   1. Review the comprehensive analysis report"
echo "   2. Analyze performance patterns"
echo "   3. Prepare research publication materials"
echo "   4. Compare with state-of-the-art baselines"
