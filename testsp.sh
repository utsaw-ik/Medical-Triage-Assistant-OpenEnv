#!/bin/bash

SPACE_URL="https://trishie-medical-triage-openenv.hf.space"

echo "🧪 Testing Medical Triage Space: $SPACE_URL"
echo "=============================================================="
echo ""

# Test 1: Health check
echo "1️⃣ Testing health endpoint..."
echo "Command: curl $SPACE_URL/"
curl -s "$SPACE_URL/" | head -n 5
echo ""
echo "✅ Health check complete"
echo ""

# Test 2: Tasks endpoint
echo "2️⃣ Testing /tasks endpoint..."
echo "Command: curl $SPACE_URL/tasks"
TASKS=$(curl -s "$SPACE_URL/tasks")
echo "$TASKS" | python3 -m json.tool 2>/dev/null || echo "$TASKS"
TASK_COUNT=$(echo $TASKS | grep -o '"name"' | wc -l)
echo ""
if [ "$TASK_COUNT" -ge 3 ]; then
    echo "✅ Tasks endpoint PASSED - Found $TASK_COUNT tasks"
else
    echo "❌ Tasks endpoint FAILED - Expected 3+ tasks, got $TASK_COUNT"
fi
echo ""

# Test 3: Reset endpoint
echo "3️⃣ Testing /reset endpoint..."
echo "Command: curl -X POST $SPACE_URL/reset"
RESET_RESPONSE=$(curl -s -X POST "$SPACE_URL/reset" \
  -H "Content-Type: application/json" \
  -d '{"task_type": "esi_assignment", "seed": 42}')
echo "$RESET_RESPONSE" | python3 -m json.tool 2>/dev/null | head -n 20 || echo "$RESET_RESPONSE" | head -n 20
echo ""
if echo "$RESET_RESPONSE" | grep -q "observation"; then
    echo "✅ Reset endpoint PASSED"
else
    echo "❌ Reset endpoint FAILED"
fi
echo ""

# Test 4: Step endpoint
echo "4️⃣ Testing /step endpoint..."
echo "Command: curl -X POST $SPACE_URL/step"
STEP_RESPONSE=$(curl -s -X POST "$SPACE_URL/step" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "assign_level", "content": "3"}')
echo "$STEP_RESPONSE" | python3 -m json.tool 2>/dev/null | head -n 20 || echo "$STEP_RESPONSE" | head -n 20
echo ""
if echo "$STEP_RESPONSE" | grep -q "reward"; then
    echo "✅ Step endpoint PASSED"
else
    echo "❌ Step endpoint FAILED"
fi
echo ""

echo "=============================================================="
echo "🎉 Testing Complete!"
echo ""
echo "📋 Submission Checklist:"
echo "  [ ] Space status shows 'Running' on HuggingFace"
echo "  [ ] /tasks returns 3 tasks"
echo "  [ ] /reset returns observation"
echo "  [ ] /step returns reward"
echo ""
echo "🚀 Ready to submit to hackathon!"