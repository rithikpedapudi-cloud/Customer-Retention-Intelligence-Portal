const express = require('express');
const router = express.Router();
const { exec } = require('child_process');
const axios = require('axios');
const mongoose = require('mongoose');
const User = require('../models/User');
const { GoogleGenAI } = require('@google/genai');
const { protect } = require('../utils/shared');

let ai_client = null;
if (process.env.GEMINI_API_KEY) {
    ai_client = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
}

// ─────────────────────────────────────────────
// GET /api/testing/run-unit-tests
// Triggers the local Jest test runner and captures stdout/stderr
// ─────────────────────────────────────────────
router.get('/run-unit-tests', protect, (req, res) => {
    // Run Jest unit tests programmatically
    const testCommand = 'npx jest backend/tests/auth.test.js --json';
    
    exec(testCommand, (error, stdout, stderr) => {
        // Jest outputs JSON test results when run with --json
        let jsonResults = null;
        try {
            jsonResults = JSON.parse(stdout);
        } catch (e) {
            // If parse fails, fallback to raw string output
            jsonResults = null;
        }

        const success = !error || error.code === 0;

        return res.json({
            success: true,
            testSuccess: success,
            rawOutput: stdout || stderr,
            jsonResults
        });
    });
});

// ─────────────────────────────────────────────
// GET /api/testing/run-ai-stress-test
// Synthesizes adversarial customer data using Gemini AI and queries ML model
// ─────────────────────────────────────────────
router.get('/run-ai-stress-test', protect, async (req, res) => {
    try {
        const mlEngineUrl = process.env.ML_ENGINE_URL || 'http://localhost:8000/predict';
        const singleUrl = mlEngineUrl.replace(/\/predict\/?$/, '/predict-single');

        // Fetch user context for medians
        const user = await User.findById(req.user.userId).select('medianMRR medianTenure').lean();
        const median_mrr = user?.medianMRR || 50;
        const median_tenure = user?.medianTenure || 12;

        let edgeCases = [];

        if (ai_client) {
            // Generate synthetic adversarial customer profiles via Gemini AI
            const prompt = `You are a Senior QA Engineer specialized in Machine Learning safety.
Generate exactly 4 highly-anomalous, corrupted, or extreme user profiles to stress-test a Churn Prediction Random Forest model.
Include:
1. One profile with an extreme tenure (e.g., negative or 1000 months).
2. One profile with extreme billing (e.g., $1,000,000 monthly charges or negative charges).
3. One profile with missing keys or empty string values in important variables (Contract, PaymentMethod, SeniorCitizen).
4. One adversarial profile designed to look like a high-churn user, but with mismatched categorical values.

Output strictly a JSON array containing 4 objects. Each object should have keys matching standard dataset fields:
- tenure (numeric)
- MonthlyCharges (numeric)
- TotalCharges (numeric)
- Contract (string)
- InternetService (string)
- PaymentMethod (string)
- SeniorCitizen (numeric: 0 or 1)
- customer_label (brief string indicating the test profile concept)

Do not return any markdown code blocks (like \`\`\`json) or extra text. Return ONLY the raw JSON string.`;

            const response = await ai_client.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: prompt,
            });

            let responseText = response.text.trim();
            if (responseText.startsWith('```json')) responseText = responseText.slice(7, -3);
            else if (responseText.startsWith('```')) responseText = responseText.slice(3, -3);

            edgeCases = JSON.parse(responseText.trim());
        } else {
            // Robust local fallback edge cases if Gemini API key is missing
            edgeCases = [
                { customer_label: "Extreme Negative Tenure", tenure: -12, MonthlyCharges: 70, TotalCharges: 840, Contract: "Month-to-month", InternetService: "Fiber optic", PaymentMethod: "Electronic check", SeniorCitizen: 0 },
                { customer_label: "Ultra High Billing Alert", tenure: 10, MonthlyCharges: 99999, TotalCharges: 999990, Contract: "One year", InternetService: "DSL", PaymentMethod: "Mailed check", SeniorCitizen: 1 },
                { customer_label: "Corrupted Empty String Fields", tenure: 2, MonthlyCharges: 15, TotalCharges: 30, Contract: "", InternetService: "", PaymentMethod: "", SeniorCitizen: 0 },
                { customer_label: "Mismatched Categorical Fields", tenure: 24, MonthlyCharges: 0, TotalCharges: 0, Contract: "Super-Long-Term", InternetService: "AlienTech", PaymentMethod: "Crypto", SeniorCitizen: 0 }
            ];
        }

        const stressResults = [];

        // Feed each edge case through the FastAPI ML server
        for (const testCase of edgeCases) {
            let modelOutcome = null;
            let status = 'Passed';
            let errorMessage = '';

            const record = { ...testCase };
            delete record.customer_label; // remove label for ML model

            try {
                const response = await axios.post(singleUrl, {
                    record,
                    median_mrr,
                    median_tenure
                });

                if (response.data && response.data.success) {
                    modelOutcome = response.data.data;
                } else {
                    status = 'Failed';
                    errorMessage = 'Invalid response structure from ML service';
                }
            } catch (err) {
                status = 'Failed';
                errorMessage = err.response?.data?.detail || err.message;
            }

            stressResults.push({
                label: testCase.customer_label,
                input: record,
                status,
                outcome: modelOutcome,
                error: errorMessage
            });
        }

        // Conduct a quick AI review of the model's tolerance
        let aiEvaluation = "AI Review not available (No Gemini API configuration). All local tests finished execution.";
        if (ai_client && stressResults.length > 0) {
            const reviewPrompt = `Analyze these Machine Learning stress-test results against a Customer Churn Prediction model:
${JSON.stringify(stressResults, null, 2)}

Provide a brief 3-sentence summary of the model's safety profile, noting whether it gracefully handled the anomalous inputs or crashed, and where developers should add validation filters.`;

            const aiReview = await ai_client.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: reviewPrompt,
            });
            aiEvaluation = aiReview.text;
        }

        return res.json({
            success: true,
            stressResults,
            aiEvaluation
        });

    } catch (err) {
        console.error("AI Stress Testing Error:", err);
        return res.status(500).json({ success: false, message: "AI Stress test execution failed: " + err.message });
    }
});

module.exports = router;
