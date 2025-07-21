document.addEventListener('DOMContentLoaded', () => {
    const params = new URLSearchParams(window.location.search);
    const result = params.get('result');
    const disease = params.get('disease');


    const inputData = JSON.parse(sessionStorage.getItem('inputData') || '{}');

    generateReport(result, disease, inputData);
});

function generateReport(result, disease, inputData) {
    const formattedDisease = disease.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());


    const predictionElement = document.getElementById('predictionResult');
    const isHighRisk = result === "1";

    predictionElement.innerHTML = `
        <div class="prediction-badge ${isHighRisk ? 'high-risk' : 'low-risk'}">
            <h2>${isHighRisk ? '⚠️ HIGH RISK' : '✅ LOW RISK'}</h2>
            <p>${formattedDisease} Prediction</p>
        </div>
    `;


    const patientInfo = document.getElementById('patientInfo');
    patientInfo.innerHTML = `
        <div class="info-item">
            <strong>Test Date:</strong> ${new Date().toLocaleDateString()}
        </div>
        <div class="info-item">
            <strong>Test Time:</strong> ${new Date().toLocaleTimeString()}
        </div>
        <div class="info-item">
            <strong>Condition Assessed:</strong> ${formattedDisease}
        </div>
        <div class="info-item">
            <strong>Prediction Model:</strong> AI-based Random Forest Classifier
        </div>
    `;


    const inputParams = document.getElementById('inputParameters');
    if (inputData.features && inputData.labels) {
        let parametersHTML = '';
        inputData.features.forEach((value, index) => {
            const label = inputData.labels[index] || `Parameter ${index + 1}`;
            const normalRange = getNormalRange(disease, index);
            const status = getParameterStatus(disease, index, value);

            parametersHTML += `
                <div class="parameter-item ${status}">
                    <div class="parameter-label">${label}</div>
                    <div class="parameter-value">${value}</div>
                    <div class="parameter-range">Normal: ${normalRange}</div>
                </div>
            `;
        });
        inputParams.innerHTML = parametersHTML;
    }


    const riskAssessment = document.getElementById('riskAssessment');
    riskAssessment.innerHTML = getRiskAssessment(disease, result);


    const recommendations = document.getElementById('recommendations');
    recommendations.innerHTML = getRecommendations(disease, result);
}

function getNormalRange(disease, index) {
    const ranges = {
        'diabetes': [
            '0-17', '70-99 mg/dL', '60-80 mmHg', '10-50 mm', '16-166 mu U/ml',
            '18.5-24.9', '0.078-2.42', '18-65 years'
        ],
        'heart_disease': [
            '25-65 years', '0-1', '0-3', '90-120 mmHg', '<200 mg/dl',
            '0-1', '0-2', '60-100 bpm', '0-1', '0-4', '1-3', '0-3', '1-3'
        ],
        'breast_cancer': Array(30).fill('Varies')
    };
    return ranges[disease]?.[index] || 'N/A';
}

function getParameterStatus(disease, index, value) {

    if (disease === 'diabetes') {
        const abnormalIndices = [1, 2, 4, 5];
        if (abnormalIndices.includes(index)) {
            if (index === 1 && value > 140) return 'abnormal';
            if (index === 2 && (value > 90 || value < 60)) return 'abnormal';
            if (index === 5 && value > 30) return 'abnormal';
        }
    }
    return 'normal';
}

function getRiskAssessment(disease, result) {
    const isHighRisk = result === "1";

    const assessments = {
        'diabetes': {
            high: `
                <div class="assessment-item high-risk">
                    <h3>High Risk Factors Identified</h3>
                    <ul>
                        <li>Elevated glucose levels or other metabolic indicators</li>
                        <li>Risk factors suggest potential diabetes development</li>
                        <li>Immediate medical consultation recommended</li>
                    </ul>
                </div>
            `,
            low: `
                <div class="assessment-item low-risk">
                    <h3>Low Risk Assessment</h3>
                    <ul>
                        <li>Current parameters within acceptable ranges</li>
                        <li>Low probability of diabetes development</li>
                        <li>Continue preventive measures and regular monitoring</li>
                    </ul>
                </div>
            `
        },
        'heart_disease': {
            high: `
                <div class="assessment-item high-risk">
                    <h3>High Risk Factors Identified</h3>
                    <ul>
                        <li>Cardiovascular risk factors present</li>
                        <li>Potential heart disease indicators detected</li>
                        <li>Urgent cardiology consultation recommended</li>
                    </ul>
                </div>
            `,
            low: `
                <div class="assessment-item low-risk">
                    <h3>Low Risk Assessment</h3>
                    <ul>
                        <li>Heart parameters within normal ranges</li>
                        <li>Low probability of heart disease</li>
                        <li>Continue heart-healthy lifestyle practices</li>
                    </ul>
                </div>
            `
        },
        'breast_cancer': {
            high: `
                <div class="assessment-item high-risk">
                    <h3>High Risk Factors Identified</h3>
                    <ul>
                        <li>Tumor characteristics suggest malignancy</li>
                        <li>Immediate oncology consultation required</li>
                        <li>Further diagnostic tests recommended</li>
                    </ul>
                </div>
            `,
            low: `
                <div class="assessment-item low-risk">
                    <h3>Low Risk Assessment</h3>
                    <ul>
                        <li>Tumor characteristics suggest benign nature</li>
                        <li>Continue regular screening and monitoring</li>
                        <li>Maintain preventive health measures</li>
                    </ul>
                </div>
            `
        }
    };

    return assessments[disease]?.[isHighRisk ? 'high' : 'low'] || 'Assessment not available';
}

function getRecommendations(disease, result) {
    const isHighRisk = result === "1";

    const recommendations = {
        'diabetes': {
            high: `
                <div class="recommendation-item urgent">
                    <h4>Immediate Actions:</h4>
                    <ul>
                        <li>Schedule appointment with endocrinologist within 1-2 weeks</li>
                        <li>Begin blood glucose monitoring</li>
                        <li>Start dietary modifications immediately</li>
                        <li>Increase physical activity gradually</li>
                    </ul>
                </div>
                <div class="recommendation-item">
                    <h4>Long-term Management:</h4>
                    <ul>
                        <li>Regular HbA1c monitoring every 3-6 months</li>
                        <li>Maintain healthy weight (BMI 18.5-24.9)</li>
                        <li>Follow diabetic diet plan</li>
                        <li>Regular exercise (150 minutes/week)</li>
                    </ul>
                </div>
            `,
            low: `
                <div class="recommendation-item">
                    <h4>Preventive Measures:</h4>
                    <ul>
                        <li>Annual diabetes screening</li>
                        <li>Maintain healthy diet and exercise routine</li>
                        <li>Monitor weight and BMI regularly</li>
                        <li>Avoid excessive sugar and processed foods</li>
                    </ul>
                </div>
            `
        },
        'heart_disease': {
            high: `
                <div class="recommendation-item urgent">
                    <h4>Immediate Actions:</h4>
                    <ul>
                        <li>Schedule cardiology consultation within 1 week</li>
                        <li>Begin cardiac monitoring if recommended</li>
                        <li>Start heart-healthy diet immediately</li>
                        <li>Avoid strenuous activities until cleared</li>
                    </ul>
                </div>
                <div class="recommendation-item">
                    <h4>Long-term Management:</h4>
                    <ul>
                        <li>Regular cardiac check-ups every 3-6 months</li>
                        <li>Monitor blood pressure and cholesterol</li>
                        <li>Medications as prescribed</li>
                        <li>Stress management and adequate sleep</li>
                    </ul>
                </div>
            `,
            low: `
                <div class="recommendation-item">
                    <h4>Preventive Measures:</h4>
                    <ul>
                        <li>Annual cardiac screening</li>
                        <li>Maintain healthy lifestyle</li>
                        <li>Regular exercise (as tolerated)</li>
                        <li>Monitor blood pressure regularly</li>
                    </ul>
                </div>
            `
        },
        'breast_cancer': {
            high: `
                <div class="recommendation-item urgent">
                    <h4>Immediate Actions:</h4>
                    <ul>
                        <li>Schedule oncology consultation within 1-2 days</li>
                        <li>Arrange for biopsy and staging tests</li>
                        <li>Begin treatment planning process</li>
                        <li>Seek emotional support resources</li>
                    </ul>
                </div>
                <div class="recommendation-item">
                    <h4>Long-term Management:</h4>
                    <ul>
                        <li>Follow oncology treatment plan</li>
                        <li>Regular monitoring and follow-ups</li>
                        <li>Maintain overall health and nutrition</li>
                        <li>Join support groups if helpful</li>
                    </ul>
                </div>
            `,
            low: `
                <div class="recommendation-item">
                    <h4>Preventive Measures:</h4>
                    <ul>
                        <li>Continue regular breast screening</li>
                        <li>Monthly self-examinations</li>
                        <li>Annual mammograms as recommended</li>
                        <li>Maintain healthy lifestyle</li>
                    </ul>
                </div>
            `
        }
    };

    return recommendations[disease]?.[isHighRisk ? 'high' : 'low'] || 'Recommendations not available';
}