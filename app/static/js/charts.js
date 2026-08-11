/**
 * AI-NIDS Charting Helpers & Utilities
 * Provides modular chart creation and refresh helpers.
 */

console.log('[AI-NIDS] Security Dashboard Charting Engine loaded');

// Helper to create or update Chart.js instances safely
window.createNIDSChart = function(canvasId, config) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    // Destroy existing chart instance if attached
    if (window[canvasId + '_instance']) {
        window[canvasId + '_instance'].destroy();
    }
    
    if (typeof Chart !== 'undefined') {
        const instance = new Chart(ctx, config);
        window[canvasId + '_instance'] = instance;
        return instance;
    }
    return null;
};
