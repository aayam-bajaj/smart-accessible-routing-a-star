/**
 * Main JavaScript file for Smart Accessible Routing System
 */

// Document ready function
document.addEventListener('DOMContentLoaded', function() {
    // Initialize accessibility features
    initAccessibilityFeatures();
    
    // Initialize form validation
    initFormValidation();
    
    // Initialize map features if map element exists
    if (document.getElementById('route-map')) {
        initMapFeatures();
    }
});

/**
 * Initialize accessibility features
 */
function initAccessibilityFeatures() {
    // Add skip to content link
    const skipLink = document.createElement('a');
    skipLink.href = '#main-content';
    skipLink.className = 'skip-link';
    skipLink.textContent = 'Skip to main content';
    document.body.insertBefore(skipLink, document.body.firstChild);
    
    // Ensure all links have meaningful text
    const links = document.querySelectorAll('a');
    links.forEach(link => {
        if (!link.textContent.trim() && link.querySelector('img')) {
            const img = link.querySelector('img');
            if (img && img.alt) {
                link.setAttribute('aria-label', img.alt);
            }
        }
    });
}

/**
 * Initialize form validation
 */
function initFormValidation() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            // Validate required fields
            const requiredFields = form.querySelectorAll('[required]');
            let isValid = true;
            
            requiredFields.forEach(field => {
                if (!field.value.trim()) {
                    field.classList.add('is-invalid');
                    isValid = false;
                } else {
                    field.classList.remove('is-invalid');
                }
            });
            
            if (!isValid) {
                e.preventDefault();
                showAlert('Please fill in all required fields.', 'error');
            }
        });
    });
}

/**
 * Initialize map features
 */
function initMapFeatures() {
    // This function will be expanded when we implement the map visualization
    console.log('Map features initialized');
}

/**
 * Show alert message
 * @param {string} message - Alert message
 * @param {string} type - Alert type (success, error, warning, info)
 */
function showAlert(message, type = 'info') {
    // Create alert element
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.role = 'alert';
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert alert at the top of the main content
    const mainContent = document.getElementById('main-content') || document.querySelector('main');
    if (mainContent && mainContent.firstChild) {
        mainContent.insertBefore(alert, mainContent.firstChild);
    } else if (mainContent) {
        mainContent.appendChild(alert);
    }
}

/**
 * Toggle accessibility preferences panel
 */
function toggleAccessibilityPanel() {
    const panel = document.getElementById('accessibility-panel');
    if (panel) {
        panel.classList.toggle('show');
    }
}

/**
 * Update route visualization
 * @param {Object} routeData - Route data object
 */
function updateRouteVisualization(routeData) {
    // This function will be expanded when we implement the route visualization
    console.log('Route visualization updated:', routeData);
}

/**
 * Handle route feedback submission
 */
function submitRouteFeedback() {
    const feedbackForm = document.getElementById('feedback-form');
    if (feedbackForm) {
        feedbackForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Get form data
            const formData = new FormData(feedbackForm);
            const feedbackData = Object.fromEntries(formData.entries());
            
            // Send feedback data to server
            fetch('/api/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(feedbackData),
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showAlert('Feedback submitted successfully!', 'success');
                    feedbackForm.reset();
                } else {
                    showAlert('Error submitting feedback. Please try again.', 'error');
                }
            })
            .catch(error => {
                console.error('Error submitting feedback:', error);
                showAlert('Error submitting feedback. Please try again.', 'error');
            });
        });
    }
}

// Export functions for global access
window.toggleAccessibilityPanel = toggleAccessibilityPanel;
window.submitRouteFeedback = submitRouteFeedback;