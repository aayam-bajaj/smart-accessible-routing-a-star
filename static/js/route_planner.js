/**
 * Route Planner JavaScript
 * Handles interactive map features, route calculation, and accessibility overlays
 */

// Global variables
let map;
let routeLayer;
let markersLayer;
let accessibilityOverlays = {};
let currentRoutes = [];
let selectedRoute = null;

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', function() {
    initializeMap();
    bindEventListeners();
    setupAccessibilityFilters();
});

/**
 * Initialize the Leaflet map
 */
function initializeMap() {
    // Initialize map centered on Mumbai (default location)
    map = L.map('route-map').setView([19.0760, 72.8777], 13);
    
    // Add OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19
    }).addTo(map);
    
    // Initialize layer groups
    routeLayer = L.layerGroup().addTo(map);
    markersLayer = L.layerGroup().addTo(map);
    
    // Initialize accessibility overlay groups
    accessibilityOverlays.ramps = L.layerGroup();
    accessibilityOverlays.elevators = L.layerGroup();
    accessibilityOverlays.widePaths = L.layerGroup();
    accessibilityOverlays.tactilePaving = L.layerGroup();
    accessibilityOverlays.obstacles = L.layerGroup();
    accessibilityOverlays.restrooms = L.layerGroup();
    
    // Add click handler for obstacle reporting
    map.on('click', function(e) {
        if (window.reportingObstacle) {
            setObstacleLocation(e.latlng);
        }
    });
    
    console.log('Map initialized successfully');
}

/**
 * Bind event listeners to UI elements
 */
function bindEventListeners() {
    // Route form submission
    document.getElementById('route-form').addEventListener('submit', handleRouteSubmission);
    
    // Map control buttons
    document.getElementById('zoom-in').addEventListener('click', () => map.zoomIn());
    document.getElementById('zoom-out').addEventListener('click', () => map.zoomOut());
    document.getElementById('current-location').addEventListener('click', getCurrentLocation);
    document.getElementById('report-obstacle').addEventListener('click', enableObstacleReporting);
    
    // Obstacle reporting
    document.getElementById('submit-obstacle').addEventListener('click', submitObstacleReport);
    
    // Feedback form
    document.getElementById('feedback-form').addEventListener('submit', handleFeedbackSubmission);
    
    // Location input fields with geocoding (simplified)
    document.getElementById('start-location').addEventListener('blur', geocodeLocation.bind(null, 'start'));
    document.getElementById('end-location').addEventListener('blur', geocodeLocation.bind(null, 'end'));
}

/**
 * Setup accessibility filter checkboxes
 */
function setupAccessibilityFilters() {
    const filterCheckboxes = {
        'show-ramps': 'ramps',
        'show-elevators': 'elevators',
        'show-wide-paths': 'widePaths',
        'show-tactile-paving': 'tactilePaving',
        'show-obstacles': 'obstacles',
        'show-restrooms': 'restrooms'
    };
    
    Object.entries(filterCheckboxes).forEach(([checkboxId, layerKey]) => {
        const checkbox = document.getElementById(checkboxId);
        checkbox.addEventListener('change', function() {
            toggleAccessibilityOverlay(layerKey, this.checked);
        });
        
        // Initialize checked overlays
        if (checkbox.checked) {
            toggleAccessibilityOverlay(layerKey, true);
        }
    });
}

/**
 * Handle route form submission
 */
async function handleRouteSubmission(e) {
    e.preventDefault();
    
    const startLocation = document.getElementById('start-location').value;
    const endLocation = document.getElementById('end-location').value;
    const routeType = document.getElementById('route-type').value;
    const mobilityMode = document.getElementById('mobility-mode').value;
    
    if (!startLocation || !endLocation) {
        showAlert('Please enter both start and destination locations.', 'warning');
        return;
    }
    
    // Show loading spinner
    showLoadingSpinner(true);
    
    try {
        const routeData = await calculateRoutes({
            start: startLocation,
            end: endLocation,
            routeType: routeType,
            mobilityMode: mobilityMode
        });
        
        displayRouteOptions(routeData);
        showLoadingSpinner(false);
        
    } catch (error) {
        console.error('Error calculating routes:', error);
        showAlert('Error calculating routes. Please try again.', 'error');
        showLoadingSpinner(false);
    }
}

/**
 * Calculate routes using the API
 */
async function calculateRoutes(params) {
    const response = await fetch('/api/routes', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            start: { lat: 19.0760, lng: 72.8777 }, // Placeholder coordinates
            end: { lat: 19.0770, lng: 72.8787 },
            user_profile: {
                mobility_mode: params.mobilityMode,
                route_preference: params.routeType
            }
        })
    });
    
    if (!response.ok) {
        throw new Error('Failed to calculate routes');
    }
    
    const data = await response.json();
    
    // Generate multiple route options for comparison
    const routes = [
        {
            id: 1,
            name: 'Recommended Route',
            ...data.route,
            color: '#007bff'
        },
        {
            id: 2,
            name: 'Most Accessible',
            distance: data.route.distance * 1.2,
            estimated_time: data.route.estimated_time * 1.1,
            accessibility_score: Math.min(data.route.accessibility_score * 1.15, 1.0),
            segments: data.route.segments,
            color: '#28a745'
        },
        {
            id: 3,
            name: 'Shortest Distance',
            distance: data.route.distance * 0.8,
            estimated_time: data.route.estimated_time * 0.9,
            accessibility_score: data.route.accessibility_score * 0.85,
            segments: data.route.segments,
            color: '#ffc107'
        }
    ];
    
    return routes;
}

/**
 * Display route options in the sidebar
 */
function displayRouteOptions(routes) {
    currentRoutes = routes;
    const resultsContainer = document.getElementById('route-results-container');
    const resultsDiv = document.getElementById('route-results');
    
    // Clear previous results
    resultsDiv.innerHTML = '';
    
    routes.forEach(route => {
        const routeElement = createRouteOptionElement(route);
        resultsDiv.appendChild(routeElement);
    });
    
    // Show results container
    resultsContainer.style.display = 'block';
    
    // Display all routes on map
    displayRoutesOnMap(routes);
    
    // Select first route by default
    if (routes.length > 0) {
        selectRoute(routes[0]);
    }
}

/**
 * Create a route option element
 */
function createRouteOptionElement(route) {
    const div = document.createElement('div');
    div.className = 'route-option';
    div.dataset.routeId = route.id;
    
    const accessibilityClass = getAccessibilityClass(route.accessibility_score);
    
    div.innerHTML = `
        <div class="d-flex justify-content-between align-items-center">
            <h6 class="mb-1">${route.name}</h6>
            <span class="badge bg-primary">${Math.round(route.distance)} m</span>
        </div>
        <div class="route-stats">
            <div class="row">
                <div class="col-6">
                    <small class="text-muted">Time:</small><br>
                    <strong>${Math.round(route.estimated_time)} min</strong>
                </div>
                <div class="col-6">
                    <small class="text-muted">Accessibility:</small><br>
                    <span class="accessibility-indicator ${accessibilityClass}"></span>
                    <strong class="${accessibilityClass}">${Math.round(route.accessibility_score * 100)}%</strong>
                </div>
            </div>
        </div>
    `;
    
    div.addEventListener('click', () => selectRoute(route));
    
    return div;
}

/**
 * Get accessibility CSS class based on score
 */
function getAccessibilityClass(score) {
    if (score >= 0.8) return 'accessibility-high';
    if (score >= 0.5) return 'accessibility-medium';
    return 'accessibility-low';
}

/**
 * Display routes on the map
 */
function displayRoutesOnMap(routes) {
    // Clear existing routes
    routeLayer.clearLayers();
    
    routes.forEach(route => {
        // Create route polyline (simplified - using sample coordinates)
        const routeCoords = generateRouteCoordinates(route);
        const polyline = L.polyline(routeCoords, {
            color: route.color,
            weight: 4,
            opacity: 0.7
        }).addTo(routeLayer);
        
        // Add route to polyline for reference
        polyline.routeData = route;
        
        // Add click handler
        polyline.on('click', () => selectRoute(route));
    });
    
    // Fit map to show all routes
    if (routeLayer.getLayers().length > 0) {
        map.fitBounds(routeLayer.getBounds(), { padding: [20, 20] });
    }
}

/**
 * Generate sample route coordinates (in real implementation, this would come from the API)
 */
function generateRouteCoordinates(route) {
    const start = [19.0760, 72.8777];
    const end = [19.0770, 72.8787];
    
    // Generate intermediate points for visualization
    const points = [start];
    const steps = 5;
    
    for (let i = 1; i < steps; i++) {
        const lat = start[0] + (end[0] - start[0]) * (i / steps) + (Math.random() - 0.5) * 0.002;
        const lng = start[1] + (end[1] - start[1]) * (i / steps) + (Math.random() - 0.5) * 0.002;
        points.push([lat, lng]);
    }
    
    points.push(end);
    return points;
}

/**
 * Select and highlight a specific route
 */
function selectRoute(route) {
    selectedRoute = route;
    
    // Update UI selection
    document.querySelectorAll('.route-option').forEach(el => {
        el.classList.remove('selected');
        if (el.dataset.routeId == route.id) {
            el.classList.add('selected');
        }
    });
    
    // Highlight route on map
    routeLayer.eachLayer(layer => {
        if (layer.routeData && layer.routeData.id === route.id) {
            layer.setStyle({ weight: 6, opacity: 1.0 });
        } else {
            layer.setStyle({ weight: 4, opacity: 0.7 });
        }
    });
    
    // Show route details
    displayRouteDetails(route);
}

/**
 * Display detailed information about the selected route
 */
function displayRouteDetails(route) {
    const detailsDiv = document.getElementById('route-details');
    const infoDiv = document.getElementById('route-info');
    
    const accessibilityClass = getAccessibilityClass(route.accessibility_score);
    
    infoDiv.innerHTML = `
        <h6>Route Summary</h6>
        <div class="row mb-3">
            <div class="col-4">
                <strong>Distance:</strong><br>
                ${Math.round(route.distance)} meters
            </div>
            <div class="col-4">
                <strong>Est. Time:</strong><br>
                ${Math.round(route.estimated_time)} minutes
            </div>
            <div class="col-4">
                <strong>Accessibility:</strong><br>
                <span class="${accessibilityClass}">${Math.round(route.accessibility_score * 100)}%</span>
            </div>
        </div>
        
        <h6>Route Segments</h6>
        ${route.segments ? route.segments.map(segment => `
            <div class="route-segment ${getAccessibilityClass(segment.accessibility_score)}">
                <div class="d-flex justify-content-between">
                    <span><strong>${Math.round(segment.distance)}m</strong></span>
                    <span class="${getAccessibilityClass(segment.accessibility_score)}">
                        ${Math.round(segment.accessibility_score * 100)}%
                    </span>
                </div>
                <small class="text-muted">Surface: ${segment.surface_type}</small>
            </div>
        `).join('') : 'No segment details available'}
    `;
    
    detailsDiv.style.display = 'block';
}

/**
 * Toggle accessibility overlay on/off
 */
function toggleAccessibilityOverlay(layerKey, show) {
    if (show) {
        if (!map.hasLayer(accessibilityOverlays[layerKey])) {
            map.addLayer(accessibilityOverlays[layerKey]);
            loadAccessibilityData(layerKey);
        }
    } else {
        if (map.hasLayer(accessibilityOverlays[layerKey])) {
            map.removeLayer(accessibilityOverlays[layerKey]);
        }
    }
}

/**
 * Load accessibility data for overlays
 */
function loadAccessibilityData(layerType) {
    // In a real implementation, this would fetch data from the API
    const sampleData = generateSampleAccessibilityData(layerType);
    
    sampleData.forEach(item => {
        const marker = createAccessibilityMarker(item, layerType);
        accessibilityOverlays[layerType].addLayer(marker);
    });
}

/**
 * Generate sample accessibility data
 */
function generateSampleAccessibilityData(layerType) {
    const bounds = map.getBounds();
    const data = [];
    const count = Math.floor(Math.random() * 10) + 5;
    
    for (let i = 0; i < count; i++) {
        const lat = bounds.getSouth() + (bounds.getNorth() - bounds.getSouth()) * Math.random();
        const lng = bounds.getWest() + (bounds.getEast() - bounds.getWest()) * Math.random();
        
        data.push({
            id: i,
            lat: lat,
            lng: lng,
            type: layerType,
            description: `Sample ${layerType} location`
        });
    }
    
    return data;
}

/**
 * Create accessibility marker
 */
function createAccessibilityMarker(item, layerType) {
    const iconMap = {
        ramps: '♿',
        elevators: '🛗',
        widePaths: '↔️',
        tactilePaving: '🟡',
        obstacles: '⚠️',
        restrooms: '🚻'
    };
    
    const colorMap = {
        ramps: '#28a745',
        elevators: '#007bff',
        widePaths: '#17a2b8',
        tactilePaving: '#ffc107',
        obstacles: '#dc3545',
        restrooms: '#6f42c1'
    };
    
    const marker = L.circleMarker([item.lat, item.lng], {
        radius: 6,
        color: colorMap[layerType],
        fillColor: colorMap[layerType],
        fillOpacity: 0.7
    });
    
    marker.bindPopup(`
        <div>
            <strong>${iconMap[layerType]} ${layerType.charAt(0).toUpperCase() + layerType.slice(1)}</strong><br>
            ${item.description}
        </div>
    `);
    
    return marker;
}

/**
 * Get current location
 */
function getCurrentLocation() {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            position => {
                const lat = position.coords.latitude;
                const lng = position.coords.longitude;
                
                map.setView([lat, lng], 15);
                
                // Add current location marker
                L.marker([lat, lng])
                    .bindPopup('Your current location')
                    .addTo(markersLayer);
                
                showAlert('Location found!', 'success');
            },
            error => {
                showAlert('Unable to get your location. Please ensure location services are enabled.', 'warning');
            }
        );
    } else {
        showAlert('Geolocation is not supported by this browser.', 'error');
    }
}

/**
 * Enable obstacle reporting mode
 */
function enableObstacleReporting() {
    window.reportingObstacle = true;
    document.body.style.cursor = 'crosshair';
    showAlert('Click on the map to report an obstacle location.', 'info');
}

/**
 * Set obstacle location from map click
 */
function setObstacleLocation(latlng) {
    document.getElementById('obstacle-lat').value = latlng.lat;
    document.getElementById('obstacle-lng').value = latlng.lng;
    document.getElementById('obstacle-location').value = `${latlng.lat.toFixed(6)}, ${latlng.lng.toFixed(6)}`;
    
    // Reset cursor and mode
    window.reportingObstacle = false;
    document.body.style.cursor = 'default';
    
    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('obstacleModal'));
    modal.show();
}

/**
 * Submit obstacle report
 */
async function submitObstacleReport() {
    const obstacleType = document.getElementById('obstacle-type').value;
    const description = document.getElementById('obstacle-description').value;
    const lat = document.getElementById('obstacle-lat').value;
    const lng = document.getElementById('obstacle-lng').value;
    
    if (!obstacleType || !description) {
        showAlert('Please fill in all required fields.', 'warning');
        return;
    }
    
    try {
        const response = await fetch('/api/obstacles', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                obstacle_type: obstacleType,
                description: description,
                location: { lat: parseFloat(lat), lng: parseFloat(lng) },
                user_id: 'current_user' // In real implementation, get from session
            })
        });
        
        if (response.ok) {
            showAlert('Obstacle reported successfully! Thank you for helping the community.', 'success');
            
            // Close modal and reset form
            bootstrap.Modal.getInstance(document.getElementById('obstacleModal')).hide();
            document.getElementById('obstacle-report-form').reset();
            
            // Refresh obstacles overlay if active
            if (document.getElementById('show-obstacles').checked) {
                accessibilityOverlays.obstacles.clearLayers();
                loadAccessibilityData('obstacles');
            }
        } else {
            throw new Error('Failed to report obstacle');
        }
    } catch (error) {
        console.error('Error reporting obstacle:', error);
        showAlert('Error reporting obstacle. Please try again.', 'error');
    }
}

/**
 * Handle feedback form submission
 */
async function handleFeedbackSubmission(e) {
    e.preventDefault();
    
    if (!selectedRoute) {
        showAlert('Please select a route first.', 'warning');
        return;
    }
    
    const rating = document.getElementById('accessibility-rating').value;
    const comments = document.getElementById('feedback-comments').value;
    
    if (!rating) {
        showAlert('Please provide an accessibility rating.', 'warning');
        return;
    }
    
    try {
        const response = await fetch('/api/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                route_id: selectedRoute.id,
                rating: parseInt(rating),
                feedback: comments,
                user_id: 'current_user' // In real implementation, get from session
            })
        });
        
        if (response.ok) {
            showAlert('Thank you for your feedback!', 'success');
            document.getElementById('feedback-form').reset();
        } else {
            throw new Error('Failed to submit feedback');
        }
    } catch (error) {
        console.error('Error submitting feedback:', error);
        showAlert('Error submitting feedback. Please try again.', 'error');
    }
}

/**
 * Simple geocoding function (placeholder)
 */
function geocodeLocation(type, event) {
    const address = event.target.value;
    if (!address) return;
    
    // In a real implementation, this would use a geocoding service
    // For now, just set some sample coordinates
    const sampleCoords = {
        lat: 19.0760 + (Math.random() - 0.5) * 0.01,
        lng: 72.8777 + (Math.random() - 0.5) * 0.01
    };
    
    document.getElementById(`${type}-lat`).value = sampleCoords.lat;
    document.getElementById(`${type}-lng`).value = sampleCoords.lng;
}

/**
 * Show loading spinner
 */
function showLoadingSpinner(show) {
    const spinner = document.querySelector('.loading-spinner');
    if (show) {
        spinner.style.display = 'block';
    } else {
        spinner.style.display = 'none';
    }
}

/**
 * Show alert message
 */
function showAlert(message, type = 'info') {
    // Use the existing showAlert function from main.js
    if (typeof window.showAlert === 'function') {
        window.showAlert(message, type);
    } else {
        // Fallback alert
        alert(message);
    }
}
