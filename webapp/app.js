// Configuration
const API_URL = "http://127.0.0.1:8000/api/issues/nearby";
const ALERT_RADIUS_METERS = 50; // Distance to trigger alert
const POLL_INTERVAL_MS = 5000;  // Poll backend every 5 seconds

// State
let map, userMarker, accuracyCircle;
let potholesLayer = L.layerGroup();
let currentLat = 0, currentLon = 0;

// Initialize Map
function initMap() {
    // Default center (will update with GPS)
    map = L.map('map').setView([0, 0], 2);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    potholesLayer.addTo(map);

    // Custom User Icon
    const userIcon = L.divIcon({
        className: 'user-marker',
        html: '<div style="background-color: #3498db; width: 12px; height: 12px; border-radius: 50%; border: 2px solid white; box-shadow: 0 0 0 2px #3498db;"></div>',
        iconSize: [16, 16],
        iconAnchor: [8, 8]
    });

    userMarker = L.marker([0, 0], { icon: userIcon }).addTo(map);
    accuracyCircle = L.circle([0, 0], { radius: 50 }).addTo(map);
}

// Update User Location
function updateLocation(position) {
    const coords = position.coords;
    currentLat = coords.latitude;
    currentLon = coords.longitude;

    // Update Debug UI
    document.getElementById('debug-info').innerText =
        `GPS: ${currentLat.toFixed(5)}, ${currentLon.toFixed(5)} (±${coords.accuracy.toFixed(0)}m)`;

    // Move Marker
    const latLng = [currentLat, currentLon];
    userMarker.setLatLng(latLng);
    accuracyCircle.setLatLng(latLng);
    accuracyCircle.setRadius(coords.accuracy);

    // First fix: zoom to user
    if (map.getZoom() === 2) {
        map.setView(latLng, 16);
    }

    // Fetch nearby potholes immediately on move
    fetchNearbyPotholes();
}

// Error Handler
function locationError(err) {
    console.warn(`ERROR(${err.code}): ${err.message}`);
    document.getElementById('debug-info').innerText = "GPS Error: " + err.message;
}

// Fetch Potholes from Backend
async function fetchNearbyPotholes() {
    if (currentLat === 0) return;

    try {
        const url = `${API_URL}?lat=${currentLat}&lon=${currentLon}&radius=500`; // Fetch wider area (500m)
        const response = await fetch(url);
        const issues = await response.json();

        updateMapMarkers(issues);
        checkProximityAlerts(issues);

    } catch (error) {
        console.error("Failed to fetch potholes:", error);
    }
}

// Update Map Markers
function updateMapMarkers(issues) {
    potholesLayer.clearLayers();

    issues.forEach(issue => {
        let color = 'green';
        if (issue.severity === 'HIGH') color = 'red';
        if (issue.severity === 'MEDIUM') color = 'orange';

        const circle = L.circleMarker([issue.latitude, issue.longitude], {
            radius: 8,
            fillColor: color,
            color: '#fff',
            weight: 1,
            opacity: 1,
            fillOpacity: 0.8
        });

        circle.bindPopup(`<b>${issue.type}</b><br>Severity: ${issue.severity}<br>Conf: ${issue.confidence}`);
        potholesLayer.addLayer(circle);
    });
}

// Client-Side Alert Logic
function checkProximityAlerts(issues) {
    let closestDist = Infinity;
    let closestIssue = null;

    issues.forEach(issue => {
        const dist = getDistanceFromLatLonInM(currentLat, currentLon, issue.latitude, issue.longitude);
        if (dist < closestDist) {
            closestDist = dist;
            closestIssue = issue;
        }
    });

    const alertBox = document.getElementById('alert-box');

    if (closestIssue && closestDist <= ALERT_RADIUS_METERS) {
        // Show Alert
        alertBox.style.display = 'block';
        alertBox.className = closestIssue.severity === 'HIGH' ? 'alert-high' : 'alert-medium';
        document.getElementById('alert-title').innerText = `⚠️ POTHOLE AHEAD! (${closestIssue.severity})`;
        document.getElementById('alert-desc').innerText = `${closestDist.toFixed(0)} meters away.`;
    } else {
        // Hide Alert
        alertBox.style.display = 'none';
    }
}

// Haversine Helper (Client Side)
function getDistanceFromLatLonInM(lat1, lon1, lat2, lon2) {
    const R = 6371000;
    const dLat = deg2rad(lat2 - lat1);
    const dLon = deg2rad(lon2 - lon1);
    const a =
        Math.sin(dLat / 2) * Math.sin(dLat / 2) +
        Math.cos(deg2rad(lat1)) * Math.cos(deg2rad(lat2)) *
        Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
}

function deg2rad(deg) {
    return deg * (Math.PI / 180);
}

// Start
initMap();

if ("geolocation" in navigator) {
    navigator.geolocation.watchPosition(updateLocation, locationError, {
        enableHighAccuracy: true,
        timeout: 5000,
        maximumAge: 0
    });
} else {
    alert("Geolocation is not supported by your browser");
}

// Poll periodically
setInterval(fetchNearbyPotholes, POLL_INTERVAL_MS);
