import { useState, useEffect, useRef } from 'react'
import { MapContainer, TileLayer, Marker, Popup, CircleMarker, Circle, useMap } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import {
    requestNotificationPermission,
    getNotificationPermission,
    triggerFullAlert,
} from '../services/notifications'

// Fix Leaflet default icon issue in React bundlers
import L from 'leaflet'
import markerIcon from 'leaflet/dist/images/marker-icon.png'
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png'
import markerShadow from 'leaflet/dist/images/marker-shadow.png'

delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
    iconUrl: markerIcon,
    iconRetinaUrl: markerIcon2x,
    shadowUrl: markerShadow,
})

const API_BASE = "http://127.0.0.1:8000"
const POLL_INTERVAL = 5000
const ALERT_RADIUS = 50 // meters
const severityColor = { HIGH: '#ef4444', MEDIUM: '#f59e0b', LOW: '#22c55e' }

// Auto-fly to user location
function FlyToUser({ position }) {
    const map = useMap()
    useEffect(() => {
        if (position) map.flyTo(position, 16, { duration: 1.5 })
    }, [position, map])
    return null
}

function Dashboard() {
    const [position, setPosition] = useState(null)
    const [issues, setIssues] = useState([])
    const [alert, setAlert] = useState(null)
    const [notifPermission, setNotifPermission] = useState(getNotificationPermission())

    // Track which potholes we've already alerted for (avoid spam)
    const alertedPotholes = useRef(new Set())

    // Request notification permission on mount
    useEffect(() => {
        requestNotificationPermission().then(perm => setNotifPermission(perm))
    }, [])

    // Watch GPS
    useEffect(() => {
        if (!navigator.geolocation) return
        const id = navigator.geolocation.watchPosition(
            (p) => setPosition([p.coords.latitude, p.coords.longitude]),
            (e) => console.warn("GPS:", e.message),
            { enableHighAccuracy: true, timeout: 10000 }
        )
        return () => navigator.geolocation.clearWatch(id)
    }, [])

    // Poll backend for nearby issues
    useEffect(() => {
        if (!position) return
        const fetchIssues = async () => {
            try {
                const [lat, lon] = position
                const res = await fetch(`${API_BASE}/api/issues/nearby?lat=${lat}&lon=${lon}&radius=2000`)
                const data = await res.json()
                setIssues(data)

                // Check proximity alerts
                let closest = null, closestDist = Infinity
                data.forEach(issue => {
                    const d = haversine(lat, lon, issue.latitude, issue.longitude)
                    if (d < closestDist) { closestDist = d; closest = issue }

                    // Trigger push notification if within alert radius & not already alerted
                    if (d <= ALERT_RADIUS && !alertedPotholes.current.has(issue.id)) {
                        alertedPotholes.current.add(issue.id)
                        triggerFullAlert(issue, Math.round(d))
                    }
                })

                // Clear alerts for potholes we've moved away from (>200m)
                alertedPotholes.current.forEach(id => {
                    const issue = data.find(i => i.id === id)
                    if (issue) {
                        const dist = haversine(lat, lon, issue.latitude, issue.longitude)
                        if (dist > 200) alertedPotholes.current.delete(id)
                    }
                })

                setAlert(closest && closestDist < ALERT_RADIUS ? { ...closest, distance: Math.round(closestDist) } : null)
            } catch (err) {
                console.error("API Error:", err)
            }
        }
        fetchIssues()
        const interval = setInterval(fetchIssues, POLL_INTERVAL)
        return () => clearInterval(interval)
    }, [position])

    const highCount = issues.filter(i => i.severity === 'HIGH').length
    const medCount = issues.filter(i => i.severity === 'MEDIUM').length
    const lowCount = issues.filter(i => i.severity === 'LOW').length

    return (
        <div className="dashboard-layout">
            {/* Sidebar */}
            <aside className="sidebar">
                {/* Stats */}
                <div className="stats-bar">
                    <div className="stat-card stat-high">
                        <div className="stat-value">{highCount}</div>
                        <div className="stat-label">High</div>
                    </div>
                    <div className="stat-card stat-medium">
                        <div className="stat-value">{medCount}</div>
                        <div className="stat-label">Medium</div>
                    </div>
                    <div className="stat-card stat-low">
                        <div className="stat-value">{lowCount}</div>
                        <div className="stat-label">Low</div>
                    </div>
                </div>

                {/* Notification Status */}
                <div className="notif-status">
                    <span className={`notif-dot ${notifPermission === 'granted' ? 'active' : ''}`} />
                    <span>
                        {notifPermission === 'granted' ? '🔔 Notifications ON' :
                            notifPermission === 'denied' ? '🔕 Notifications Blocked' :
                                '🔔 Notifications Pending'}
                    </span>
                    {notifPermission !== 'granted' && notifPermission !== 'denied' && (
                        <button
                            className="notif-btn"
                            onClick={() => requestNotificationPermission().then(p => setNotifPermission(p))}
                        >
                            Enable
                        </button>
                    )}
                </div>

                {/* GPS Status */}
                <div className="gps-status">
                    <div className={`gps-dot ${position ? '' : 'inactive'}`} />
                    <span>{position ? 'GPS Active' : 'Waiting for GPS...'}</span>
                    {position && (
                        <span className="gps-coords">
                            {position[0].toFixed(4)}, {position[1].toFixed(4)}
                        </span>
                    )}
                </div>

                {/* Issue List */}
                <div className="issue-list-header">
                    Nearby Issues ({issues.length})
                </div>
                <div className="issue-list">
                    {issues.length === 0 ? (
                        <div className="empty-state">
                            <div className="empty-icon">📍</div>
                            <p>No issues detected nearby.<br />Run the upload script to add data.</p>
                        </div>
                    ) : (
                        issues.map(issue => (
                            <div className="issue-card" key={issue.id}>
                                <div className={`severity-badge ${issue.severity?.toLowerCase()}`} />
                                <div className="issue-info">
                                    <div className="issue-title">Pothole #{issue.id}</div>
                                    <div className="issue-meta">
                                        {issue.latitude.toFixed(4)}, {issue.longitude.toFixed(4)} · {issue.severity}
                                    </div>
                                </div>
                                <div className="issue-confidence">
                                    {issue.confidence ? `${(issue.confidence * 100).toFixed(0)}%` : '—'}
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </aside>

            {/* Map */}
            <main className="map-area">
                <MapContainer
                    center={position || [12.8546, 80.068]}
                    zoom={position ? 16 : 14}
                    style={{ height: '100%', width: '100%' }}
                >
                    <TileLayer
                        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    />
                    {position && <FlyToUser position={position} />}
                    {position && (
                        <>
                            <Marker position={position}>
                                <Popup>📍 Your Location</Popup>
                            </Marker>
                            <Circle
                                center={position}
                                radius={ALERT_RADIUS}
                                pathOptions={{ color: '#3b82f6', fillColor: '#3b82f6', fillOpacity: 0.1, weight: 1, dashArray: '5,10' }}
                            />
                        </>
                    )}
                    {issues.map(issue => (
                        <CircleMarker
                            key={issue.id}
                            center={[issue.latitude, issue.longitude]}
                            radius={10}
                            pathOptions={{
                                color: '#fff',
                                weight: 2,
                                fillColor: severityColor[issue.severity] || '#22c55e',
                                fillOpacity: 0.85,
                            }}
                        >
                            <Popup>
                                <div style={{ color: '#333', minWidth: 150 }}>
                                    <strong>🕳️ Pothole #{issue.id}</strong><br />
                                    <b>Severity:</b> {issue.severity}<br />
                                    <b>Confidence:</b> {issue.confidence ? `${(issue.confidence * 100).toFixed(0)}%` : 'N/A'}<br />
                                    <b>Status:</b> {issue.status}
                                </div>
                            </Popup>
                        </CircleMarker>
                    ))}
                </MapContainer>

                {/* In-App Alert */}
                {alert && (
                    <div className="alert-overlay">
                        <div className={`alert-card ${alert.severity?.toLowerCase()}`}>
                            <div className="alert-icon">⚠️</div>
                            <div className="alert-text">
                                <h3>POTHOLE AHEAD!</h3>
                                <p>{alert.distance}m away · Severity: {alert.severity}</p>
                            </div>
                        </div>
                    </div>
                )}
            </main>
        </div>
    )
}

function haversine(lat1, lon1, lat2, lon2) {
    const R = 6371e3
    const toRad = d => d * Math.PI / 180
    const dLat = toRad(lat2 - lat1), dLon = toRad(lon2 - lon1)
    const a = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
}

export default Dashboard
