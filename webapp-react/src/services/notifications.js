// ─── Notification Service ───
// Handles browser push notifications, vibration, and audio alerts

const ALERT_SOUND_FREQ = 800 // Hz
const ALERT_DURATION = 200   // ms

// Request notification permission
export async function requestNotificationPermission() {
    if (!('Notification' in window)) {
        console.warn('Browser does not support notifications')
        return 'denied'
    }

    if (Notification.permission === 'granted') return 'granted'
    if (Notification.permission === 'denied') return 'denied'

    const result = await Notification.requestPermission()
    return result
}

// Get current permission status
export function getNotificationPermission() {
    if (!('Notification' in window)) return 'unsupported'
    return Notification.permission
}

// Send a browser notification
export function sendNotification(title, options = {}) {
    if (!('Notification' in window) || Notification.permission !== 'granted') return null

    const notification = new Notification(title, {
        icon: '🚧',
        badge: '⚠️',
        vibrate: [200, 100, 200, 100, 200], // vibrate pattern
        requireInteraction: true,  // stays until user dismisses
        ...options,
    })

    // Auto-close after 8 seconds
    setTimeout(() => notification.close(), 8000)
    return notification
}

// Trigger vibration (mobile devices)
export function triggerVibration(severity) {
    if (!('vibrate' in navigator)) return

    const patterns = {
        HIGH: [300, 100, 300, 100, 300, 100, 300],   // aggressive
        MEDIUM: [200, 100, 200, 100, 200],             // moderate
        LOW: [150, 100, 150],                           // gentle
    }

    navigator.vibrate(patterns[severity] || patterns.LOW)
}

// Play alert beep using Web Audio API (no audio file needed)
export function playAlertSound(severity) {
    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)()

        const freqs = { HIGH: 900, MEDIUM: 700, LOW: 500 }
        const repeats = { HIGH: 3, MEDIUM: 2, LOW: 1 }

        const freq = freqs[severity] || 600
        const count = repeats[severity] || 1

        for (let i = 0; i < count; i++) {
            const oscillator = audioCtx.createOscillator()
            const gainNode = audioCtx.createGain()

            oscillator.connect(gainNode)
            gainNode.connect(audioCtx.destination)

            oscillator.type = 'sine'
            oscillator.frequency.value = freq

            // Fade in and out for a clean beep
            const startTime = audioCtx.currentTime + (i * 0.35)
            gainNode.gain.setValueAtTime(0, startTime)
            gainNode.gain.linearRampToValueAtTime(0.3, startTime + 0.05)
            gainNode.gain.linearRampToValueAtTime(0, startTime + 0.2)

            oscillator.start(startTime)
            oscillator.stop(startTime + 0.25)
        }
    } catch (e) {
        console.warn('Audio alert failed:', e)
    }
}

// Full alert: notification + vibration + sound
export function triggerFullAlert(issue, distance) {
    const severity = issue.severity || 'MEDIUM'
    const title = `⚠️ POTHOLE AHEAD! (${distance}m)`
    const body = `Severity: ${severity} | Confidence: ${issue.confidence ? (issue.confidence * 100).toFixed(0) + '%' : 'N/A'}\nSlow down and stay alert!`

    // 1. Browser notification
    sendNotification(title, {
        body,
        tag: `pothole-${issue.id}`, // prevents duplicate notifications for same pothole
        renotify: false,
    })

    // 2. Vibration
    triggerVibration(severity)

    // 3. Sound
    playAlertSound(severity)
}
