function About() {
    return (
        <div className="about-page">
            <div className="about-card">
                <h2>About This Project</h2>
                <p>
                    <strong>Pothole Alert System</strong> is an academic prototype for a civic infrastructure
                    monitoring platform. It detects potholes using computer vision (YOLOv8) and alerts
                    drivers in real time through a Progressive Web App.
                </p>

                <h3>How It Works</h3>
                <div className="about-steps">
                    <div className="about-step">
                        <span className="step-num">1</span>
                        <div>
                            <strong>Detection</strong>
                            <p>YOLOv8 processes road footage and identifies potholes with severity classification.</p>
                        </div>
                    </div>
                    <div className="about-step">
                        <span className="step-num">2</span>
                        <div>
                            <strong>Backend</strong>
                            <p>FastAPI server stores detections in PostgreSQL (Neon) with GPS coordinates.</p>
                        </div>
                    </div>
                    <div className="about-step">
                        <span className="step-num">3</span>
                        <div>
                            <strong>Alert</strong>
                            <p>This PWA fetches nearby potholes and warns you when you're approaching one.</p>
                        </div>
                    </div>
                </div>

                <h3>Tech Stack</h3>
                <div className="tech-list">
                    <span className="tech-tag">YOLOv8</span>
                    <span className="tech-tag">FastAPI</span>
                    <span className="tech-tag">PostgreSQL (Neon)</span>
                    <span className="tech-tag">React</span>
                    <span className="tech-tag">Leaflet.js</span>
                    <span className="tech-tag">Geolocation API</span>
                </div>

                <h3>Team</h3>
                <p>Built as a minor academic project for civic infrastructure improvement.</p>
            </div>
        </div>
    )
}

export default About
