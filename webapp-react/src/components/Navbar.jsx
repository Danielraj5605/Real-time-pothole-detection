import { NavLink } from 'react-router-dom'

function Navbar() {
    return (
        <nav className="navbar">
            <div className="nav-brand">
                <span className="nav-icon">🚧</span>
                <span className="nav-title">Pothole Alert</span>
            </div>
            <div className="nav-links">
                <NavLink to="/" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
                    Dashboard
                </NavLink>
                <NavLink to="/about" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
                    About
                </NavLink>
            </div>
        </nav>
    )
}

export default Navbar
