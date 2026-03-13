def estimate_rul(smoothed_error, anomaly_threshold):
    """
    Simple working RUL:
    RUL follows health directly, so whenever health changes, RUL changes.
    """
    health = compute_health(smoothed_error, anomaly_threshold)

    if health <= 0:
        return 0.0, "failed"

    if health >= 95:
        return float(clamp(health, 0.0, MAX_RUL_HOURS)), "healthy"
    elif health >= 70:
        return float(clamp(health, 0.0, MAX_RUL_HOURS)), "good"
    elif health >= 40:
        return float(clamp(health, 0.0, MAX_RUL_HOURS)), "warning"
    else:
        return float(clamp(health, 0.0, MAX_RUL_HOURS)), "critical"