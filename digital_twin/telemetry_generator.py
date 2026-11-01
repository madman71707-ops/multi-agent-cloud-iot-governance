def generate_telemetry(twin):
    devices = twin.devices

    return {
        "avg_cpu_load": float(sum(device.cpu_load for device in devices) / len(devices)),
        "avg_memory_load": float(sum(device.memory_load for device in devices) / len(devices)),
        "avg_network_load": float(sum(device.network_load for device in devices) / len(devices)),
        "avg_trust_score": float(sum(device.trust_score for device in devices) / len(devices)),
        "total_cost_per_hour": float(sum(device.cost_per_hour for device in devices)),
        "total_energy_kwh": float(sum(device.energy_kwh for device in devices)),
    }
