# Monitoring with Prometheus and Grafana

This directory contains the monitoring solution for the Road Object Detection project using Prometheus and Grafana.

## Overview

The monitoring stack provides real-time visibility into the performance of the road object detection system, including:

- System resource utilization (CPU, memory, disk, GPU)
- Kubernetes cluster health
- Model inference performance metrics
- Detection statistics by object class
- API request rates and error rates

## Components

### Prometheus

Prometheus is used for metrics collection and alerting. It scrapes metrics from various components of the system:

- CARLA simulation service
- Data processing service
- Model training service
- Model inference service
- Kubernetes nodes and pods
- Node exporter (system metrics)
- Kube-state-metrics (Kubernetes state)

### Grafana

Grafana provides visualization dashboards for the collected metrics. The main dashboard includes:

- Cluster overview (CPU, memory, GPU usage)
- Pod status monitoring
- Model performance metrics (inference latency, detection counts)
- Average confidence by object class
- API request rates and error rates
- Resource usage by service

### Exporters

- **Node Exporter**: Collects system-level metrics from Kubernetes nodes
- **Kube-state-metrics**: Collects metrics about the state of Kubernetes objects

## Deployment

The monitoring stack is deployed to Kubernetes using the manifests in the `kubernetes` directory:

1. `prometheus.yaml`: Deploys Prometheus server with configuration and alerting rules
2. `grafana.yaml`: Deploys Grafana with pre-configured datasources and dashboards
3. `exporters.yaml`: Deploys node-exporter and kube-state-metrics
4. `deploy.sh`: Script to deploy the entire monitoring stack

To deploy the monitoring stack:

```bash
cd kubernetes
chmod +x deploy.sh
./deploy.sh
```

## Alerting

The monitoring system includes alerts for:

- High CPU/memory usage
- Pod crash looping
- High inference latency
- High error rates
- Job failures
- Service downtime
- Low disk space
- High GPU utilization

## Customization

To customize the monitoring solution:

1. Edit `prometheus/prometheus.yml` to modify scrape configurations
2. Edit `prometheus/prometheus-rules.yml` to modify alerting rules
3. Import additional dashboards to Grafana or modify the existing ones

## Access

After deployment, Grafana is accessible via a LoadBalancer service. The default credentials are:

- Username: admin
- Password: admin123

It is recommended to change the default password after the first login.
