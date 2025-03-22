#!/bin/bash
# Script to deploy the monitoring stack for the Road Object Detection project

# Set variables
NAMESPACE="monitoring"
PROJECT_NAMESPACE="road-object-detection"

# Create namespace if it doesn't exist
kubectl get namespace $NAMESPACE || kubectl create namespace $NAMESPACE

# Apply Kubernetes manifests
echo "Deploying Prometheus..."
kubectl apply -f prometheus.yaml

echo "Deploying Grafana..."
kubectl apply -f grafana.yaml

echo "Deploying exporters (node-exporter and kube-state-metrics)..."
kubectl apply -f exporters.yaml

# Wait for deployments to be ready
echo "Waiting for Prometheus to be ready..."
kubectl -n $NAMESPACE rollout status deployment/prometheus

echo "Waiting for Grafana to be ready..."
kubectl -n $NAMESPACE rollout status deployment/grafana

echo "Waiting for kube-state-metrics to be ready..."
kubectl -n $NAMESPACE rollout status deployment/kube-state-metrics

# Get service URLs
GRAFANA_URL=$(kubectl -n $NAMESPACE get service grafana -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
PROMETHEUS_URL=$(kubectl -n $NAMESPACE get service prometheus -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

echo "Monitoring stack deployment complete!"
echo "Grafana URL: http://$GRAFANA_URL:3000"
echo "Prometheus URL: http://$PROMETHEUS_URL:9090"
echo "Default Grafana credentials:"
echo "  Username: admin"
echo "  Password: admin123"
echo "Please change the default password after first login."
