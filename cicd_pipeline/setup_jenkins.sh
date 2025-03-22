#!/bin/bash
# Script to set up Jenkins for the Road Object Detection project

# Set variables
JENKINS_NAMESPACE="jenkins"
PROJECT_NAMESPACE="road-object-detection"
AWS_REGION="us-east-1"
CLUSTER_NAME="road-object-detection-cluster"

# Create Jenkins namespace if it doesn't exist
kubectl get namespace $JENKINS_NAMESPACE || kubectl create namespace $JENKINS_NAMESPACE

# Create service account for Jenkins
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: jenkins
  namespace: $JENKINS_NAMESPACE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: jenkins-cluster-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets", "namespaces", "persistentvolumeclaims"]
  verbs: ["create", "delete", "get", "list", "patch", "update", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "statefulsets", "replicasets"]
  verbs: ["create", "delete", "get", "list", "patch", "update", "watch"]
- apiGroups: ["batch"]
  resources: ["jobs", "cronjobs"]
  verbs: ["create", "delete", "get", "list", "patch", "update", "watch"]
- apiGroups: ["networking.k8s.io"]
  resources: ["ingresses"]
  verbs: ["create", "delete", "get", "list", "patch", "update", "watch"]
- apiGroups: ["storage.k8s.io"]
  resources: ["storageclasses"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: jenkins-cluster-role-binding
subjects:
- kind: ServiceAccount
  name: jenkins
  namespace: $JENKINS_NAMESPACE
roleRef:
  kind: ClusterRole
  name: jenkins-cluster-role
  apiGroup: rbac.authorization.k8s.io
EOF

# Create persistent volume for Jenkins
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: jenkins-pvc
  namespace: $JENKINS_NAMESPACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
EOF

# Deploy Jenkins
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jenkins
  namespace: $JENKINS_NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: jenkins
  template:
    metadata:
      labels:
        app: jenkins
    spec:
      serviceAccountName: jenkins
      containers:
      - name: jenkins
        image: jenkins/jenkins:lts
        ports:
        - containerPort: 8080
        - containerPort: 50000
        volumeMounts:
        - name: jenkins-home
          mountPath: /var/jenkins_home
        env:
        - name: JAVA_OPTS
          value: "-Djenkins.install.runSetupWizard=false"
      volumes:
      - name: jenkins-home
        persistentVolumeClaim:
          claimName: jenkins-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: jenkins
  namespace: $JENKINS_NAMESPACE
spec:
  type: LoadBalancer
  ports:
  - port: 8080
    targetPort: 8080
    name: http
  - port: 50000
    targetPort: 50000
    name: jnlp
  selector:
    app: jenkins
EOF

# Wait for Jenkins to be ready
echo "Waiting for Jenkins to be ready..."
kubectl -n $JENKINS_NAMESPACE rollout status deployment/jenkins

# Get Jenkins URL
JENKINS_URL=$(kubectl -n $JENKINS_NAMESPACE get service jenkins -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
echo "Jenkins URL: http://$JENKINS_URL:8080"

# Get initial admin password
echo "Retrieving initial admin password..."
sleep 60  # Wait for Jenkins to initialize
JENKINS_POD=$(kubectl -n $JENKINS_NAMESPACE get pods -l app=jenkins -o jsonpath='{.items[0].metadata.name}')
JENKINS_PASSWORD=$(kubectl -n $JENKINS_NAMESPACE exec $JENKINS_POD -- cat /var/jenkins_home/secrets/initialAdminPassword)
echo "Jenkins initial admin password: $JENKINS_PASSWORD"

echo "Jenkins setup complete!"
echo "Please access Jenkins at http://$JENKINS_URL:8080"
echo "Use the initial admin password: $JENKINS_PASSWORD"
echo "After logging in, install the following plugins:"
echo "- Kubernetes plugin"
echo "- Docker plugin"
echo "- AWS Credentials plugin"
echo "- Pipeline plugin"
echo "- Blue Ocean plugin"
echo "- JUnit plugin"
echo "- Cobertura plugin"
echo "- Email Extension plugin"
