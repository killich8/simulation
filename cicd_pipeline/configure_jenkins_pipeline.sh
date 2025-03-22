#!/bin/bash
# Script to configure Jenkins pipeline for the Road Object Detection project

# Set variables
JENKINS_URL="http://localhost:8080"
JENKINS_USER="admin"
JENKINS_PASSWORD="$1"  # Pass the Jenkins admin password as the first argument
PROJECT_NAME="road-object-detection"
GITHUB_REPO="https://github.com/your-org/road-object-detection-project.git"
GITHUB_CREDENTIALS_ID="github-credentials"
AWS_CREDENTIALS_ID="aws-credentials"

# Check if Jenkins CLI is available, download if not
if [ ! -f jenkins-cli.jar ]; then
    echo "Downloading Jenkins CLI..."
    curl -O $JENKINS_URL/jnlpJars/jenkins-cli.jar
fi

# Function to create a Jenkins job
create_job() {
    local job_name=$1
    local job_xml=$2
    
    echo "Creating/updating job: $job_name"
    java -jar jenkins-cli.jar -s $JENKINS_URL -auth $JENKINS_USER:$JENKINS_PASSWORD create-job $job_name < $job_xml || \
    java -jar jenkins-cli.jar -s $JENKINS_URL -auth $JENKINS_USER:$JENKINS_PASSWORD update-job $job_name < $job_xml
}

# Create the main pipeline job XML
cat > road-object-detection-pipeline.xml << EOF
<?xml version='1.1' encoding='UTF-8'?>
<flow-definition plugin="workflow-job@2.40">
  <description>Road Object Detection CI/CD Pipeline</description>
  <keepDependencies>false</keepDependencies>
  <properties>
    <com.coravy.hudson.plugins.github.GithubProjectProperty plugin="github@1.33.1">
      <projectUrl>$GITHUB_REPO</projectUrl>
      <displayName></displayName>
    </com.coravy.hudson.plugins.github.GithubProjectProperty>
    <org.jenkinsci.plugins.workflow.job.properties.PipelineTriggersJobProperty>
      <triggers>
        <hudson.triggers.SCMTrigger>
          <spec>H/15 * * * *</spec>
          <ignorePostCommitHooks>false</ignorePostCommitHooks>
        </hudson.triggers.SCMTrigger>
      </triggers>
    </org.jenkinsci.plugins.workflow.job.properties.PipelineTriggersJobProperty>
  </properties>
  <definition class="org.jenkinsci.plugins.workflow.cps.CpsScmFlowDefinition" plugin="workflow-cps@2.90">
    <scm class="hudson.plugins.git.GitSCM" plugin="git@4.7.1">
      <configVersion>2</configVersion>
      <userRemoteConfigs>
        <hudson.plugins.git.UserRemoteConfig>
          <url>$GITHUB_REPO</url>
          <credentialsId>$GITHUB_CREDENTIALS_ID</credentialsId>
        </hudson.plugins.git.UserRemoteConfig>
      </userRemoteConfigs>
      <branches>
        <hudson.plugins.git.BranchSpec>
          <name>*/main</name>
        </hudson.plugins.git.BranchSpec>
      </branches>
      <doGenerateSubmoduleConfigurations>false</doGenerateSubmoduleConfigurations>
      <submoduleCfg class="empty-list"/>
      <extensions/>
    </scm>
    <scriptPath>cicd_pipeline/Jenkinsfile</scriptPath>
    <lightweight>true</lightweight>
  </definition>
  <triggers/>
  <disabled>false</disabled>
</flow-definition>
EOF

# Create the job
create_job $PROJECT_NAME-pipeline road-object-detection-pipeline.xml

# Create a folder for the project
cat > road-object-detection-folder.xml << EOF
<?xml version='1.1' encoding='UTF-8'?>
<com.cloudbees.hudson.plugins.folder.Folder plugin="cloudbees-folder@6.15">
  <description>Road Object Detection Project</description>
  <displayName>Road Object Detection</displayName>
  <properties/>
  <folderViews class="com.cloudbees.hudson.plugins.folder.views.DefaultFolderViewHolder">
    <views>
      <hudson.model.AllView>
        <owner class="com.cloudbees.hudson.plugins.folder.Folder" reference="../../../.."/>
        <name>All</name>
        <filterExecutors>false</filterExecutors>
        <filterQueue>false</filterQueue>
        <properties class="hudson.model.View$PropertyList"/>
      </hudson.model.AllView>
    </views>
    <tabBar class="hudson.views.DefaultViewsTabBar"/>
  </folderViews>
  <healthMetrics/>
  <icon class="com.cloudbees.hudson.plugins.folder.icons.StockFolderIcon"/>
</com.cloudbees.hudson.plugins.folder.Folder>
EOF

# Create the folder
create_job $PROJECT_NAME road-object-detection-folder.xml

# Move the pipeline job to the folder
echo "Moving pipeline job to folder..."
java -jar jenkins-cli.jar -s $JENKINS_URL -auth $JENKINS_USER:$JENKINS_PASSWORD groovy = << EOF
def job = Jenkins.instance.getItem("$PROJECT_NAME-pipeline")
def folder = Jenkins.instance.getItem("$PROJECT_NAME")
if (job != null && folder != null) {
    Jenkins.instance.getItems().remove(job)
    folder.getItems().add(job)
    job.setParent(folder)
    println "Job moved successfully"
} else {
    println "Job or folder not found"
}
EOF

# Create credentials for GitHub and AWS
echo "Creating credentials..."
java -jar jenkins-cli.jar -s $JENKINS_URL -auth $JENKINS_USER:$JENKINS_PASSWORD groovy = << EOF
import com.cloudbees.plugins.credentials.domains.Domain
import com.cloudbees.plugins.credentials.impl.UsernamePasswordCredentialsImpl
import com.cloudbees.plugins.credentials.CredentialsScope
import com.cloudbees.jenkins.plugins.awscredentials.AWSCredentialsImpl
import com.cloudbees.plugins.credentials.SystemCredentialsProvider

def createOrUpdateCredentials(String id, String description, def credentials) {
    def credentialsStore = Jenkins.instance.getExtensionList('com.cloudbees.plugins.credentials.SystemCredentialsProvider')[0].getStore()
    def existingCredentials = credentialsStore.getCredentials(Domain.global()).find { it.id == id }
    
    if (existingCredentials) {
        credentialsStore.removeCredentials(Domain.global(), existingCredentials)
    }
    
    credentialsStore.addCredentials(Domain.global(), credentials)
    println "Credentials '$id' created/updated"
}

// Create GitHub credentials (placeholder - you'll need to update with real values)
def githubCredentials = new UsernamePasswordCredentialsImpl(
    CredentialsScope.GLOBAL,
    "$GITHUB_CREDENTIALS_ID",
    "GitHub credentials for $PROJECT_NAME",
    "github-username",
    "github-password"
)
createOrUpdateCredentials("$GITHUB_CREDENTIALS_ID", "GitHub credentials", githubCredentials)

// Create AWS credentials (placeholder - you'll need to update with real values)
def awsCredentials = new AWSCredentialsImpl(
    CredentialsScope.GLOBAL,
    "$AWS_CREDENTIALS_ID",
    "AWS access key ID",
    "AWS secret access key",
    "AWS credentials for $PROJECT_NAME",
    null,
    null
)
createOrUpdateCredentials("$AWS_CREDENTIALS_ID", "AWS credentials", awsCredentials)
EOF

echo "Jenkins pipeline configuration complete!"
echo "Please update the GitHub and AWS credentials with real values in the Jenkins UI."
echo "Pipeline is available at: $JENKINS_URL/job/$PROJECT_NAME/job/$PROJECT_NAME-pipeline/"
