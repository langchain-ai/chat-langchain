locals {
  secret_json = jsondecode(data.google_secret_manager_secret_version.chat_langchain_backend_secrets.secret_data)
  region      = "YOUR REGION"
  project_id  = "YOUR PROJECT ID"
}

provider "google" {
  project = local.project_id
  region  = local.region
}

# Load secrets from Secret Manager. You can specify your secrets in anyway you see fit.
data "google_secret_manager_secret_version" "chat_langchain_backend_secrets" {
  secret = "chat-looker-docs-backend"
}

module "chat_looker_docs_backend" {
  source = "./modules/chat_looker_docs_backend"

  project_id                  = local.project_id
  region                      = local.region
  chat_looker-docs_backend_name = "chat-looker-docs-backend"
  domain_name                 = "chat-looker-docs.vercel.app"
  image_tag                   = "docker.io/langchain/chat-looker-docs-backend:0.0.1"
  openai_api_key              = local.secret_json["openai_api_key"]
  weaviate_api_key            = local.secret_json["weaviate_api_key"]
  weaviate_url                = local.secret_json["weaviate_url"]
  langsmith_api_key           = local.secret_json["langsmith_api_key"]
}
