locals {
  secret_json = jsondecode(data.google_secret_manager_secret_version.chat_langchain_backend_secrets.secret_data)
  region      = "us-central1"
  project_id  = "langchain-test-387119"
}

provider "google" {
  project = local.project_id
  region  = local.region
}

data "google_secret_manager_secret_version" "chat_langchain_backend_secrets" {
  secret = "chat-langchain-backend"
}

module "chat_langchain_backend" {
  source = "./modules/chat_langchain_backend"

  project_id                  = local.project_id
  region                      = local.region
  chat_langchain_backend_name = "chat-langchain-backend"
  domain_name                 = "chat-langchain-backend.langchain.dev"
  image_tag                   = "docker.io/langchain/chat-langchain-backend:0.0.1"
  openai_api_key              = local.secret_json["openai_api_key"]
  weaviate_api_key            = local.secret_json["weaviate_api_key"]
  weaviate_url                = local.secret_json["weaviate_url"]
  langsmith_api_key           = local.secret_json["langsmith_api_key"]
}
