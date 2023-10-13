terraform {
  backend "gcs" {
    bucket = "langchain-terraform-state-dev"
    prefix = "terraform/state/chat-langchain"
  }
}
