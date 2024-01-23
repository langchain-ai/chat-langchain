variable "chat_langchain_backend_name" {
  description = "Name to use for resources that will be created"
  type        = string
}

variable "project_id" {
  description = "The ID of the project"
  type        = string
}

variable "region" {
  description = "The region to deploy to"
  type        = string
}

variable "image_tag" {
  description = "The tag of the Chat Langchain Docker image to deploy"
  type        = string
}

variable "domain_name" {
  description = "The domain name for the backend"
  type        = string
}

variable "openai_api_key" {
  description = "Openai api key to use for the backend"
  type        = string
}

variable "weaviate_url" {
  description = "Weaviate url to use for the backend"
  type        = string
}

variable "weaviate_api_key" {
  description = "Weaviate api key to use for the backend"
  type        = string
}

variable "voyage_ai_model" {
  description = "Voyage AI model to use for the backend"
  type        = string
  default     = ""
}

variable "voyage_api_key" {
  description = "Voyage API key url to use for the backend"
  type        = string
  default     = ""
}

variable "langsmith_api_key" {
  description = "Langsmith api key to use for the backend"
  type        = string
}

variable "langchain_project" {
  description = "Langchain project to use for the backend"
  type        = string
  default     = "chat-langchain"
}

variable "min_instance_count" {
  description = "Minimum number of instances to run"
  type        = number
  default     = 1
}

variable "max_instance_count" {
  description = "Maximum number of instances to run"
  type        = number
  default     = 50
}

variable "max_instance_request_concurrency" {
  description = "Maximum number of requests to process concurrently per instance"
  type        = number
  default     = 50
}

variable "langchain_endpoint" {
  description = "Endpoint to use for LangSmith tracing"
  type        = string
  default     = "https://api.smith.langchain.com"
}

variable "fireworks_api_key" {
  description = "Fireworks api key to use for the backend"
  type        = string
  default     = ""
}

variable "anthropic_api_key" {
  description = "Anthropic api key to use for the backend"
  type        = string
  default     = ""
}

variable "env_vars" {
  description = "Environment variables to set on the backend"
  type        = map(string)
  default     = {}
}
