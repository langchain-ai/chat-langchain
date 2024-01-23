# Common environment variables
locals {
  voyager_vars = var.voyage_ai_model != "" && var.voyage_api_key != "" ? {
    VOYAGE_AI_MODEL = var.voyage_ai_model
    VOYAGE_API_KEY  = var.voyage_api_key
  } : {}
  env_vars = merge(local.voyager_vars, {
    OPENAI_API_KEY       = var.openai_api_key
    WEAVIATE_URL         = var.weaviate_url
    WEAVIATE_API_KEY     = var.weaviate_api_key
    LANGCHAIN_TRACING_V2 = true
    LANGCHAIN_ENDPOINT   = var.langchain_endpoint
    LANGCHAIN_API_KEY    = var.langsmith_api_key
    LANGCHAIN_PROJECT    = var.langchain_project
    FIREWORKS_API_KEY    = var.fireworks_api_key
    ANTHROPIC_API_KEY    = var.anthropic_api_key
    }, var.env_vars
  )
}

# No auth policy since auth is handled by Supabase or internal API Key auth
data "google_iam_policy" "noauth" {
  binding {
    role = "roles/run.invoker"

    members = [
      "allUsers",
    ]
  }
}

# Web service
resource "google_cloud_run_v2_service" "chat_langchain_backend" {
  name     = var.chat_langchain_backend_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"

  template {
    max_instance_request_concurrency = var.max_instance_request_concurrency
    scaling {
      min_instance_count = var.min_instance_count
      max_instance_count = var.max_instance_count
    }
    containers {
      image = var.image_tag

      dynamic "env" {
        for_each = local.env_vars
        content {
          name  = env.key
          value = env.value
        }
      }
      resources {
        limits = {
          cpu    = 2
          memory = "4Gi"
        }
        startup_cpu_boost = true
        cpu_idle          = false
      }
    }
  }

  project = var.project_id
}

resource "google_cloud_run_v2_service_iam_policy" "web_noauth" {
  location    = google_cloud_run_v2_service.chat_langchain_backend.location
  project     = google_cloud_run_v2_service.chat_langchain_backend.project
  name        = google_cloud_run_v2_service.chat_langchain_backend.name
  policy_data = data.google_iam_policy.noauth.policy_data
}

resource "google_compute_region_network_endpoint_group" "web_serverless_neg" {
  provider              = google-beta
  project               = var.project_id
  name                  = "hub-serverless-neg-web"
  network_endpoint_type = "SERVERLESS"
  region                = var.region
  cloud_run {
    service = google_cloud_run_v2_service.chat_langchain_backend.name
  }
}

resource "google_compute_security_policy" "hub_web_noauth_lb_http" {
  name        = "hub-web-noauth-lb-http-authorization-throttle"
  project     = var.project_id
  description = "Web Security Policy"

  rule {
    action      = "throttle"
    description = "IP Address Throttle"
    priority    = "2147483647"

    match {
      config {
        src_ip_ranges = ["*"]
      }
      versioned_expr = "SRC_IPS_V1"
    }

    rate_limit_options {
      conform_action = "allow"
      exceed_action  = "deny(429)"

      enforce_on_key = "IP"

      rate_limit_threshold {
        count        = 5000
        interval_sec = 60
      }
    }

    preview = false
  }
}

module "lb-http-web" {
  source  = "GoogleCloudPlatform/lb-http/google//modules/serverless_negs"
  version = "9.0"
  name    = "hub-web-lb-http"
  project = var.project_id

  ssl                             = true
  managed_ssl_certificate_domains = [var.domain_name]
  https_redirect                  = true
  create_address                  = true

  backends = {
    default = {
      description = null
      groups = [
        {
          group = google_compute_region_network_endpoint_group.web_serverless_neg.id
        }
      ]
      enable_cdn              = false
      edge_security_policy    = null
      security_policy         = google_compute_security_policy.hub_web_noauth_lb_http.id
      custom_request_headers  = null
      custom_response_headers = null

      iap_config = {
        enable               = false
        oauth2_client_id     = ""
        oauth2_client_secret = ""
      }
      log_config = {
        enable      = false
        sample_rate = null
      }
      protocol         = null
      port_name        = null
      compression_mode = null
    }
  }
}
