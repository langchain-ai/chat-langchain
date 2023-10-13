terraform {
  backend "gcs" {
    bucket = "YOUR BUCKET"
    prefix = "YOUR PREFIX"
  }
}
