import csv
import os
import sys

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from loguru import logger

from app.models import config
from app.models.schemas import Legislation, Policy

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")
settings = config.get_settings()

logger.remove()
logger.add(sys.stderr, level="DEBUG")


@router.get("/")
async def get(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "websocketEndpoint": settings.websocket_endpoint},
    )


@router.get("/about")
async def get_about(request: Request):
    return templates.TemplateResponse(
        "about.html",
        {"request": request},
    )


@router.get("/acts")
async def get_acts(request: Request):
    file_path = os.getcwd() + "/scripts/legislation.csv"
    acts = []
    with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            logger.debug(row)
            acts.append(Legislation(act_name=row[0], year=row[1], url=row[2]))

    return templates.TemplateResponse("acts.html", {"request": request, "acts": acts})


@router.get("/policies")
async def get_policies(request: Request):
    file_path = os.getcwd() + "/scripts/policies.csv"
    policies = []
    with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            logger.debug(row)
            policies.append(Policy(policy_name=row[0], url=row[1]))

    return templates.TemplateResponse(
        "policies.html", {"request": request, "policies": policies}
    )


@router.get("/faq")
async def get_faq(request: Request):
    return templates.TemplateResponse(
        "faq.html",
        {"request": request},
    )
