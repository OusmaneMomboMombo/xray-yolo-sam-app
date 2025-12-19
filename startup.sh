#!/bin/bash

echo "🔵 Démarrage de l'application Flask avec Gunicorn..."

gunicorn app:app \
  --bind=0.0.0.0:8000 \
  --workers=1 \
  --timeout 600
