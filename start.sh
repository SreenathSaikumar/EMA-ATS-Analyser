echo "Starting FastAPI app"

if alembic upgrade head; then
    echo "Migrations applied successfully"
else
    echo "Failed to apply migrations"
    exit 1
fi

granian src/app:app --interface asgi --host 0.0.0.0 --port 8001