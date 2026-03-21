echo "Starting application"

if alembic upgrade head; then
    echo "Migrations applied successfully"
else
    echo "Failed to apply migrations"
    exit 1
fi

echo "Creating SQS queues"
if python -m src.startup_scripts.create_sqs_queues; then
    echo "SQS queues created successfully"
else
    echo "Failed to create SQS queues"
    exit 1
fi


if [ $SERVICE = "api" ]; then
    granian src/app:app --interface asgi --host 0.0.0.0 --port 8001
elif [ $SERVICE = "consumer" ]; then
    python -m src.ats_processor_consumer
else
    echo "Invalid service"
    exit 1
fi
