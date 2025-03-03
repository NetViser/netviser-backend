import json
from typing import Literal
import boto3
from botocore.exceptions import ClientError
from fastapi import HTTPException
from app.configs.config import get_settings

settings = get_settings()


class LambdaService:
    """
    Singleton class for interacting with AWS Lambda for asynchronous invocation (Event).
    """

    _instance = None

    def __new__(cls):
        """Create a new instance of the LambdaService class if it doesn't already exist."""
        if cls._instance is None:
            cls._instance = super(LambdaService, cls).__new__(cls)
            cls._instance.initialize_client()
        return cls._instance

    def initialize_client(self):
        """Initialize the boto3 client for Lambda."""
        region_name = settings.AWS_REGION
        access_key = settings.AWS_ACCESS_KEY_ID
        secret_access_key = settings.AWS_SECRET_ACCESS_KEY

        if not all([access_key, secret_access_key, region_name]):
            raise ValueError("Some environment variables are missing.")

        # Create a boto3 client for Lambda
        self.client = boto3.client(
            "lambda",
            region_name=region_name,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_access_key,
        )
        self.functions = {
            "inference_func": settings.LAMBDA_INFERENCE_FUNCTION_NAME,
        }

    async def invoke_function(
        self,
        function_name: Literal["inference_func"],
        function_params: dict,
    ) -> dict:
        """
        Invoke a Lambda function asynchronously (Event).

        :param function_name: The name of the Lambda function to invoke.
        :param function_params: The parameters to send to the Lambda function as a dictionary.
        :param qualifier: The version or alias of the Lambda function to invoke (optional).
        :return: The response from the Lambda function invocation, only including the status code.
        """
        try:
            # Invoke the function asynchronously using InvocationType "Event"
            response = self.client.invoke(
                FunctionName=self.functions[function_name],
                InvocationType="RequestResponse",
                Payload=json.dumps(
                    function_params
                ),  # The payload must be a JSON string
            )

            payload = response.get("Payload").read()
            body = json.loads(payload)["body"]

            # Return only the status code as per the requirement
            return {"statusCode": response["StatusCode"], **body}

        except ClientError as e:
            raise HTTPException(
                status_code=500, detail="Failed to invoke Lambda function."
            ) from e
