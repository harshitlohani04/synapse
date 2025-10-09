### This is the DB creation file ###

from dotenv import load_dotenv
import os
from typing import List
from pydantic import BaseModel

# for sql database
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

# for non-sql database
import boto3
# loading the env vars
load_dotenv()

masterPass = os.getenv("MASTER_PASS_DB")
urlRelational = f"postgresql://trello_app_dbs:{masterPass}@database-1.cxaiiq2u0r8b.ap-south-1.rds.amazonaws.com/database-1"

dynamodb = boto3.resource(
    "dynamodb",
    region_name = "ap-south-1",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY")
)

engine_relational = create_engine(url=urlRelational)
Base = declarative_base() # which class inherits this should be treated as a table

# will contain the users login credentials, later on even the trello credentials
class Users(Base):
    __tablename__ = "userscredentials"
    user_id = Column(Integer, primary_key=True)
    username = Column(String(100), nullable = False)
    password = Column(String(100), nullable=False)

# class defining the data format for the data from celery worker
class Data(BaseModel):
    board_desc: str
    keywords: List[str]

# will contain the celery task information and the completion info (non-relational db)
# this only defines the key attributes rest is all schema less
class CeleryTaskDB:
    def __init__(self):
        self.table = dynamodb.create_table(
            TableName="callbackdata",
            KeySchema=[
                {"AttributeName": "task_id", "KeyType": "HASH"}  # Partition key on the basis of the task id
            ],
            AttributeDefinitions=[
                {"AttributeName": "username", "AttributeType": "S"} # VERIFY THE DTYPE OF THE CELERY TASK
            ],
            ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5}
        )
    
    def flagPushDatainDB(self, task_id: str, data: Data):
        board_desc = data.board_desc
        keywords = data.keywords

        # final pushing
        try:
            self.table.push_item(
                Item = {
                    "task_id": task_id,
                    "board_desc": board_desc,
                    "keywords": keywords
                }
            )
            return True
        except Exception as e:
            print(f"Encountered issue in pushing data to db: {e}")
            return None



