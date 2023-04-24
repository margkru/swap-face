import os

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

USER = os.environ.get('DB_USER')
PASS = os.environ.get('DB_PASS')
HOST = os.environ.get('DB_HOST')
NAME = os.environ.get('DB_NAME')


postgresql = {'host': HOST,
              'user': USER,
              'passwd': PASS,
              'db': NAME}

postgresqlConfig = "postgresql://{}:{}@{}/{}".format(postgresql['user'], postgresql['passwd'],
                                                              postgresql['host'], postgresql['db'])


engine = create_engine(postgresqlConfig)
Base = declarative_base()
SessionLocal = sessionmaker(autoflush=False, bind=engine)
db = SessionLocal()
