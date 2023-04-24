from sqlalchemy import Integer, Column, DateTime, String
from config_db import Base, engine


class CreatedImagesModel(Base):
    __tablename__ = "created_images"

    id = Column(Integer, primary_key=True)
    creating_date = Column(DateTime)
    name = Column(String)

# Base.metadata.create_all(bind=engine)