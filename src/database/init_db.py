#!/usr/bin/env python

import os
import logging
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey, Table, Boolean, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import yaml
import datetime

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config():
    config_path = os.path.join(os.path.dirname(
        __file__), '..', '..', 'config', 'database.yml')
    if not os.path.exists(config_path):
        config = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'product_data',
                'user': 'postgres',
                'password': 'postgres'
            }
        }
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        return config

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_engine(config):
    db_config = config['database']
    connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['name']}"
    return create_engine(connection_string)


Base = declarative_base()

product_certification = Table(
    'product_certification', Base.metadata,
    Column('product_id', Integer, ForeignKey('products.id')),
    Column('certification_id', Integer, ForeignKey('certifications.id'))
)

product_attribute = Table(
    'product_attribute', Base.metadata,
    Column('product_id', Integer, ForeignKey('products.id')),
    Column('attribute_id', Integer, ForeignKey('attributes.id'))
)


class Document(Base):
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)
    processed = Column(Boolean, default=False)
    pages = Column(Integer)

    products = relationship("Product", back_populates="document")

    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}')>"


class Manufacturer(Base):
    __tablename__ = 'manufacturers'

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    website = Column(String(255))

    products = relationship("Product", back_populates="manufacturer")

    def __repr__(self):
        return f"<Manufacturer(id={self.id}, name='{self.name}')>"


class Product(Base):
    __tablename__ = 'products'

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, index=True)
    sku = Column(String(100), index=True)
    description = Column(Text)
    document_id = Column(Integer, ForeignKey('documents.id'))
    manufacturer_id = Column(Integer, ForeignKey('manufacturers.id'))
    date_extracted = Column(DateTime, default=datetime.datetime.utcnow)
    confidence_score = Column(Float)

    document = relationship("Document", back_populates="products")
    manufacturer = relationship("Manufacturer", back_populates="products")
    specifications = relationship("Specification", back_populates="product")
    certifications = relationship(
        "Certification", secondary=product_certification)
    attributes = relationship("Attribute", secondary=product_attribute)

    def __repr__(self):
        return f"<Product(id={self.id}, name='{self.name}')>"


class Specification(Base):
    __tablename__ = 'specifications'

    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('products.id'))
    name = Column(String(255), nullable=False)
    value = Column(String(255))
    unit = Column(String(50))

    product = relationship("Product", back_populates="specifications")

    def __repr__(self):
        return f"<Specification(id={self.id}, name='{self.name}', value='{self.value}')>"


class Certification(Base):
    __tablename__ = 'certifications'

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, index=True)
    issuing_body = Column(String(255))
    description = Column(Text)

    def __repr__(self):
        return f"<Certification(id={self.id}, name='{self.name}')>"


class Attribute(Base):
    __tablename__ = 'attributes'

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, index=True)
    value = Column(String(255))

    def __repr__(self):
        return f"<Attribute(id={self.id}, name='{self.name}', value='{self.value}')>"


class ExtractedText(Base):
    __tablename__ = 'extracted_text'

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id'))
    page_number = Column(Integer)
    text = Column(Text)

    def __repr__(self):
        return f"<ExtractedText(id={self.id}, document_id={self.document_id}, page_number={self.page_number})>"


def init_db():
    config = load_config()
    engine = get_engine(config)

    Base.metadata.create_all(engine)

    logger.info("Database schema created successfully.")

    Session = sessionmaker(bind=engine)
    session = Session()

    initial_certs = [
        Certification(name="CE", issuing_body="European Union",
                      description="Conformité Européenne - Indicates conformity with health, safety, and environmental protection standards for products sold within the European Economic Area."),
        Certification(name="UL", issuing_body="Underwriters Laboratories",
                      description="Safety certification for products sold in the United States."),
        Certification(name="RoHS", issuing_body="European Union",
                      description="Restriction of Hazardous Substances Directive."),
        Certification(name="ISO 9001", issuing_body="International Organization for Standardization",
                      description="Quality management systems standard.")
    ]

    for cert in initial_certs:
        session.add(cert)

    session.commit()
    session.close()

    logger.info("Initial data populated.")


if __name__ == "__main__":
    init_db()
