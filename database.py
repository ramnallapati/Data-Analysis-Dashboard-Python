
import os
import pandas as pd
import json
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pickle
import io

# Get the database URL from environment variables
DATABASE_URL = os.environ.get('DATABASE_URL')

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create a base class for models
Base = declarative_base()

class UploadedDataset(Base):
    """Model for storing uploaded datasets"""
    __tablename__ = 'uploaded_datasets'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    upload_date = Column(DateTime, default=datetime.now)
    data = Column(LargeBinary, nullable=False)  # Store pickled dataframe
    column_types = Column(Text, nullable=False)  # Store column types as JSON
    row_count = Column(Integer, nullable=False)
    column_count = Column(Integer, nullable=False)
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'upload_date': self.upload_date.strftime('%Y-%m-%d %H:%M:%S'),
            'row_count': self.row_count,
            'column_count': self.column_count
        }

class AnalysisResult(Base):
    """Model for storing analysis results"""
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, nullable=False)
    analysis_type = Column(String(100), nullable=False)
    result_data = Column(Text, nullable=False)  # Store result as JSON
    created_at = Column(DateTime, default=datetime.now)
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'dataset_id': self.dataset_id,
            'analysis_type': self.analysis_type,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
        }

# Create all tables
Base.metadata.create_all(engine)

# Create session factory
Session = sessionmaker(bind=engine)

def save_dataframe(name, dataframe, description=None):
    """
    Save a pandas DataFrame to the database
    
    Args:
        name (str): Name of the dataset
        dataframe (pandas.DataFrame): The DataFrame to save
        description (str, optional): Description of the dataset
        
    Returns:
        int: ID of the saved dataset
    """
    # Pickle the dataframe
    pickled_df = pickle.dumps(dataframe)
    
    # Store column types as JSON
    column_types = {col: str(dtype) for col, dtype in dataframe.dtypes.items()}
    
    # Create new record
    session = Session()
    new_dataset = UploadedDataset(
        name=name,
        description=description,
        data=pickled_df,
        column_types=json.dumps(column_types),
        row_count=len(dataframe),
        column_count=len(dataframe.columns)
    )
    
    try:
        session.add(new_dataset)
        session.commit()
        return new_dataset.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_all_datasets():
    """
    Get all datasets from the database
    
    Returns:
        list: List of dataset dictionaries
    """
    session = Session()
    try:
        datasets = session.query(UploadedDataset).all()
        return [dataset.to_dict() for dataset in datasets]
    finally:
        session.close()

def get_dataframe(dataset_id):
    """
    Retrieve a pandas DataFrame from the database
    
    Args:
        dataset_id (int): ID of the dataset to retrieve
        
    Returns:
        pandas.DataFrame: The retrieved DataFrame
    """
    session = Session()
    try:
        dataset = session.query(UploadedDataset).filter_by(id=dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset with ID {dataset_id} not found")
        
        # Unpickle the dataframe
        dataframe = pickle.loads(dataset.data)
        return dataframe
    finally:
        session.close()

def save_analysis_result(dataset_id, analysis_type, result_data):
    """
    Save analysis results to the database
    
    Args:
        dataset_id (int): ID of the associated dataset
        analysis_type (str): Type of analysis performed
        result_data (dict): Results data
        
    Returns:
        int: ID of the saved analysis
    """
    session = Session()
    analysis = AnalysisResult(
        dataset_id=dataset_id,
        analysis_type=analysis_type,
        result_data=json.dumps(result_data)
    )
    
    try:
        session.add(analysis)
        session.commit()
        return analysis.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_analysis_results(dataset_id=None):
    """
    Get analysis results from the database
    
    Args:
        dataset_id (int, optional): Filter by dataset ID
        
    Returns:
        list: List of analysis result dictionaries
    """
    session = Session()
    try:
        query = session.query(AnalysisResult)
        if dataset_id:
            query = query.filter_by(dataset_id=dataset_id)
        
        results = query.all()
        return [result.to_dict() for result in results]
    finally:
        session.close()

def delete_dataset(dataset_id):
    """
    Delete a dataset and its associated analysis results
    
    Args:
        dataset_id (int): ID of the dataset to delete
        
    Returns:
        bool: True if successful
    """
    session = Session()
    try:
        # First delete associated analysis results
        session.query(AnalysisResult).filter_by(dataset_id=dataset_id).delete()
        
        # Then delete the dataset
        dataset = session.query(UploadedDataset).filter_by(id=dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset with ID {dataset_id} not found")
        
        session.delete(dataset)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()